import torch
from torch.nn import Module
from torch import nn

class DepthwiseConv2d(Module):
    """
    Depthwise convolution

    """
    def __init__(self, in_channel, kernel_size, stride=1, padding=0,
                 bias=True, device='cuda'):
        super(DepthwiseConv2d, self).__init__()
        self.in_channel = in_channel
        self.device = device

        # self.depthwise = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride,
        #                             padding=padding, bias=bias).to(self.device) for i in range(self.in_channel)])
        self.depthwise = nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel,
                                   kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=bias, groups=self.in_channel).to(self.device)
        self.bn = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU()

    def forward(self, input):
        """
        Not optimized yet

        :param input: pytorch tensor (batch size, channel, height, width)
        :return: depthwise feature map
        """
        # process each channel seperatly
        assert input.shape[1] == self.in_channel, "input channel and output channel does not match! input : {}, filter : {}".format(input.shape[1], self.in_channel)
        batch_size, channel, height, width = input.shape
        # do depthwise convolution
        # apply conv filter per each input channel
        # not parallelized yet
        # depthwise_feature = []
        # for i in range(self.in_channel):
        #     depthwise_feature.append(self.depthwise[i](input[:,i,:,:].reshape(-1, 1, height, width)))
        #
        # depthwise_feature = torch.cat(dim=1, tensors=depthwise_feature)

        # use group parameter to optimized depthwise conv
        depthwise_feature = self.depthwise(input)
        depthwise_feature = self.bn(depthwise_feature)
        depthwise_feature = self.relu(depthwise_feature)

        return depthwise_feature

class PointwiseConv2d(Module):
    def __init__(self, in_channel, out_channel, bias=True, device='cuda'):
        super(PointwiseConv2d, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.device = device

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0,
                      bias=bias),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU()
        ).to(self.device)

    def forward(self, input):
        """
        Take depthwise feature map, then return pointwise feature map
        Simple 1x1 conv2d operation will work

        :param input: Depthwise fature map
        :return: pointwise feature map
        """
        return self.pointwise(input)

class DepthwiseSaperableConv2D(Module):
    """
    Factorized single traditional convolution operation

    Not optimized
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0,
                 bias=True, device='cuda'):
        super(DepthwiseSaperableConv2D, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.device = device

        self.depthwise = DepthwiseConv2d(self.in_channel, kernel_size, stride=stride,
                                         padding=padding, bias=bias, device=self.device).to(self.device)
        self.pointwise = PointwiseConv2d(self.in_channel, self.out_channel,
                                         bias=bias, device=self.device).to(self.device)

    def forward(self, input):
        input = self.depthwise(input)
        input = self.pointwise(input)
        return input

class MiniMobileNet(Module):
    def __init__(self, n_classes, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(MiniMobileNet, self).__init__()

        self.n_classes = n_classes
        self.device = device

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ).to(self.device)

        self.dw1 = DepthwiseSaperableConv2D(in_channel=32, out_channel=64, kernel_size=3, stride=1, padding=1,
                                            bias=False, device=self.device)
        self.dw2 = DepthwiseSaperableConv2D(in_channel=64, out_channel=128, kernel_size=3, stride=2, padding=1,
                                            bias=False, device=self.device)
        self.dw3 = DepthwiseSaperableConv2D(in_channel=128, out_channel=128, kernel_size=3, stride=1, padding=1,
                                            bias=False, device=self.device)
        self.dw4 = DepthwiseSaperableConv2D(in_channel=128, out_channel=256, kernel_size=3, stride=2, padding=1,
                                            bias=False, device=self.device)
        self.dw5 = DepthwiseSaperableConv2D(in_channel=256, out_channel=256, kernel_size=3, stride=1, padding=1,
                                            bias=False, device=self.device)
        self.dw6 = DepthwiseSaperableConv2D(in_channel=256, out_channel=512, kernel_size=3, stride=1, padding=1,
                                            bias=False, device=self.device)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1)).to(self.device)
        self.linear = nn.Linear(512, self.n_classes).to(self.device)
        self.softmax = nn.Softmax(dim=1).to(self.device)

    def forward(self, input):
        # input shape : b, c, h, w (batch size, channels, height, width)
        input = self.conv1(input)
        # feature map shape : b, 32, h/2, w/2
        input = self.dw1(input)
        # feature map shape : b, 64, h/2, w/2
        input = self.dw2(input)
        # feature map shape : b, 128, h/4, w/4
        input = self.dw3(input)
        # feature map shape : b, 128, h/4, w/4
        input = self.dw4(input)
        # feature map shape : b, 256, h/8, w/8
        input = self.dw5(input)
        # feature map shape : b, 256, h/8, w/8
        input = self.dw6(input)
        # feature map shape : b, 512, h/8, w/8

        input = self.avgpool(input)
        input = input.reshape(input.shape[0], -1)
        # feature map shape : b, 512
        input = self.linear(input)
        # feature map shape : b, n_classes
        input = self.softmax(input)

        return input

class MobileNet(Module):
    def __init__(self, n_classes, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(MobileNet, self).__init__()

        self.n_classes = n_classes
        self.device = device

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ).to(self.device)

        self.dw1 = DepthwiseSaperableConv2D(in_channel=32, out_channel=64, kernel_size=3, stride=1, padding=1,
                                            bias=False, device=self.device)
        self.dw2 = DepthwiseSaperableConv2D(in_channel=64, out_channel=128, kernel_size=3, stride=2, padding=1,
                                            bias=False, device=self.device)
        self.dw3 = DepthwiseSaperableConv2D(in_channel=128, out_channel=128, kernel_size=3, stride=1, padding=1,
                                            bias=False, device=self.device)
        self.dw4 = DepthwiseSaperableConv2D(in_channel=128, out_channel=256, kernel_size=3, stride=2, padding=1,
                                            bias=False, device=self.device)
        self.dw5 = DepthwiseSaperableConv2D(in_channel=256, out_channel=256, kernel_size=3, stride=1, padding=1,
                                            bias=False, device=self.device)
        self.dw6 = DepthwiseSaperableConv2D(in_channel=256, out_channel=512, kernel_size=3, stride=2, padding=1,
                                            bias=False, device=self.device)

        self.dw7 = DepthwiseSaperableConv2D(in_channel=512, out_channel=512, kernel_size=3, stride=1, padding=1,
                                            bias=False, device=self.device)
        self.dw8 = DepthwiseSaperableConv2D(in_channel=512, out_channel=512, kernel_size=3, stride=1, padding=1,
                                            bias=False, device=self.device)
        self.dw9 = DepthwiseSaperableConv2D(in_channel=512, out_channel=512, kernel_size=3, stride=1, padding=1,
                                            bias=False, device=self.device)
        self.dw10 = DepthwiseSaperableConv2D(in_channel=512, out_channel=512, kernel_size=3, stride=1, padding=1,
                                            bias=False, device=self.device)
        self.dw11 = DepthwiseSaperableConv2D(in_channel=512, out_channel=512, kernel_size=3, stride=1, padding=1,
                                            bias=False, device=self.device)

        self.dw12 = DepthwiseSaperableConv2D(in_channel=512, out_channel=1024, kernel_size=3, stride=2, padding=1,
                                            bias=False, device=self.device)
        self.dw13 = DepthwiseSaperableConv2D(in_channel=1024, out_channel=1024, kernel_size=3, stride=1, padding=1,
                                             bias=False, device=self.device)

        self.avgpool = nn.AvgPool2d(kernel_size=7).to(self.device)
        self.linear = nn.Linear(1024, self.n_classes).to(self.device)
        self.softmax = nn.Softmax(dim=1).to(self.device)

    def forward(self, input):
        # input shape : b, c, h, w (batch size, channels, height, width)
        input = self.conv1(input)
        # feature map shape : b, 32, h/2, w/2
        input = self.dw1(input)
        # feature map shape : b, 64, h/2, w/2
        input = self.dw2(input)
        # feature map shape : b, 128, h/4, w/4
        input = self.dw3(input)
        # feature map shape : b, 128, h/4, w/4
        input = self.dw4(input)
        # feature map shape : b, 256, h/8, w/8
        input = self.dw5(input)
        # feature map shape : b, 256, h/8, w/8
        input = self.dw6(input)
        # feature map shape : b, 512, h/16, w/16
        input = self.dw7(input)
        # feature map shape : b, 512, h/16, w/16
        input = self.dw8(input)
        # feature map shape : b, 512, h/16, w/16
        input = self.dw9(input)
        # feature map shape : b, 512, h/16, w/16
        input = self.dw10(input)
        # feature map shape : b, 512, h/16, w/16
        input = self.dw11(input)
        # feature map shape : b, 512, h/16, w/16
        input = self.dw12(input)
        # feature map shape : b, 1024, h/32, w/32
        input = self.dw13(input)
        # feature map shape : b, 1024, h/32, w/32
        input = self.avgpool(input)
        input = input.reshape(input.shape[0], -1)
        # feature map shape : b, 1024
        input = self.linear(input)
        # feature map shape : b, n_classes
        input = self.softmax(input)

        return input

if __name__ == '__main__':
    # cuda test done
    print("test")
    device = torch.device('cuda')

    test_in = torch.randn(size=(10,3,64,64)).to(device)
    test_target = torch.tensor([0,1,1,2,1,4,3,4,2,0]).to(device)

    model = MiniMobileNet(10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters())

    for name, param in model.named_parameters():
        print(name)


    out = model(test_in)
    loss = criterion(out, test_target)
    print(loss)

    loss.backward()
    optimizer.step()
