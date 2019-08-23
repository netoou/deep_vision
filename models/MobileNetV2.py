import torch

from torch import nn
from torch.nn import Module

from models.MoblieNet import DepthwiseConv2d

a = DepthwiseConv2d(1,1,1)
a.relu = nn.ReLU6()

class BottleneckBlock(Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0,
                 bias=True, device='cuda', expension_ratio=1, residual=True):
        super(BottleneckBlock, self).__init__()
        self.in_channel = in_channel
        self.device = device
        self.middle_expension = expension_ratio * in_channel
        self.out_channel = out_channel
        self.residual = residual
        # Define 1x1 conv2d for expension feature map
        self.expension = nn.Sequential(
            nn.Conv2d(self.in_channel, self.middle_expension, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.middle_expension),
            nn.ReLU6()
        ).to(self.device)

        # Define depthwise
        self.depthwise = DepthwiseConv2d(self.middle_expension, kernel_size=kernel_size,
                                         stride=stride, padding=padding, bias=bias, device=self.device)
        self.depthwise.relu = nn.ReLU6() # might not be supported cuda on this code ????????? check and fix it

        # define bottleneck
        self.linear_bottleneck = nn.Conv2d(self.middle_expension, self.out_channel,
                                           kernel_size=1, stride=1, padding=0, bias=bias).to(self.device)

    def forward(self, input):
        """
        Bottleneck Block
        :param input:
        :return:
        """
        feature_map = self.expension(input)

        feature_map = self.depthwise(feature_map)

        feature_map = self.linear_bottleneck(feature_map)

        if self.residual:
            feature_map += input

        return feature_map


class MobileNetV2(Module):
    def __init__(self, n_classes, device):
        super(MobileNetV2, self).__init__()
        self.n_classes = n_classes
        self.conv2d = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )
        # 32 to 16 channels
        self.bottleneck1 = BottleneckBlock(32, 16, 3, stride=1, padding=1, device=device, expension_ratio=1, residual=False)
        # 16 to 24 and 24 to 24 channels
        self.bottleneck2 = BottleneckBlock(16, 24, 3, stride=2, padding=1, device=device, expension_ratio=6, residual=False)
        self.bottleneck3 = BottleneckBlock(24, 24, 3, stride=1, padding=1, device=device, expension_ratio=6, residual=True)
        # 24 to 32 and 32 to 32 then 32 to 32 channels
        self.bottleneck4 = BottleneckBlock(24, 32, 3, stride=2, padding=1, device=device, expension_ratio=6, residual=False)
        self.bottleneck5 = BottleneckBlock(32, 32, 3, stride=1, padding=1, device=device, expension_ratio=6, residual=True)
        self.bottleneck6 = BottleneckBlock(32, 32, 3, stride=1, padding=1, device=device, expension_ratio=6, residual=True)
        # 32 to 64 and 64 to 64 * 3 channels
        self.bottleneck7 = BottleneckBlock(32, 64, 3, stride=2, padding=1, device=device, expension_ratio=6,
                                           residual=False)
        self.bottleneck8 = BottleneckBlock(64, 64, 3, stride=1, padding=1, device=device, expension_ratio=6,
                                           residual=True)
        self.bottleneck9 = BottleneckBlock(64, 64, 3, stride=1, padding=1, device=device, expension_ratio=6,
                                           residual=True)
        self.bottleneck10 = BottleneckBlock(64, 64, 3, stride=1, padding=1, device=device, expension_ratio=6,
                                           residual=True)
        # 64 to 96 and 96 to 96 then 96 to 96 channels
        self.bottleneck11 = BottleneckBlock(64, 96, 3, stride=1, padding=1, device=device, expension_ratio=6,
                                           residual=False)
        self.bottleneck12 = BottleneckBlock(96, 96, 3, stride=1, padding=1, device=device, expension_ratio=6,
                                           residual=True)
        self.bottleneck13 = BottleneckBlock(96, 96, 3, stride=1, padding=1, device=device, expension_ratio=6,
                                           residual=True)

        # 96 to 160 and 160 to 160 then 160 to 160 channels
        self.bottleneck14 = BottleneckBlock(96, 160, 3, stride=2, padding=1, device=device, expension_ratio=6,
                                            residual=False)
        self.bottleneck15 = BottleneckBlock(160, 160, 3, stride=1, padding=1, device=device, expension_ratio=6,
                                            residual=True)
        self.bottleneck16 = BottleneckBlock(160, 160, 3, stride=1, padding=1, device=device, expension_ratio=6,
                                            residual=True)
        # 160 to 320
        self.bottleneck17 = BottleneckBlock(160, 320, 3, stride=1, padding=1, device=device, expension_ratio=6,
                                            residual=False)
        self.conv2d1x1_1 = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1),
            nn.BatchNorm2d(1280),
            nn.ReLU6()
        )

        self.avgpool = nn.AvgPool2d(kernel_size=7)
        self.conv2d1x1_2 = nn.Conv2d(1280, self.n_classes, kernel_size=1)

    def forward(self, input):
        input = self.conv2d(input)
        input = self.bottleneck1(input)
        input = self.bottleneck2(input)
        input = self.bottleneck3(input)
        input = self.bottleneck4(input)
        input = self.bottleneck5(input)
        input = self.bottleneck6(input)
        input = self.bottleneck7(input)
        input = self.bottleneck8(input)
        input = self.bottleneck9(input)
        input = self.bottleneck10(input)
        input = self.bottleneck11(input)
        input = self.bottleneck12(input)
        input = self.bottleneck13(input)
        input = self.bottleneck14(input)
        input = self.bottleneck15(input)
        input = self.bottleneck16(input)
        input = self.bottleneck17(input)
        input = self.conv2d1x1_1(input)
        input = self.avgpool(input)
        input = self.conv2d1x1_2(input)
        input = input.reshape(-1, self.n_classes)

        return input



if __name__ == "__main__":
    device = torch.device("cuda")
    #bottle = BottleneckBlock(4,8,3,device=device).to(device)
    model = MobileNetV2(10, device=device).to(device)
    in_feature = torch.randn(size=(1,3,224,224)).to(device)

    out = model(in_feature)

    print(out.shape)