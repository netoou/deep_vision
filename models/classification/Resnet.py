from torch.nn import Module
from torch import nn
import torch


class ResBlock(Module):
    def __init__(self, in_channel, out_channel, stride=1, downsampling=False, bn=False, activation=nn.ReLU):
        super(ResBlock, self).__init__()

        self.bn = bn
        self.downsampling = downsampling

        self.conv_1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2 if downsampling else stride, padding=1)
        self.activation1 = activation(inplace=True)
        self.conv_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.activation2 = activation(inplace=True)

        self.conv_down = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2, bias=False)
        if self.bn:
            self.batchnorm1 = nn.BatchNorm2d(out_channel)
            self.batchnorm2 = nn.BatchNorm2d(out_channel)

    def forward(self, inputs):
        res_out = inputs
        out = self.conv_1(inputs)
        if self.bn:
            out = self.batchnorm1(out)
        out = self.activation1(out)

        out = self.conv_2(out)
        if self.bn:
            out = self.batchnorm1(out)

        if self.downsampling:
            res_out = self.conv_down(res_out)

        out = out + res_out
        out = self.activation2(out)

        return out


class ResBottleneck(Module):
    def __init__(self, in_channel, mid_channel, expensions=4, stride=1,
                 downsampling=False, bn=False, activation=nn.ReLU):
        super(ResBottleneck, self).__init__()

        self.bn = bn
        self.downsampling = downsampling

        self.conv1x1 = nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=stride, bias=False)
        self.activation1 = activation(inplace=False)

        self.conv3x3 = nn.Conv2d(mid_channel, mid_channel, kernel_size=3,
                                 stride=2 if downsampling else stride, padding=1, bias=False)
        self.activation2 = activation(inplace=False)

        self.conv_expension = nn.Conv2d(mid_channel, expensions * mid_channel, kernel_size=1, stride=stride, bias=False)
        self.activation3 = activation(inplace=False)

        self.expension_input = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel * expensions,
                      kernel_size=1, stride=2 if downsampling else stride, bias=False),
            nn.BatchNorm2d(mid_channel * expensions)
        )

        if self.bn:
            self.bn1 = nn.BatchNorm2d(mid_channel)
            self.bn2 = nn.BatchNorm2d(mid_channel)
            self.bn3 = nn.BatchNorm2d(expensions * mid_channel)

    def forward(self, inputs):
        res_out = self.expension_input(inputs)
        out = self.conv1x1(inputs)
        if self.bn:
            out = self.bn1(out)
        out = self.activation1(out)

        out = self.conv3x3(out)
        if self.bn:
            out = self.bn2(out)
        out = self.activation2(out)

        out = self.conv_expension(out)
        if self.bn:
            out = self.bn3(out)

        out = out + res_out
        out = self.activation3(out)
        return out


class Resnet18(Module):
    def __init__(self, n_classes, in_channel=3, bn=True, activation=nn.ReLU):
        super(Resnet18, self).__init__()

        self.n_classes = n_classes
        self.batchnorm = nn.BatchNorm2d(64)
        self.activation = activation(inplace=True)
        self.conv_1 = nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=7, stride=2, padding=3)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_2_1 = ResBlock(64, 64, stride=1, downsampling=False, bn=bn, activation=activation)
        self.conv_2_2 = ResBlock(64, 64, stride=1, downsampling=False, bn=bn, activation=activation)

        self.conv_3_1 = ResBlock(64, 128, stride=2, downsampling=True, bn=bn, activation=activation)
        self.conv_3_2 = ResBlock(128, 128, stride=1, downsampling=False, bn=bn, activation=activation)

        self.conv_4_1 = ResBlock(128, 256, stride=2, downsampling=True, bn=bn, activation=activation)
        self.conv_4_2 = ResBlock(256, 256, stride=1, downsampling=False, bn=bn, activation=activation)

        self.conv_5_1 = ResBlock(256, 512, stride=2, downsampling=True, bn=bn, activation=activation)
        self.conv_5_2 = ResBlock(512, 512, stride=1, downsampling=False, bn=bn, activation=activation)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # batch x 512 x 1 x 1
        self.linear = nn.Linear(512, self.n_classes)

    def forward(self, inputs):
        out = self.conv_1(inputs)
        out = self.batchnorm(out)
        out = self.activation(out)
        out = self.maxpool(out)

        out = self.conv_2_1(out)
        out = self.conv_2_2(out)

        out = self.conv_3_1(out)
        out = self.conv_3_2(out)

        out = self.conv_4_1(out)
        out = self.conv_4_2(out)

        out = self.conv_5_1(out)
        out = self.conv_5_2(out)

        out = self.avg_pool(out)
        # batch x 512 x 1 x 1
        out = out.view(inputs.size(0), -1)
        # batch x 512

        out = self.linear(out)

        return out.reshape((-1))


class Resnet50(Module):
    def __init__(self, n_classes, in_channel=3, bn=True, activation=nn.ReLU):
        """
        Resnet-50 model

        :param n_classes: number of classes of the target dataset
        :param in_channel: number of input image's channel
        :param bn: use batch normalization if True
        :param activation: assign activation you want to use, default : nn.ReLU
        """
        super(Resnet50, self).__init__()

        self.n_classes = n_classes
        self.bn = bn

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=7, stride=2, padding=3)
        if bn:
            self.batchnorm = nn.BatchNorm2d(64)
        self.activation = activation(inplace=True)

        self.conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResBottleneck(64, 64, bn=self.bn, activation=activation),
            ResBottleneck(256, 64, bn=self.bn, activation=activation),
            ResBottleneck(256, 64, bn=self.bn, activation=activation),
        )

        self.conv3_x = nn.Sequential(
            ResBottleneck(256, 128, bn=self.bn, downsampling=True, activation=activation),
            ResBottleneck(512, 128, bn=self.bn, activation=activation),
            ResBottleneck(512, 128, bn=self.bn, activation=activation),
            ResBottleneck(512, 128, bn=self.bn, activation=activation),
        )

        self.conv4_x = nn.Sequential(
            ResBottleneck(512, 256, bn=self.bn, activation=activation, downsampling=True),
            ResBottleneck(1024, 256, bn=self.bn, activation=activation),
            ResBottleneck(1024, 256, bn=self.bn, activation=activation),
            ResBottleneck(1024, 256, bn=self.bn, activation=activation),
            ResBottleneck(1024, 256, bn=self.bn, activation=activation),
            ResBottleneck(1024, 256, bn=self.bn, activation=activation),
        )

        self.conv5_x = nn.Sequential(
            ResBottleneck(1024, 512, bn=self.bn, activation=activation, downsampling=True),
            ResBottleneck(2048, 512, bn=self.bn, activation=activation),
            ResBottleneck(2048, 512, bn=self.bn, activation=activation),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(2048, self.n_classes)

    def forward(self, inputs):
        outs = self.conv1(inputs)
        if self.bn:
            outs = self.batchnorm(outs)
        outs = self.activation(outs)

        outs = self.conv2_x(outs)
        outs = self.conv3_x(outs)
        outs = self.conv4_x(outs)
        outs = self.conv5_x(outs)

        outs = self.avg_pool(outs)
        outs = outs.view(inputs.size(0), -1)
        outs = self.linear(outs)

        return outs

# TODO Compound scaling grid search

if __name__=="__main__":
    aaa = torch.zeros((1, 3, 224, 224))
    model = Resnet50(n_classes=10)

    out = model(aaa)

    print(out.shape)

