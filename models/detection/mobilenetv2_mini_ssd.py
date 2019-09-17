from models.classification.MobileNetV2 import BottleneckBlock
from models.classification.MoblieNet import DepthwiseSaperableConv2D
from models.detection.ssd import SSD
from torch.nn import Module
from torch import nn
import torch

class Baseline_net(Module):
    def __init__(self, expension_ratio=6):
        super(Baseline_net, self).__init__()
        self.expension_ratio = expension_ratio
        self.conv2d = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )
        self.bottleneck1 = BottleneckBlock(32, 32, 3, stride=1, padding=1,
                                           expension_ratio=1, residual=True)
        # 32 -> 64
        self.bottleneck2 = BottleneckBlock(32, 64, 3, stride=2, padding=1,
                                           expension_ratio=self.expension_ratio, residual=False)
        self.bottleneck3 = BottleneckBlock(64, 64, 3, stride=1, padding=1,
                                           expension_ratio=self.expension_ratio, residual=True)
        # 64 -> 96
        self.bottleneck4 = BottleneckBlock(64, 96, 3, stride=2, padding=1,
                                           expension_ratio=self.expension_ratio, residual=False)
        self.bottleneck5 = BottleneckBlock(96, 96, 3, stride=1, padding=1,
                                           expension_ratio=self.expension_ratio, residual=True)

    def forward(self, input):
        # feature map size : * 1/2
        input = self.conv2d(input)
        input = self.bottleneck1(input)
        # feature map size : * 1/4
        input = self.bottleneck2(input)
        input = self.bottleneck3(input)
        # feature map size : * 1/8
        input = self.bottleneck4(input)
        input = self.bottleneck5(input)
        return input

def depthwise_seperable(in_channel, out_channel, kernel_size,
                        stride=1, padding=0, bias=True, bn=True):
    conv = nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size,
                  stride=stride, padding=padding, groups=in_channel, bias=bias),
        nn.BatchNorm2d(in_channel),
        nn.ReLU6(),
        nn.Conv2d(in_channel, out_channels=out_channel, kernel_size=1),
    )
    return conv

def last_depthwise_seperable(in_channel, out_channel, kernel_size,
                        stride=1, padding=0, bias=True, bn=True):
    conv = nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size,
                  stride=stride, padding=padding, groups=in_channel, bias=bias),
        nn.ReLU6(),
        nn.Conv2d(in_channel, out_channels=out_channel, kernel_size=1),
    )
    return conv


def mobilenetv2_mini_ssd(n_classes: int, input_size: tuple, expension_ratio=6) -> SSD:
    base_net = Baseline_net(expension_ratio)
    extra_layers = nn.ModuleList([
        # 12x12
        BottleneckBlock(96, 96, 3, stride=2, padding=1,
                        expension_ratio=expension_ratio, residual=False),
        # 6x6
        BottleneckBlock(96, 96, 3, stride=2, padding=1,
                        expension_ratio=expension_ratio, residual=False),
        # 3x3
        BottleneckBlock(96, 96, 3, stride=2, padding=1,
                        expension_ratio=expension_ratio, residual=False),
    ])
    pred_layers = nn.ModuleList([
        depthwise_seperable(96, 4 * (n_classes + 4), kernel_size=3, stride=1, padding=1),
        depthwise_seperable(96, 4 * (n_classes + 4), kernel_size=3, stride=1, padding=1),
        depthwise_seperable(96, 4 * (n_classes + 4), kernel_size=3, stride=1, padding=1),
        last_depthwise_seperable(96, 4 * (n_classes + 4), kernel_size=3, stride=1, padding=1),
    ])
    return SSD(n_classes, input_size, base_net, extra_layers, pred_layers)

if __name__=='__main__':
    model = mobilenetv2_mini_ssd(21, (192, 192)).to('cuda')
    # print(model.default_anchors.shape)
    # print(model.default_anchors.max(), model.default_anchors.min())
    # print(model.reduction_ratio)
    # print(model.featuremap_sizes)
    # print(model.default_anchors[:10])
    sample_in = torch.randn(1,3,192,192).to('cuda')
    cls, reg = model(sample_in)

    print(cls.shape)
    print(reg.shape)
    print(reg[0][:10])

