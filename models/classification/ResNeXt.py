import torch
from torch import nn
from torch.nn import Module

from models.classification.util_module import conv_bn_act, drop_connection, parameter_calculator, Flatten
from models.classification.MobileNetV3 import EfficientLastStage


class ResNeXtBlock(Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, block_width=4,
                 cardinality=32, activation=nn.ReLU6, drop_connect=0):
        super(ResNeXtBlock, self).__init__()

        self.residual = True if stride == 1 and in_channel == out_channel else False
        self.drop_connect = drop_connect

        mid_channel = int(cardinality * block_width)
        self.conv1 = conv_bn_act(in_channel, mid_channel, 1, 1, 0, bias=False, activation=activation)
        self.group_conv = conv_bn_act(mid_channel, mid_channel, kernel_size, stride, kernel_size // 2,
                                      groups=cardinality, bias=False, activation=activation)
        self.bottleneck = conv_bn_act(mid_channel, out_channel, 1, 1, 0, bias=False, activation=activation)

    def forward(self, input):
        feature = self.conv1(input)
        feature = self.group_conv(feature)
        feature = self.bottleneck(feature)

        if self.residual:
            if self.drop_connect:
                feature += drop_connection(input, self.training, self.drop_connect)
            else:
                feature += input

        return feature


class ResNeXt(Module):
    def __init__(self, n_classes, arc, block_width, efficient_last_stage=False):
        super(ResNeXt, self).__init__()
        self.n_classes = n_classes
        self.arc = arc
        self.block_width = block_width
        self.efficient_last_stage = efficient_last_stage

        self.model = self._build_model()

    def forward(self, input):
        return self.model(input)

    def _build_model(self):
        layers = []
        stages = list(self.arc.keys())
        last_stage = stages.pop(-1)
        layers.append(
            nn.Sequential(
                conv_bn_act(3, 64, 7, 2, 7 // 2, bias=False),
                nn.MaxPool2d(3, 2),
            )
        )
        b_width = int(self.block_width / 2)
        in_channel = 64
        for stage in stages:
            b_width = 2 * b_width
            kernel_size, stride, out_channel, cardinality, n_blocks = self.arc[stage]
            for i in range(n_blocks):
                layers.append(ResNeXtBlock(in_channel, out_channel, kernel_size, stride if i == 0 else 1,
                                           b_width, cardinality))
                in_channel = out_channel

        # use efficient last stage
        if self.efficient_last_stage:
            mid_channel, out_channel = self.arc[last_stage]
            layers.append(
                EfficientLastStage(self.n_classes, in_channel, mid_channel, out_channel)
            )
        else:
            layers.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    Flatten(),
                    nn.Linear(in_channel, self.n_classes),
                )
            )

        return nn.Sequential(*layers)

def resnext(n_classes, arch):
    assert arch in ['resnext-mini-16-4d', 'resnext50-32-4d']

    if arch == 'resnext-mini-16-4d':
        model = ResNeXt(n_classes, resnext_mini_16_4d_arch, 4, True)
    elif arch == 'resnext50-32-4d':
        model = ResNeXt(n_classes, resnext50_32_4d_arch, 4, False)

    return model


resnext50_32_4d_arch = {  # kernel_size, stride, out_channel, cardinality, n_blocks
    'stage1': (3, 1, 256, 32, 3),
    'stage2': (3, 2, 512, 32, 4),
    'stage3': (3, 2, 1024, 32, 6),
    'stage4': (3, 2, 2048, 32, 3),
    'last_stage': (1640, 1280),  # mid_channel, out_channel
}

resnext_mini_16_4d_arch = {  # kernel_size, stride, out_channel, cardinality, n_blocks
    'stage1': (3, 1, 128, 16, 2),
    'stage2': (3, 2, 256, 16, 3),
    'stage3': (3, 1, 128*3, 16, 3),
    'stage4': (3, 1, 512, 16, 3),
    'last_stage': (840, 1280),
}


if __name__=='__main__':
    #model = ResNeXt(1000, resnext50_32_4d_arch, 4, False)
    model = ResNeXt(100, resnext_mini_16_4d_arch, 4, True)
    print(model)
    # t_inp = torch.rand((4,3,64,64))
    # print(model(t_inp).shape)

