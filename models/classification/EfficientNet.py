import torch
from torch import nn
from torch.nn import Module

from models.classification.MoblieNet import DepthwiseConv2d
from models.classification.SqueezeExcitation import SEblock
from models.classification.util_module import drop_connection, Flatten

import math

# mobile inverted residual bottleneck(MobilenetV2 with SEblock~~~)


class ConvBlock(Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding=(kernel_size // 2), bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(),
        )

    def forward(self, input):
        return self.conv(input)


class MBConv(Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, expension_ratio=6, se_ratio=8, residual=True, drop_connect=0.3):
        super(MBConv, self).__init__()
        self.residual = residual if stride == 1 and in_channel == out_channel else False
        self.drop_connect = drop_connect

        self.expension_layer = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * expension_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channel * expension_ratio),
            nn.ReLU6()
        )

        self.depthwise_layer = DepthwiseConv2d(in_channel * expension_ratio, kernel_size, stride=stride,
                                               padding=(kernel_size // 2), bias=False, activation=nn.ReLU6)

        self.se_block = SEblock(in_channel * expension_ratio, se_ratio, activation=nn.ReLU6)

        self.linear_bottleneck = nn.Conv2d(in_channel * expension_ratio, out_channel, 1, bias=False)

    def forward(self, input):

        feature_map = self.expension_layer(input)
        feature_map = self.depthwise_layer(feature_map)
        feature_map = self.se_block(feature_map)
        feature_map = self.linear_bottleneck(feature_map)

        if self.residual:
            if self.drop_connect:
                feature_map += drop_connection(input, self.training, self.drop_connect)
            else:
                feature_map += input

        return feature_map


class FCBlock(Module):
    def __init__(self, in_channel, mid_channel, n_classes, dropout=0.3):
        super(FCBlock, self).__init__()
        self.linear_expansion = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, 1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU6(),
        )

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Dropout(dropout),
            nn.Linear(mid_channel, n_classes),
        )

    def forward(self, input):
        input = self.linear_expansion(input)
        return self.fc(input)



class EfficientNet(Module):
    def __init__(self, n_classes, depth_scale, width_scale, dataset='cifar'):
        super(EfficientNet, self).__init__()
        # EfficientNet-B0 baseline network architecture
        self.depth_scale = depth_scale
        self.width_scale = width_scale
        self.n_classes = n_classes
        if dataset == 'cifar':
            self.baseline_arc = {  # block, bottleneck expension, kernel_size, stride, n_chennel, n_layers
                'stage1': (ConvBlock, None, 3, 1, 32, None),
                'stage2': (MBConv, 1, 3, 1, 16, 1),
                'stage3': (MBConv, 6, 3, 1, 24, 2),
                'stage4': (MBConv, 6, 5, 2, 40, 2),
                'stage5': (MBConv, 6, 3, 2, 80, 3),
                'stage6': (MBConv, 6, 5, 1, 112, 3),
                'stage7': (MBConv, 6, 5, 1, 192, 4),
                'stage8': (MBConv, 6, 3, 2, 320, 1),
                'stage9': (FCBlock, None, None, None, 1280, None),
            }
        else:
            self.baseline_arc = {  # block, bottleneck expension, kernel_size, stride, n_chennel, n_layers
                'stage1': (ConvBlock, None, 3, 2, 32, None),
                'stage2': (MBConv, 1, 3, 1, 16, 1),
                'stage3': (MBConv, 6, 3, 1, 24, 2),
                'stage4': (MBConv, 6, 5, 2, 40, 2),
                'stage5': (MBConv, 6, 3, 2, 80, 3),
                'stage6': (MBConv, 6, 5, 2, 112, 3),
                'stage7': (MBConv, 6, 5, 1, 192, 4),
                'stage8': (MBConv, 6, 3, 2, 320, 1),
                'stage9': (FCBlock, None, None, None, 1280, None),
            }

        self.net = self._build_model()

    def forward(self, input):
        return self.net(input)

    def _build_model(self):
        nn_layers = []
        stages = list(self.baseline_arc.keys())
        in_channel = 3
        for stage in stages:
            if stage == stages[0]:  # first stage
                block, _, kernel_size, stride, n_channel, _ = self.baseline_arc[stage]
                _, n_channel = self._compound_scaling(width=n_channel)
                nn_layers.append(block(in_channel, n_channel, kernel_size, stride))
                in_channel = n_channel

            elif stage == stages[-1]:  # last stage
                block, _, _, _, n_channel, _ = self.baseline_arc[stage]
                _, n_channel = self._compound_scaling(width=n_channel)
                nn_layers.append(block(in_channel, n_channel, self.n_classes))

            else:  # MB stages
                block, expension_ratio, kernel_size, stride, n_channel, n_layers = self.baseline_arc[stage]
                n_layers, n_channel = self._compound_scaling(depth=n_layers, width=n_channel)
                for layer in range(n_layers):
                    nn_layers.append(block(in_channel, n_channel, kernel_size,
                                           stride if layer == 0 else 1, expension_ratio))
                    in_channel = n_channel

        return nn.Sequential(*nn_layers)

    def _compound_scaling(self, depth=0, width=0):
        return int(math.ceil(depth * self.depth_scale)), int(math.ceil(width * self.width_scale))

if __name__=='__main__':
    nt = EfficientNet(10, 1, 1)
    print(nt)