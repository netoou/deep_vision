import torch
from torch import nn
from torch.nn import Module

from models.classification.MoblieNet import DepthwiseConv2d
from models.classification.SqueezeExcitation import SEblock
from models.classification.util_module import Flatten, parameter_calculator, HSwish


class MBv3SE(Module):
    def __init__(self, channel, reduction_ratio=8, bias=False):
        super(MBv3SE, self).__init__()
        assert channel > reduction_ratio, 'Too small channels, reduce reduction ratio.'

        self.se = nn.Sequential(
            nn.AdaptiveMaxPool2d((1,1)),
            Flatten(),
            nn.Linear(channel, int(channel // reduction_ratio), bias=bias),
            nn.ReLU(),
            nn.Linear(int(channel // reduction_ratio), channel, bias=bias),
            HSwish(),
        )

    def forward(self, input):
        batch, channel, w, h = input.size()
        sqz = self.se(input)
        return input * sqz.reshape(batch, channel, 1, 1)


class MBlockv3(Module):
    # Mobilenet-v2 + Squeeze-and-Excitation
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0,
                 bias=False, expension_ratio=6, reduction_ratio=8, nl=nn.ReLU6, se=True):
        super(MBlockv3, self).__init__()
        # nl : non-linearity
        self.in_channel = in_channel
        self.middle_expension = int(expension_ratio * in_channel)
        self.out_channel = out_channel

        self.se = se
        self.residual = True if stride == 1 and in_channel == out_channel else False

        # Define 1x1 conv2d for expension feature map
        self.expension = nn.Sequential(
            nn.Conv2d(self.in_channel, self.middle_expension, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.middle_expension),
            nl(),
        )

        # Define depthwise
        self.depthwise = DepthwiseConv2d(self.middle_expension, kernel_size=kernel_size,
                                         stride=stride, padding=padding, bias=bias, activation=nl)

        self.se = MBv3SE(self.middle_expension, reduction_ratio)

        # define bottleneck
        self.linear_bottleneck = nn.Sequential(
            nn.Conv2d(self.middle_expension, self.out_channel, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(self.out_channel),
            nl(),
        )

    def forward(self, input):
        feature = self.expension(input)
        feature = self.depthwise(feature)
        if self.se:
            feature = self.se(feature)
        feature = self.linear_bottleneck(feature)
        if self.residual:
            feature += input
        return feature


class LastStage(Module):
    def __init__(self, n_classes, in_channel, mid_channel, out_channel):
        super(LastStage, self).__init__()

        self.expension_layer = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, 1, bias=False),
            nn.BatchNorm2d(mid_channel),
            HSwish(),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(mid_channel, mid_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            HSwish(),
        )

        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(mid_channel, mid_channel / 3, 1, bias=False),
            nn.BatchNorm2d(mid_channel / 3),
        )

        self.bottleneck2 = nn.Sequential(
            nn.Conv2d(mid_channel / 3, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            HSwish(),
        )

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channel, n_classes, 1, bias=False),
            Flatten(),
        )

    def forward(self, input):
        input = self.expension_layer(input)
        input = self.conv(input)
        input = self.bottleneck1(input)
        input = self.bottleneck2(input)
        input = self.fc(input)
        return input


class EfficientLastStage(Module):
    def __init__(self, n_classes, in_channel, mid_channel, out_channel):
        super(EfficientLastStage, self).__init__()

        self.expension_layer = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, 1, bias=False),
            nn.BatchNorm2d(mid_channel),
            HSwish(),
        )

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channel, out_channel, 1, bias=False),
            HSwish(),
            nn.Conv2d(out_channel, n_classes, 1, bias=False),
            Flatten(),
        )

    def forward(self, input):
        input = self.expension_layer(input)
        input = self.fc(input)
        return input


class MobileNetV3(Module):
    def __init__(self, n_classes, arc: dict):
        super(MobileNetV3, self).__init__()
        self.arc = arc
        self.n_classes = n_classes

        self.network = self._build()

    def forward(self, input):
        return self.network(input)

    def _build(self):
        stages = list(self.arc)
        last_stage = stages.pop(-1)
        layers = []
        # conv block
        layers.append(
            nn.Sequential(
                nn.Conv2d(3, 16, 3, 2, 1, bias=False),
                nn.BatchNorm2d(16),
                HSwish(),
            )
        )
        in_channel = 16
        # Mobile bottlenect v3
        for stage in stages:
            expension, kernel_size, stride, out_channel, se, nl = self.arc[stage]
            layers.append(
                MBlockv3(in_channel, out_channel, kernel_size, stride, kernel_size // 2,
                         expension_ratio=expension, nl=nl, se=se)
            )
            in_channel = out_channel

        mid_channel, out_channel = self.arc[last_stage]
        layers.append(
            EfficientLastStage(self.n_classes, in_channel, mid_channel, out_channel)
        )

        return nn.Sequential(*layers)


mbv3_large_dict = {  # expension_ratio, kernel_size, stride, n_channel, se, nl
    'stage1': (1, 3, 1, 16, False, nn.ReLU),
    'stage2': (4, 3, 2, 24, False, nn.ReLU),
    'stage3': (3, 3, 1, 24, False, nn.ReLU),
    'stage4': (3, 5, 2, 40, True, nn.ReLU),
    'stage5': (3, 5, 1, 40, True, nn.ReLU),
    'stage6': (3, 5, 1, 40, True, nn.ReLU),
    'stage7': (6, 3, 2, 80, False, HSwish),
    'stage8': (2.5, 3, 1, 80, False, HSwish),
    'stage9': (2.3, 3, 1, 80, False, HSwish),
    'stage10': (2.3, 3, 1, 80, False, HSwish),
    'stage11': (6, 3, 1, 112, True, HSwish),
    'stage12': (6, 3, 1, 112, True, HSwish),
    'stage13': (6, 5, 2, 160, True, HSwish),
    'stage14': (6, 5, 1, 160, True, HSwish),
    'stage15': (6, 5, 1, 160, True, HSwish),
    'last': (960, 1280),  # mid channel, out channel
}

mbv3_small_dict = {  # expension_ratio, kernel_size, stride, n_channel, se, nl
    'stage1': (1, 3, 2, 16, True, nn.ReLU),
    'stage2': (6, 3, 2, 24, False, nn.ReLU),
    'stage3': (3.6, 3, 1, 24, False, nn.ReLU),
    'stage4': (4, 5, 2, 40, True, nn.ReLU),
    'stage5': (6, 5, 1, 40, True, HSwish),
    'stage6': (6, 5, 1, 40, True, HSwish),
    'stage7': (3, 5, 1, 48, True, HSwish),
    'stage8': (3, 5, 1, 48, True, HSwish),
    'stage9': (6, 5, 2, 96, True, HSwish),
    'stage10': (6, 5, 1, 96, True, HSwish),
    'stage11': (6, 5, 1, 96, True, HSwish),
    'last': (576, 1280),  # mid channel, out channel
}

mbv3_small2_dict = {  # expension_ratio, kernel_size, stride, n_channel, se, nl
    'stage1': (1, 3, 1, 16, True, nn.ReLU),
    'stage2': (6, 3, 2, 24, False, nn.ReLU),
    'stage4': (4, 5, 1, 40, True, nn.ReLU),
    'stage5': (6, 5, 1, 40, True, HSwish),
    'stage8': (3, 5, 1, 48, True, HSwish),
    'stage9': (6, 5, 2, 96, True, HSwish),
    'stage10': (6, 5, 1, 96, True, HSwish),
    'stage11': (6, 5, 1, 96, True, HSwish),
    'last': (96*4, 96*5),  # mid channel, out channel
}

def mobilenet_v3(n_classes, arc='small'):
    assert arc in ['small', 'large', 'small2']
    if arc == 'small':
        model_arc = mbv3_small_dict
    elif arc == 'large':
        model_arc = mbv3_large_dict
    elif arc == 'small2':
        model_arc = mbv3_small2_dict
    model = MobileNetV3(n_classes, model_arc)
    return model

if __name__=='__main__':
    model = mobilenet_v3(100, 'small')
    print(parameter_calculator(model))

