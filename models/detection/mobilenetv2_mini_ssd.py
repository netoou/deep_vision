from models.classification.MobileNetV2 import BottleneckBlock
from models.classification.MoblieNet import DepthwiseSaperableConv2D
from models.detection.ssd import SSD, ssd_box_matching, multiboxLoss, ssd_box_matching_batch, multiboxLoss_batch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch import nn
import torch
import numpy as np

import sys

from datasets.voc2012 import VOCDataset

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
    dset = VOCDataset('/home/ailab/data/', input_size=(192, 192))
    model = mobilenetv2_mini_ssd(21, (192, 192)).to('cuda')

    dload = DataLoader(dset, batch_size=3)
    diter = iter(dload)
    img, objs = next(diter)

    df_anc = model.default_anchors
    matching = ssd_box_matching_batch(df_anc, objs)
    cls_score, reg_coord = model(img.to('cuda'))
    print('-'*40)
    print(matching.shape)
    print(cls_score.shape)
    print(reg_coord.shape)
    print(objs.shape)
    # print(np.argwhere(res==True)) # np.argwhere로 out boxes의 위치와 해당 gt_box의 idx를 알아내자
    print(multiboxLoss_batch(matching, cls_score, reg_coord, objs.to('cuda')))
    torch.cuda.empty_cache()
    sys.exit()