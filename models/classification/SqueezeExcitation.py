import torch
from torch.nn import Module
from torch import nn

from models.classification.util_module import Flatten

class SEblock(Module):
    def __init__(self, channel, reduction_ratio=8, activation=None, bias=False):
        super(SEblock, self).__init__()
        if activation == None:
            activation = nn.ReLU(inplace=True)

        self.se = nn.Sequential(
            nn.AdaptiveMaxPool2d((1,1)),
            Flatten(),
            nn.Linear(channel, channel // reduction_ratio, bias=bias),
            activation,
            nn.Linear(channel // reduction_ratio, channel, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, input):
        batch, channel, w, h = input.size()
        sqz = self.se(input)
        return input * sqz.reshape(batch, channel, 1, 1)


if __name__=='__main__':
    inp = torch.zeros((4, 256, 16, 16))
    se = SEblock(256, 8)
    se(inp)