import torch
from torch.nn import Module
from torch import nn


class Flatten(Module):
    def forward(self, input):
        batch, _, _, _ = input.size()
        return input.reshape(batch, -1)


class SwishActivation(Module):
    def __init__(self, activation):
        super(SwishActivation, self).__init__()
        self.activation = activation

    def forward(self, input):
        return input * self.activation(input)


class HSwish(Module):
    # For Mobilenet-v3
    def __init__(self):
        super(HSwish, self).__init__()
        self.relu = nn.ReLU6()

    def forward(self, input):
        return input * self.relu(input + 3) / 6


def drop_connection(input, training, drop_ratio):
    if not training:
        return input

    surviving_prob = 1.0 - drop_ratio
    batch_size, _, _, _ = input.size()
    feature = (torch.rand(batch_size, 1, 1, 1) < drop_ratio).to(device=input.device, dtype=input.dtype)

    return (input / surviving_prob) * feature  # tricks to avoid inference time computation of original paper https://arxiv.org/pdf/1603.09382.pdf


def conv_bn_act(in_channels: int, out_channels: int, kernel_size: int, stride = 1, padding = 0,
                 dilation = 1, groups = 1, bias = True, padding_mode = 'zeros', activation = nn.ReLU6):
    cbr = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode),
        nn.BatchNorm2d(out_channels),
        activation()
    )
    return cbr


def parameter_calculator(model: Module):
    return sum(param.numel() for param in model.parameters())


if __name__=='__main__':
    tsr = torch.randn((8, 3, 4, 4))

    a = drop_connection(tsr, True, 0.3)

    print(tsr[:,0,0,0])
    print(a[:,0,0,0])
