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