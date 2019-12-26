import torch
from torch.nn import Module
from torch import nn

class Flatten(Module):
    def forward(self, input):
        batch, _, _, _ = input.size()
        return input.reshape(batch, -1)