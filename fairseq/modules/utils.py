import torch
from torch import nn as nn


class Transpose(nn.Module):

    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class Permute3D(nn.Module):

    def __init__(self, dim0, dim1, dim2):
        super(Permute3D, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.permute(self.dim0, self.dim1, self.dim2)
