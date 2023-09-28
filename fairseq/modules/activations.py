import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable


def get_activation_fn(activation: str) -> Callable:
    """ Returns the activation function corresponding to `activation` """
    from fairseq.modules import gelu, gelu_accurate

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_fast":
        return gelu_accurate
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "swish":
        return torch.nn.SiLU
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


def get_activation_class(activation: str, dim=None):
    """ Returns the activation function corresponding to `activation` """

    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "glu":
        assert dim is not None
        return nn.GLU(dim=dim)
    elif activation == "swish":
        return Swish()
    elif activation == "none":
        return nn.Identity()
    else:
        raise RuntimeError("activation function {} not supported".format(activation))


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * x.sigmoid()
