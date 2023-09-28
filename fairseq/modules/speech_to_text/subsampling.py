import torch
import torch.nn as nn

from typing import List

from fairseq.modules.activations import Swish
from fairseq.modules.layer_norm import LayerNorm


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


class TransposeLast(nn.Module):

    @staticmethod
    def forward(x):
        return x.transpose(-1, -2).contiguous()


def get_norm(norm_type, size, transpose=False):
    trans = nn.Identity()
    if transpose:
        trans = TransposeLast()
    if norm_type == "batch1d":
        return nn.Sequential(trans, nn.BatchNorm1d(size), trans)
    elif norm_type == "batch2d":
        return nn.Sequential(trans, nn.BatchNorm2d(size), trans)
    elif norm_type == "layer":
        return nn.Sequential(trans, LayerNorm(size), trans)
    elif norm_type == "none":
        return nn.Identity()
    else:
        raise RuntimeError("normalization type {} not supported".format(norm_type))


class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
            self,
            in_channels: int,
            mid_channels: int,
            out_channels: int,
            kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        inner_x = []
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
            inner_x.append(x)
        _, _, out_seq_len = x.size()
        # x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        out_inner_x = []
        for x in inner_x:
            out_inner_x.append(x.transpose(1, 2).transpose(0, 1).contiguous())
        return out_inner_x, self.get_out_seq_lens_tensor(src_lengths)


# fairseq style
class Conv1dSubsampling(nn.Module):
    """Conv1d Subsampling Block

    Args:
        num_layers: number of strided convolution layers
        in_dim: input feature dimension
        filters: list of convolution layers filters
        kernel_size: convolution kernel size
        norm: normalization
        act: activation function

    Shape:
        Input: (batch_size, in_length, in_dim)
        Output: (batch_size, out_length, out_dim)

    """

    def __init__(self, num_layers,
                 in_dim, filters, kernel_size, stride=2,
                 norm="none", act="glu"):
        super(Conv1dSubsampling, self).__init__()

        # Assert
        assert norm in ["batch1d", "layer", "none"]
        assert act in ["relu", "swish", "glu", "none"]

        # Layers
        self.layers = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_dim if layer_id == 0 else filters[layer_id - 1] // 2 if act == "glu" else filters[layer_id - 1],
                      filters[layer_id] * 2 if act == "glu" and layer_id == num_layers - 1 else filters[layer_id],
                      kernel_size,
                      stride=stride,
                      padding=(kernel_size - 1) // 2),
            get_norm(norm,
                     filters[layer_id] * 2 if act == "glu" and layer_id == num_layers - 1 else filters[layer_id],
                     transpose=True if norm == "layer" else False),
            get_activation_class(act, dim=1)
        ) for layer_id in range(num_layers)])

    def forward(self, x, x_len):

        # (T, B, D) -> (B, D, T)
        x = x.permute(1, 2, 0)
        # Layers
        for layer in self.layers:
            x = layer(x)

            # Update Sequence Lengths
            if x_len is not None:
                x_len = torch.div(x_len - 1, 2, rounding_mode='floor') + 1

        # (B, D, T) -> (T, B, D)
        x = x.permute(2, 0, 1)
        return x, x_len


class Conv2dSubsampling(nn.Module):
    """Conv2d Subsampling Block

    Args:
        num_layers: number of strided convolution layers
        filters: list of convolution layers filters
        kernel_size: convolution kernel size
        norm: normalization
        act: activation function

    Shape:
        Input: (in_length, batch_size in_dim)
        Output: (out_length, batch_size, out_dim)

    """

    def __init__(self, num_layers,
                 in_dim, filters, kernel_size, stride=2,
                 norm="none", act="glu"):
        super(Conv2dSubsampling, self).__init__()

        # Assert
        assert norm in ["batch2d", "none"]
        assert act in ["relu", "swish", "glu", "none"]

        # Conv 2D Subsampling Layers

        self.layers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1 if layer_id == 0 else filters[layer_id - 1] // 2 if act == "glu" else filters[layer_id - 1],
                      filters[layer_id] * 2 if act == "glu" and layer_id == num_layers - 1 else filters[layer_id],
                      kernel_size,
                      stride=stride,
                      # padding=(kernel_size - 1) // 2
                      ),
            get_norm(norm,
                     filters[layer_id] * 2 if act == "glu" and layer_id == num_layers - 1 else filters[layer_id],
                     transpose=True if norm == "layer" else False),
            get_activation_class(act, dim=1)
        ) for layer_id in range(num_layers)])

        dim = in_dim
        for _ in range(num_layers):
            dim = (dim - 1) // 2
        self.linear = nn.Linear(dim*filters[-1], filters[-1])

    def forward(self, x, x_len):

        # (T, B, D) -> (B, D, T) -> (B, 1, D, T)
        x = x.permute(1, 2, 0).unsqueeze(dim=1)

        # Layers
        for layer in self.layers:
            x = layer(x)

            # Update Sequence Lengths
            if x_len is not None:
                x_len = torch.div(x_len - 1, 2, rounding_mode='floor')

        # (B, C, D // S, T // S) -> (B,  C * D // S, T // S)
        batch_size, channels, subsampled_dim, subsampled_length = x.size()
        assert subsampled_length == max(x_len), \
            ("The lengths are mismatched: %d and %d." % (subsampled_length, max(x_len)))

        x = x.reshape(batch_size, channels * subsampled_dim, subsampled_length).permute(2, 0, 1)
        x = self.linear(x)

        return x, x_len


def subsampling(args, in_dim=None, out_dim=None):
    subsampling_type = getattr(args, "subsampling_type", "conv1d")
    layers = getattr(args, "subsampling_layers", 2)
    if in_dim is None:
        in_dim = args.input_feat_per_channel * args.input_channels
    filters = [getattr(args, "subsampling_filter")] * (layers - 1) + [args.encoder_embed_dim if out_dim is None else out_dim] 
    kernel_size = getattr(args, "subsampling_kernel", 5)
    stride = getattr(args, "subsampling_stride", 2)
    norm = getattr(args, "subsampling_norm", "none")
    activation = getattr(args, "subsampling_activation", "none")

    if subsampling_type == "conv1d":
        return Conv1dSubsampling(layers, in_dim, filters, kernel_size, stride, norm, activation)
    elif subsampling_type == "conv2d":
        return Conv2dSubsampling(layers, in_dim, filters, kernel_size, stride, norm, activation)
    else:
        raise RuntimeError("Subsampling type {} not supported".format(subsampling_type))
