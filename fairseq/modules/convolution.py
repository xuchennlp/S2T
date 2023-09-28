import torch
from torch import nn

from fairseq.modules.activations import get_activation_class
from fairseq.modules.layer_norm import LayerNorm


class ConvolutionModule(nn.Module):
    """Convolution block used in the conformer block"""

    def __init__(
            self,
            embed_dim,
            expand_embed_dim,
            depthwise_kernel_size,
            dropout,
            activation_fn="swish",
            bias=False,
            stride=1,
            padding=None,
            export=False,
            norm_type="batch_norm"
    ):
        """
        Args:
            embed_dim: Embedding dimension
            expand_embed_dim: Number of output embedding dimension
            depthwise_kernel_size: Depthwise conv layer kernel size
            dropout: dropout value
            activation_fn: Activation function to use after depthwise convolution kernel
            bias: If bias should be added to conv layers
            export: If layernorm should be exported to jit
        """
        super(ConvolutionModule, self).__init__()

        self.stride = stride
        # assert (
        #         depthwise_kernel_size - 1
        #         ) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        self.pointwise_conv1 = torch.nn.Conv1d(
            embed_dim,
            2 * expand_embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.glu = torch.nn.GLU(dim=1)
        self.depthwise_conv = torch.nn.Conv1d(
            expand_embed_dim,
            expand_embed_dim,
            depthwise_kernel_size,
            stride=stride,
            padding=(depthwise_kernel_size - 1) // 2 if padding is None else padding,
            groups=expand_embed_dim,
            bias=bias,
        )
        self.norm_type = norm_type
        if norm_type == "batch_norm":
            self.norm = nn.BatchNorm1d(expand_embed_dim)
        elif norm_type == "layer_norm":
            self.norm = LayerNorm(expand_embed_dim)
        else:
            assert False, "Unsupported normalization type %s in convolution module" % norm_type
        self.activation = get_activation_class(activation_fn)
        self.pointwise_conv2 = torch.nn.Conv1d(
            expand_embed_dim,
            expand_embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask_pad=None):
        """
        Args:
            x: Input of shape B X T X C
        Returns:
            Tensor of shape B X T X C
        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)

        # mask batch padding
        if mask_pad is not None:
            zero_mask_pad = mask_pad.unsqueeze(1)
            x = x.masked_fill(zero_mask_pad, 0.0)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*expand_embed_dim, dim)
        x = self.glu(x)  # (batch, expand_embed_dim, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)

        if self.norm_type == "layer_norm":
            x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.activation(x)
        if self.norm_type == "layer_norm":
            x = x.transpose(1, 2)

        x = self.pointwise_conv2(x)

        # mask batch padding
        if mask_pad is not None:
            if self.stride != 1:
                mask_pad = mask_pad[:, ::self.stride]
                zero_mask_pad = mask_pad.unsqueeze(1)
                x = x.masked_fill(zero_mask_pad, 0.0)


            x = x.masked_fill(zero_mask_pad, 0.0)

        x = x.transpose(1, 2)
        x = self.dropout(x)

        return x

# class ConvolutionModule(nn.Module):
#     """ConvolutionModule in Conformer model."""
#     def __init__(self,
#                  channels: int,
#                  kernel_size: int = 15,
#                  norm: str = "batch_norm",
#                  bias: bool = True):
#         """Construct an ConvolutionModule object.
#         Args:
#             channels (int): The number of channels of conv layers.
#             kernel_size (int): Kernel size of conv layers.
#             causal (int): Whether use causal convolution or not
#         """
#         super().__init__()
#
#         self.pointwise_conv1 = nn.Conv1d(
#             channels,
#             2 * channels,
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             bias=bias,
#         )
#
#         # kernel_size should be an odd number for none causal convolution
#         assert (kernel_size - 1) % 2 == 0
#         padding = (kernel_size - 1) // 2
#
#         self.depthwise_conv = nn.Conv1d(
#             channels,
#             channels,
#             kernel_size,
#             stride=1,
#             padding=padding,
#             groups=channels,
#             bias=bias,
#         )
#
#         assert norm in ['batch_norm', 'layer_norm']
#         if norm == "batch_norm":
#             self.use_layer_norm = False
#             self.norm = nn.BatchNorm1d(channels)
#         else:
#             self.use_layer_norm = True
#             self.norm = LayerNorm(channels)
#
#         self.pointwise_conv2 = nn.Conv1d(
#             channels,
#             channels,
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             bias=bias,
#         )
#         self.activation = get_activation_class("swish")
#
#     def forward(
#         self,
#         x: torch.Tensor,
#         mask_pad: Optional[torch.Tensor] = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Compute convolution module.
#         Args:
#             x (torch.Tensor): Input tensor (#batch, time, channels).
#             mask_pad (torch.Tensor): used for batch padding
#         Returns:
#             torch.Tensor: Output tensor (#batch, time, channels).
#         """
#         # exchange the temporal dimension and the feature dimension
#         x = x.transpose(1, 2)
#
#         # zero_mask_pad = mask_pad.unsqueeze(1).repeat(1, x.size(1), 1)
#         # # mask batch padding
#         # if mask_pad is not None:
#         #     x.masked_fill_(zero_mask_pad, 0.0)
#
#         # GLU mechanism
#         x = self.pointwise_conv1(x)  # (batch, 2*channel, time)
#         x = nn.functional.glu(x, dim=1)  # (batch, channel, time)
#
#         # 1D Depthwise Conv
#         x = self.depthwise_conv(x)
#         if self.use_layer_norm:
#             x = x.transpose(1, 2)
#         x = self.activation(self.norm(x))
#         if self.use_layer_norm:
#             x = x.transpose(1, 2)
#         x = self.pointwise_conv2(x)
#
#         # # mask batch padding
#         # if zero_mask_pad is not None:
#         #     x.masked_fill_(zero_mask_pad, 0.0)
#
#         return x.transpose(1, 2)
