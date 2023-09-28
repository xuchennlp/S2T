from typing import Optional, Tuple

import torch
from torch import nn
from fairseq.modules.layer_norm import LayerNorm


class Swish(nn.Module):
    """Construct an Swish object."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Swish activation function."""
        return x * torch.sigmoid(x)


class DownSampleConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model."""
    def __init__(self,
                channels: int,
                kernel_size: int = 15,
                input_channels: int = None,
                activation: nn.Module = Swish(),
                norm: str = "batch_norm",
                stride: int = 1,
                causal: bool = False,
                bias: bool = True):
        """Construct an ConvolutionModule object.
        Args:
            channels (int): The number of channels of conv layers.
            kernel_size (int): Kernel size of conv layers.
            causal (int): Whether use causal convolution or not
        """
        super().__init__()

        if input_channels is None:
            input_channels = channels
        self.pointwise_conv1 = nn.Conv1d(
            input_channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        # padding = kernel_size // 2
        padding = 0

        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=channels,
            bias=bias,
        )

        assert norm in ['batch_norm', 'layer_norm']
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.use_layer_norm = True
            self.norm = LayerNorm(channels)

        self.stride = stride
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = Swish()

    def forward(
        self,
        x: torch.Tensor,
        mask_pad: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)

        zero_mask_pad = mask_pad.unsqueeze(1).repeat(1, x.size(1), 1)
        # mask batch padding
        if mask_pad is not None:
            x.masked_fill_(zero_mask_pad, 0.0)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, time)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)

        # mask batch padding
        bsz, dim, seq_len = x.size()
        lengths = (~mask_pad).sum(-1)
        lengths = (lengths / self.stride).long()
        max_length = x.size(-1)
        assert max_length >= max(lengths), (max_length, max(lengths))
        mask = torch.arange(max_length).to(lengths.device).view(1, max_length)
        mask_pad = mask.expand(bsz, -1) >= lengths.view(bsz, 1).expand(-1, max_length)
        zero_mask_pad = mask_pad.unsqueeze(1).repeat(1, x.size(1), 1)

        if zero_mask_pad is not None:
            x.masked_fill_(zero_mask_pad, 0.0)

        return x.permute(2, 0, 1)
