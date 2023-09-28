# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn as nn
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
    RelPositionMultiheadAttention,
    RelativeMultiheadAttention,
    ConvolutionModule,
    ESPNETMultiHeadedAttention,
    RelPositionMultiHeadedAttention,
    LegacyRelPositionMultiHeadedAttention,
    RotaryPositionMultiHeadedAttention,
)
from fairseq.modules.fairseq_dropout import FairseqDropout
from torch import Tensor
from fairseq.modules.activations import get_activation_class


class FeedForwardModule(torch.nn.Module):
    """Positionwise feed forward layer used in conformer"""

    def __init__(
        self,
        input_feat,
        hidden_units,
        dropout1,
        dropout2,
        activation_fn="relu",
        bias=True,
    ):
        """
        Args:
            input_feat: Input feature dimension
            hidden_units: Hidden unit dimension
            dropout1: dropout value for layer1
            dropout2: dropout value for layer2
            activation_fn: Name of activation function
            bias: If linear layers should have bias
        """

        super(FeedForwardModule, self).__init__()
        self.w_1 = torch.nn.Linear(input_feat, hidden_units, bias=bias)
        self.w_2 = torch.nn.Linear(hidden_units, input_feat, bias=bias)
        self.dropout1 = torch.nn.Dropout(dropout1)
        self.dropout2 = torch.nn.Dropout(dropout2)
        self.activation = get_activation_class(activation_fn)

    def forward(self, x):
        """
        Args:
            x: Input Tensor of shape  T X B X C
        Returns:
            Tensor of shape T X B X C
        """
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.w_2(x)
        return self.dropout2(x)


class S2TTransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        embed_dim = args.encoder_embed_dim
        ffn_dim = args.encoder_ffn_embed_dim
        dropout = args.dropout
        self.embed_dim = embed_dim
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8) or 8
        self.attn_type = getattr(args, "encoder_attention_type", "selfattn")
        self.self_attn = self.build_self_attention(args, self.embed_dim)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        activation = getattr(args, 'encoder_activation_fn', 'relu')

        if args.macaron_style:
            self.macaron_ffn = FeedForwardModule(
                embed_dim,
                ffn_dim,
                dropout,
                dropout,
                activation
            )
            self.macaron_norm = LayerNorm(embed_dim)
            self.ffn_scale = 0.5
        else:
            self.macaron_ffn = None
            self.macaron_norm = None
            self.ffn_scale = 1.0

        if args.use_cnn_module:
            self.conv_norm = LayerNorm(embed_dim)
            self.conv_module = ConvolutionModule(
                self.embed_dim,
                self.embed_dim,
                depthwise_kernel_size=args.cnn_module_kernel,
                dropout=args.dropout,
                activation_fn=getattr(args, 'activation_fn', 'swish'),
                norm_type=args.cnn_module_norm
            )
            self.final_norm = LayerNorm(embed_dim)
        else:
            self.conv_norm = None
            self.conv_module = None
            self.final_norm = None

        self.ffn = FeedForwardModule(
                embed_dim,
                ffn_dim,
                dropout,
                dropout,
                activation
            )
        self.ffn_norm = LayerNorm(self.embed_dim)

    def build_self_attention(self, args, embed_dim):
        attention_heads = args.encoder_attention_heads
        dropout = args.dropout

        if self.attn_type == "selfattn":
            attn_func = MultiheadAttention
        elif self.attn_type == "rel_selfattn":
            attn_func = RelPositionMultiheadAttention
        elif self.attn_type == "relative":
            max_relative_length = max(getattr(args, "max_encoder_relative_length", -1),
                                      getattr(args, "max_relative_length", -1))
            if max_relative_length != -1:
                return RelativeMultiheadAttention(
                    embed_dim,
                    attention_heads,
                    dropout=dropout,
                    self_attention=True,
                    q_noise=self.quant_noise,
                    qn_block_size=self.quant_noise_block_size,
                    max_relative_length=max_relative_length,
                )
            else:
                print("The maximum encoder relative length %d can not be -1!" % max_relative_length)
                exit(1)
        elif self.attn_type == "rel_pos":
            return RelPositionMultiHeadedAttention(
                embed_dim,
                attention_heads,
                dropout=dropout,
            )
        elif self.attn_type == "rel_pos_legacy":
            return LegacyRelPositionMultiHeadedAttention(
                embed_dim,
                attention_heads,
                dropout=dropout,
            )
        elif self.attn_type == "rope":
            return RotaryPositionMultiHeadedAttention(
                embed_dim,
                attention_heads,
                dropout=dropout,
                precision=args.fp16
            )
        elif self.attn_type == "abs":
            return ESPNETMultiHeadedAttention(
                embed_dim,
                attention_heads,
                dropout=dropout,
            )
        else:
            attn_func = MultiheadAttention
            print("The encoder attention type %s is not supported!" % self.attn_type)
            exit(1)

        return attn_func(
            embed_dim,
            attention_heads,
            dropout=dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def dump(self, fstream, info=""):
        self.self_attn.dump(fstream, "%s Self Attn" % (info)) if hasattr(self.self_attn, "dump") else None

    def set_flag(self, **kwargs):
        self.self_attn.set_flag(**kwargs)

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x,
                encoder_padding_mask: Optional[Tensor],
                attn_mask: Optional[Tensor] = None,
                pos_emb: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.
            pos_emb (Tensor): the position embedding for relative position encoding

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -float('inf'))

        # whether to use macaron style
        if self.macaron_norm is not None:
            residual = x
            if self.normalize_before:
                x = self.macaron_norm(x)
            x = self.macaron_ffn(x)
            x = residual + self.ffn_scale * x
            if not self.normalize_before:
                x = self.macaron_norm(x)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if self.attn_type in ["rel_pos", "rel_pos_legacy", "rel_selfattn"]:
            assert pos_emb is not None, "Positions is necessary for RPE!"
            x, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                need_weights=False,
                attn_mask=attn_mask,
                pos_emb=pos_emb
            )
        else:
            x, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                need_weights=False,
                attn_mask=attn_mask,
            )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # convolution module
        if self.conv_module is not None:
            residual = x

            x = x.transpose(0, 1)
            if self.normalize_before:
                x = self.conv_norm(x)

            # x = self.conv_module(x)
            x = self.conv_module(x, encoder_padding_mask)
            x = x.transpose(0, 1)
            x = residual + x

            if not self.normalize_before:
                x = self.conv_norm(x)

        residual = x
        if self.normalize_before:
            x = self.ffn_norm(x)
        x = self.ffn(x)
        x = self.residual_connection(self.ffn_scale * x, residual)
        if not self.normalize_before:
            x = self.ffn_norm(x)

        if self.conv_module is not None:
            x = self.final_norm(x)

        return x
