# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional
from numpy.random import uniform

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
    RelPositionMultiheadAttention,
    RelativeMultiheadAttention,
    LocalMultiheadAttention,
    SEAttention,
)
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor


class TransformerS2EncoderLayer(nn.Module):
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
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8) or 8
        self.attn_type = getattr(args, "encoder_attention_type", "selfattn")
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu') or "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.encoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)

        self.use_se = getattr(args, "squeeze_excitation", False)
        if self.use_se:
            self.se_attn = SEAttention(self.embed_dim, 16)

        self.use_s2_attn_norm = getattr(args, "encoder_use_s2_attn_norm", True)
        if self.use_s2_attn_norm:
            self.s2_norm = LayerNorm(self.embed_dim)

        self.encoder_collaboration_mode = args.encoder_collaboration_mode
        if self.encoder_collaboration_mode != "none":
            if self.encoder_collaboration_mode == "serial":
                self.s2_attn_norm = LayerNorm(self.embed_dim)
            self.s2_attn = MultiheadAttention(
                self.embed_dim,
                args.encoder_attention_heads,
                kdim=getattr(args, "s2_encoder_embed_dim", self.embed_dim),
                vdim=getattr(args, "s2_encoder_embed_dim", self.embed_dim),
                dropout=args.attention_dropout,
                self_attention=False,
            )
        
        self.league_s1_ratio = args.encoder_league_s1_ratio
        self.league_s2_ratio = args.encoder_league_s2_ratio
        self.league_out_norm = getattr(args, "encoder_league_out_norm", False)
        if self.league_out_norm:
            self.league_s1_norm = LayerNorm(self.embed_dim)
            self.league_s2_norm = LayerNorm(self.embed_dim)

        self.league_gated = getattr(args, "encoder_league_gated", False)
        if self.league_gated:
            self.gate_linear = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.league_drop_net = args.encoder_league_drop_net
        self.league_drop_net_prob = args.encoder_league_drop_net_prob
        self.league_drop_net_mix = args.encoder_league_drop_net_mix

    def dump(self, fstream, info=""):
        self.self_attn.dump(fstream, "%s Self Attn" % (info)) if hasattr(self.self_attn, "dump") else None
        if hasattr(self, "s2_attn"):
            self.s2_attn.dump(fstream, "%s S2 Attn" % (info)) if hasattr(self.s2_attn, "dump") else None

    def set_flag(self, **kwargs):
        if hasattr(self.self_attn, "set_flag"):
            self.self_attn.set_flag(**kwargs)
        if hasattr(self, "s2_attn") and hasattr(self.s2_attn, "set_flag"):
            self.s2_attn.set_flag(**kwargs)

    def get_ratio(self):
        if self.league_drop_net:
            lam = float(uniform(0, 1))
            if self.league_drop_net_mix and self.training:
                return [lam, 1 - lam]
            if lam < self.league_drop_net_prob and self.training:
                return [1, 0]
            elif lam > 1 - self.league_drop_net_prob and self.training:
                return [0, 1]
            else:
                return [0.5, 0.5]
        else:
            return [self.league_s1_ratio, self.league_s2_ratio]

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, args):
        if self.attn_type == "selfattn":
            attn_func = MultiheadAttention
        elif self.attn_type == "rel_selfattn":
            attn_func = RelPositionMultiheadAttention
        elif self.attn_type == "relative":
            # max_relative_length = getattr(args, "max_encoder_relative_length", -1)
            max_relative_length = max(getattr(args, "max_encoder_relative_length", -1), getattr(args, "max_relative_length", -1))
            if max_relative_length != -1:
                return RelativeMultiheadAttention(
                    embed_dim,
                    args.encoder_attention_heads,
                    dropout=args.attention_dropout,
                    self_attention=True,
                    q_noise=self.quant_noise,
                    qn_block_size=self.quant_noise_block_size,
                    max_relative_length=max_relative_length,
                )
            else:
                print("The maximum encoder relative length %d can not be -1!" % max_relative_length)
                exit(1)
        elif self.attn_type == "local":
            hard_mask_window = getattr(args, "hard_mask_window", 0)
            gauss_mask_sigma = getattr(args, "gauss_mask_sigma", 0)
            init_mask_weight = getattr(args, "init_mask_weight", 0)
            return LocalMultiheadAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
                hard_mask_window=hard_mask_window,
                gauss_mask_sigma=gauss_mask_sigma,
                init_mask_weight=init_mask_weight
            )
        else:
            print("The encoder attention type %s is not supported!" % self.attn_type)
            exit(1)

        return attn_func(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

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
                s2 = None,
                s2_encoder_padding_mask = None,
                s2_need_norm = False,
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
            attn_mask = attn_mask.masked_fill(
                attn_mask.to(torch.bool), -float('inf') # -1e8 if x.dtype == torch.float32 else -1e4
            )

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        attn_x = x
        if self.attn_type == "rel_selfattn":
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
        if s2 is None or self.encoder_collaboration_mode == "serial":
            if self.encoder_collaboration_mode == "serial" and \
                self.training and self.league_drop_net and float(uniform(0, 1)) < self.league_drop_net_prob:
                x = residual
            else:
                x = self.residual_connection(x, residual)
                if not self.normalize_before:
                    x = self.self_attn_layer_norm(x)

        if s2 is not None:
            if s2_need_norm and self.use_s2_attn_norm:
                s2 = self.s2_norm(s2)

            if self.encoder_collaboration_mode == "serial":
                residual = x
                x = self.s2_attn_norm(x)
                x, _ = self.s2_attn(
                    query=x,
                    key=s2,
                    value=s2,
                    key_padding_mask=s2_encoder_padding_mask)
                x = self.dropout_module(x)
                if self.league_gated:
                    lam = self.gate_linear(torch.cat([x, residual], dim=-1)).sigmoid()
                    x = x * lam + residual * (1 - lam)
                else:
                    x = self.residual_connection(x, residual)

            elif self.encoder_collaboration_mode == "parallel":
                x2, _ = self.s2_attn(
                    query=attn_x,
                    key=s2,
                    value=s2,
                    key_padding_mask=s2_encoder_padding_mask)
                x2 = self.dropout_module(x2)

                # league
                if self.league_out_norm:
                    x = self.league_s1_norm(x)
                    x2 = self.league_s2_norm(x2)

                if self.league_gated:
                    lam = self.gate_linear(torch.cat([x, x2], dim=-1)).sigmoid()
                    x = x * lam + x2 * (1 - lam)
                else:
                    ratio = self.get_ratio()
                    x = x * ratio[0] + x2 * ratio[1]
                x = self.residual_connection(x, residual)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)

        # use squeeze-and-excitation method
        if self.use_se:
            x = self.se_attn(x)

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class TransformerS2DecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.attn_type = getattr(args, "decoder_attention_type", "selfattn")
        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

        self.use_s2_attn_norm = getattr(args, "decoder_use_s2_attn_norm", True)
        if self.use_s2_attn_norm:
            self.s2_encoder_out_norm = LayerNorm(getattr(args, "encoder_embed_dim", self.embed_dim))

        self.decoder_collaboration_mode = args.decoder_collaboration_mode
        if self.decoder_collaboration_mode != "none":
            if self.decoder_collaboration_mode == "serial":
                self.s2_encoder_attn_layer_norm = LayerNorm(self.embed_dim)

            self.s2_encoder_attn = MultiheadAttention(
                self.embed_dim,
                args.decoder_attention_heads,
                kdim=getattr(args, "s2_encoder_embed_dim", self.embed_dim),
                vdim=getattr(args, "s2_encoder_embed_dim", self.embed_dim),
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
            )

        self.league_s1_ratio = args.decoder_league_s1_ratio
        self.league_s2_ratio = args.decoder_league_s2_ratio
        self.league_out_norm = getattr(args, "encoder_league_out_norm", False)
        if self.league_out_norm:
            self.league_s1_norm = LayerNorm(self.embed_dim)
            self.league_s2_norm = LayerNorm(self.embed_dim)

        self.league_gated = getattr(args, "encoder_league_gated", False)
        if self.league_gated:
            self.gate_linear = nn.Linear(2 * self.embed_dim, self.embed_dim)

        self.league_drop_net = args.decoder_league_drop_net
        self.league_drop_net_prob = args.decoder_league_drop_net_prob
        self.league_drop_net_mix = args.decoder_league_drop_net_mix

    def get_ratio(self):
        if self.league_drop_net:
            lam = float(uniform(0, 1))
            if self.league_drop_net_mix and self.training:
                return [lam, 1 - lam]
            if lam < self.league_drop_net_prob and self.training:
                return [1, 0]
            elif lam > 1 - self.league_drop_net_prob and self.training:
                return [0, 1]
            else:
                return [0.5, 0.5]
        else:
            return [self.league_s1_ratio, self.league_s2_ratio]

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        if self.attn_type == "selfattn":
            attn_func = MultiheadAttention
        elif self.attn_type == "rel_selfattn":
            attn_func = RelPositionMultiheadAttention
        elif self.attn_type == "relative":
            max_relative_length = max(getattr(args, "max_decoder_relative_length", -1), getattr(args, "max_relative_length", -1))
            if max_relative_length != -1:
                return RelativeMultiheadAttention(
                    embed_dim,
                    args.decoder_attention_heads,
                    dropout=args.attention_dropout,
                    self_attention=True,
                    q_noise=self.quant_noise,
                    qn_block_size=self.quant_noise_block_size,
                    max_relative_length=max_relative_length,
                )
            else:
                print("The maximum decoder relative length %d can not be -1!" % max_relative_length)
                exit(1)
        else:
            print("The decoder attention type %s is not supported!" % self.attn_type)
            exit(1)

        return attn_func(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        encoder_out_s2 = None,
        encoder_padding_mask_s2 = None,
        encoder_out_s2_need_norm = False,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        pos_emb: Optional[Tensor] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)

        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )

                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        if self.attn_type == "rel_selfattn":
            assert pos_emb is not None, "Positions is necessary for RPE!"
            x, attn = self.self_attn(
                query=x,
                key=y,
                value=y,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                need_weights=False,
                attn_mask=self_attn_mask,
                pos_emb=pos_emb
            )
        else:
            x, attn = self.self_attn(
                query=x,
                key=y,
                value=y,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                need_weights=False,
                attn_mask=self_attn_mask,
            )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            cross_attn_x = x

            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
                # notice here
                # self.s2_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            if encoder_out_s2 is None or self.decoder_collaboration_mode == "serial":
                x = self.residual_connection(x, residual)
                if not self.normalize_before:
                    x = self.encoder_attn_layer_norm(x)

            if encoder_out_s2 is not None:
                if encoder_out_s2_need_norm and self.use_s2_attn_norm:
                    encoder_out_s2 = self.s2_encoder_out_norm(encoder_out_s2)

                if self.decoder_collaboration_mode == "serial":
                    residual = x
                    x = self.s2_encoder_attn_layer_norm(x)
                    x, _ = self.s2_encoder_attn(
                        query=x,
                        key=encoder_out_s2,
                        value=encoder_out_s2,
                        key_padding_mask=encoder_padding_mask_s2,
                        incremental_state=incremental_state,
                        static_kv=True,
                        need_weights=need_attn or (not self.training and self.need_attn),
                        need_head_weights=need_head_weights,
                    )
                    x = self.dropout_module(x)
                    x = self.residual_connection(x, residual)
                elif self.decoder_collaboration_mode == "parallel":
                    x2, _ = self.s2_encoder_attn(
                        query=cross_attn_x,
                        key=encoder_out_s2,
                        value=encoder_out_s2,
                        key_padding_mask=encoder_padding_mask_s2,
                        incremental_state=incremental_state,
                        static_kv=True,
                        need_weights=need_attn or (not self.training and self.need_attn),
                        need_head_weights=need_head_weights,
                    )
                    x2 = self.dropout_module(x2)

                    # league
                    if self.league_out_norm:
                        x = self.league_s1_norm(x)
                        x2 = self.league_s2_norm(x2)

                    if self.league_gated:
                        lam = self.gate_linear(torch.cat([x, x2], dim=-1)).sigmoid()
                        x = x * lam + x2 * (1 - lam)
                    else:
                        ratio = self.get_ratio()
                        x = x * ratio[0] + x2 * ratio[1] 

                    x = self.residual_connection(x, residual)
                    if not self.normalize_before:
                        x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn
