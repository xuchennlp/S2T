# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional, Tuple
import logging

import torch
import torch.nn as nn
from fairseq import checkpoint_utils, utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import TransformerEncoder, TransformerDecoder
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    RelPositionalEncoding,
    LegacyRelPositionalEncoding,
    DynamicLinearCombination,
    TransformerS2DecoderLayer,
    TransformerS2EncoderLayer
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

logger = logging.getLogger(__name__)


class TransformerS2Encoder(TransformerEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

        self.s2 = None
        self.s2_padding_mask = None

    def build_encoder_layer(self, args):
        layer = TransformerS2EncoderLayer(args)
        if getattr(args, "checkpoint_activations", False):
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        layer = fsdp_wrap(layer, min_num_params=1e8)
        return layer

    def set_s2(self, s2, s2_padding_mask):
        self.s2 = s2
        self.s2_padding_mask = s2_padding_mask
    
    def dump(self, fstream, info=""):
        for i, layer in enumerate(self.layers):
            layer.dump(fstream, "%s Layer %d" % (info, i)) if hasattr(layer, "dump") else None

    def forward(
            self,
            src_tokens,
            src_lengths: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(src_tokens,
                                       src_lengths,
                                       return_all_hiddens,
                                       token_embeddings)

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
            self,
            src_tokens,
            src_lengths: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = (src_tokens.device.type == "xla" or encoder_padding_mask.any())

        if self.history is not None:
            self.history.clean()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if encoder_padding_mask is not None:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # add emb into history
        if self.history is not None:
            self.history.push(x)

        # encoder layers
        for layer in self.layers:
            if self.history is not None:
                x = self.history.pop()

            x = layer(
                x, 
                encoder_padding_mask=encoder_padding_mask if has_pads else None,
                s2=self.s2,
                s2_encoder_padding_mask=self.s2_padding_mask,
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

            if self.history is not None:
                self.history.push(x)

        if self.history is not None:
            x = self.history.pop()

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }


class TransformerS2Decoder(TransformerDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        self.args = args
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = TransformerS2DecoderLayer(args, no_encoder_attn)
        if getattr(args, "checkpoint_activations", False):
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        layer = fsdp_wrap(layer, min_num_params=1e8)
        return layer

    def dump(self, fstream, info=""):
        for i, layer in enumerate(self.layers):
            layer.dump(fstream, "%s Layer %d" % (info, i)) if hasattr(layer, "dump") else None

    def extract_features_scriptable(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]],
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if self.history is not None:
            self.history.clean()

        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None and self.attn_type != "rel_selfattn":
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # add emb into history
        if self.history is not None:
            self.history.push(x)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        mixup = None
        if "mixup" in encoder_out and encoder_out["mixup"] is not None:
            mixup = encoder_out["mixup"]

            coef = mixup["coef"]
            idx1 = mixup["index1"]
            idx2 = mixup["index2"]

            x1 = x[:, idx1]
            x2 = x[:, idx2]
            x = coef * x1 + (1 - coef) * x2

            if self_attn_padding_mask is not None:
                pad1 = self_attn_padding_mask[idx1]
                pad2 = self_attn_padding_mask[idx2]
                self_attn_padding_mask = pad1 & pad2

        encoder_out_s2 = encoder_padding_mask_s2 = None
        if "s2_encoder_out" in encoder_out and len(encoder_out["s2_encoder_out"]) > 0:
            encoder_out_s2 = encoder_out["s2_encoder_out"][0]
            if len(encoder_out["s2_encoder_padding_mask"]):
                encoder_padding_mask_s2 = encoder_out["s2_encoder_padding_mask"][0]

        # decoder layers
        avg_attn = None
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if self.history is not None:
                x = self.history.pop()

            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                        encoder_out is not None
                        and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                encoder_out_s2=encoder_out_s2,
                encoder_padding_mask_s2=encoder_padding_mask_s2,
                incremental_state=incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer) or self.gather_attn_weight),
                need_head_weights=bool((idx == alignment_layer) or self.gather_attn_weight),
                pos_emb=positions
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)
            if self.history is not None:
                self.history.push(x)
            if self.gather_attn_weight:
                if avg_attn is None:
                    avg_attn = layer_attn
                else:
                    avg_attn += layer_attn

        if self.gather_attn_weight:
            avg_attn = avg_attn / len(self.layers)
            attn = avg_attn.mean(0).sum(-2)
            attn = torch.reshape(attn, [attn.numel()])
            attn = attn // 0.001
            attn = attn.int().cpu()

            if len(encoder_out["encoder_padding_mask"]) > 0:
                mask = encoder_out["encoder_padding_mask"][0]
                mask = torch.reshape(mask, [mask.numel()])
            else:
                mask = None

            i = -1
            for item in attn:
                i += 1
                if mask[i]:
                    continue
                idx = int(item) * 0.001
                if idx not in self.attn_weights:
                    self.attn_weights[idx] = 0
                self.attn_weights[idx] += 1
        elif attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.history is not None:
            x = self.history.pop()

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states, "mixup": mixup}


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m
