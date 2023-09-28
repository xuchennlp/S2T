import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules.speech_to_text import Adapter, CTC
from fairseq.models.transformer import Embedding, TransformerDecoder
from fairseq.models.speech_to_text import S2TTransformerModel, S2TTransformerEncoder
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    LegacyRelPositionalEncoding,
    RelPositionalEncoding,
    S2TTransformerEncoderLayer,
    DynamicLinearCombination,
)
from fairseq.modules.speech_to_text import (
    subsampling
)

from torch import Tensor

logger = logging.getLogger(__name__)


@register_model("s2t_dynamic_transformer")
class S2TDynamicTransformerModel(S2TTransformerModel):
    """Adapted Transformer model (https://arxiv.org/abs/1706.03762) for
    speech-to-text tasks. The Transformer encoder/decoder remains the same.
    A trainable input subsampler is prepended to the Transformer encoder to
    project inputs into the encoder dimension as well as downsample input
    sequence for computational efficiency."""

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        S2TTransformerModel.add_args(parser)

        # dynamic compression
        parser.add_argument(
            "--compression-metric",
            default="ratio",
            type=str,
            help="the metric of compression",
        )
        parser.add_argument(
            "--compression-mode",
            default="create",
            type=str,
            help="the mode of compression",
        )
        parser.add_argument(
            "--compression-layers",
            default=None,
            type=str,
            help="the layers to compress",
        )
        parser.add_argument(
            "--compression-threshold",
            default="1",
            type=str,
            help="compress the units below the threshold",
        )
        parser.add_argument(
            "--compression-ratio",
            default="0",
            type=str,
            help="compress the units of fix ratio",
        )
        parser.add_argument(
            "--compression-norm",
            action="store_true",
            help="normalize the representation after compression",
        )
        parser.add_argument(
            "--compression-pos",
            action="store_true",
            help="add the position embedding after compression",
        )
        pass

    @classmethod
    def build_encoder(cls, args, task=None, embed_tokens=None):
        encoder = S2TDynamicTransformerEncoder(args, task, embed_tokens)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from, strict=False
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )

        return encoder


class S2TDynamicTransformerEncoder(S2TTransformerEncoder):
    """Speech-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args, task=None, embed_tokens=None):
        super().__init__(args, task, embed_tokens)

        self.compression_metric = args.compression_metric
        self.compression_mode = args.compression_mode

        compression_num = len(args.compression_layers.split(",")) if args.compression_layers is not None else 0
        if compression_num > 0:
            self.compression_layers = [int(i) for i in args.compression_layers.split(",")]

            thresholds = [float(n) for n in args.compression_threshold.split(",")]
            assert len(thresholds) == 1 or len(thresholds) == compression_num
            compression_threshold = thresholds if len(
                thresholds) == compression_num else thresholds * compression_num
            for i, layer in enumerate(self.compression_layers):
                threshold = compression_threshold[i]
                setattr(self, "compression_threshold%d" % layer, threshold)

            ratios = [float(n) for n in args.compression_ratio.split(",")]
            assert len(ratios) == 1 or len(ratios) == compression_num
            self.compression_ratio = ratios if len(ratios) == compression_num else ratios * compression_num

            self.compression_pos = args.compression_pos
            if self.compression_pos:
                self.compression_embed_positions = PositionalEmbedding(
                    args.max_source_positions, args.encoder_embed_dim, self.padding_idx,
                    False
                    # True
                )

            self.compression_norm = args.compression_norm
            if self.compression_norm:
                for layer in self.compression_layers:
                    norm = LayerNorm(args.encoder_embed_dim)
                    setattr(self, "compression_norm%d" % layer, norm)
        else:
            self.compression_layers = []

        self.compression_stat = False

    def set_flag(self, **kwargs):
        super().set_flag(**kwargs)

        self.compression_stat = kwargs.get("compression_stat", False)
        if self.compression_stat:
            self.ctc_stat = dict()

            for layer_idx in self.compression_layers:
                self.ctc_stat[layer_idx]["distribution"] = dict()
                self.ctc_stat[layer_idx]["distribution_ratio"] = dict()
                self.ctc_stat[layer_idx]["length_ratio"] = []

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        layer_idx = -1
        mixup = None

        if self.history is not None:
            self.history.clean()

        # (B, T, D) -> (T, B, D)
        x = src_tokens.transpose(0, 1)
        input_lengths = src_lengths

        self.show_debug(x, "input x")
        # gather cosine similarity
        cos_sim_idx = -1
        dis = self.dis
        if self.gather_cos_sim:
            self.add_to_dict(x, dis, cos_sim_idx)

        if self.training and self.mixup and layer_idx == self.mixup_layer:
            if torch.rand(1) < self.mixup_prob:
                encoder_padding_mask = lengths_to_padding_mask(input_lengths)
                x, encoder_padding_mask, input_lengths, mixup = self.apply_mixup(x, encoder_padding_mask)

        # down-sampling
        x, input_lengths = self.subsample(x, input_lengths)
        self.show_debug(x, "x after subsampling")

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        if encoder_padding_mask is not None:
            x = x * (1 - encoder_padding_mask.transpose(0, 1).unsqueeze(-1).type_as(x))

        if self.encoder_embed_norm:
            x = self.embed_ln(x)
            self.show_debug(x, "x after embed norm")

        # embedding scaling
        x = self.embed_scale * x
        self.show_debug(x, "x after scale")

        # position embedding
        if self.attn_type in ["rel_pos", "rel_pos_legacy", "rel_selfattn"]:
            positions = self.embed_positions(x)

        elif self.attn_type == "rope":
            positions = None

        else:
            positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
            x += positions
            positions = None
        self.show_debug(x, "x after position embedding")

        if self.encoder_embed_linear:
            x = self.linear(x)
            self.show_debug(x, "x after embed linear")

        x = self.dropout_module(x)

        # add emb into history
        if self.history is not None:
            self.history.push(x)

        # gather cosine similarity
        cos_sim_idx = (cos_sim_idx + 10) // 10 * 10 - 1
        if self.gather_cos_sim:
            cos_sim_idx += 1
            self.add_to_dict(x, dis, cos_sim_idx)

        layer_idx += 1
        ctc_logit = None
        interleaved_ctc_logits = []

        if self.training and self.mixup and layer_idx == self.mixup_layer:
            if torch.rand(1) <= self.mixup_prob:
                x, encoder_padding_mask, input_lengths, mixup = self.apply_mixup(x, encoder_padding_mask)

        self.show_debug(x, "x before encoding")
        for layer in self.layers:
            if self.history is not None:
                x = self.history.pop()

            # encoder layer
            x = layer(x, encoder_padding_mask, pos_emb=positions)
            layer_idx += 1
            self.show_debug(x, "x after layer %d" % layer_idx)

            if self.training and self.mixup and layer_idx == self.mixup_layer:
                if torch.rand(1) < self.mixup_prob:
                    x, encoder_padding_mask, input_lengths, mixup = self.apply_mixup(x, encoder_padding_mask)

            if self.use_ctc and self.inter_ctc and self.ctc_layer == layer_idx:
                ctc_logit = self.ctc(x.clone(), encoder_padding_mask, "Source Layer %d" % layer_idx)

            # interleaved CTC
            if layer_idx in self.interleaved_ctc_layers:
                if self.interleaved_ctc_drop_prob > 0:
                    p = torch.rand(1).uniform_()
                    if p < self.interleaved_ctc_drop_prob:
                        break

                if self.share_interleaved_ctc:
                    inter_ctc = self.ctc
                    sae = self.sae
                    layer_norm = self.layer_norm
                else:
                    inter_ctc = getattr(self, "inter_ctc%d" % layer_idx)
                    sae = getattr(self, "sae%d" % layer_idx)
                    layer_norm = getattr(self, "inter_layer_norm%d" % layer_idx)

                x = layer_norm(x)
                logit = inter_ctc(x, encoder_padding_mask, "Source Layer %d" % layer_idx)
                interleaved_ctc_logits.append([logit, encoder_padding_mask.clone()])

                if sae.adapter_type != "none":
                    x, encoder_padding_mask = sae([x, logit], encoder_padding_mask)
                    self.show_debug(x, "x after sae")

                if layer_idx in self.compression_layers:
                    ctc_prob = utils.softmax(logit, dim=-1)  # (T B C)
                    blank_prob = ctc_prob[:, :, 0]
                    bsz = x.size(1)

                    if self.compression_metric == "threshold":
                        threshold = getattr(self, "compression_threshold%d" % layer_idx)
                        keep_flag = blank_prob < threshold
                        # keep_flag = keep_flag.masked_fill(encoder_padding_mask.clone().transpose(0, 1), False)
                        keep_flag = keep_flag.masked_fill(encoder_padding_mask.clone().transpose(0, 1), False)
                        max_len = max(keep_flag.sum(0))
                        min_len = min(keep_flag.sum(0))

                        if self.compression_mode == "create":
                            # logger.info(keep_flag.sum(0))
                            # logger.info("layer: %d max len: %d min len: %d" % (layer_idx, max_len, min_len))

                            if min_len > 0 and max_len > 0 and not keep_flag.all():
                                out = x.new_zeros(max_len, bsz, x.size(2))
                                out_encoder_padding_mask = encoder_padding_mask.new_ones(bsz, max_len).bool()

                                for i in range(bsz):
                                    item_flag = keep_flag[:, i]
                                    org_tensor = x[:, i, :]
                                    new_tensor = org_tensor[item_flag]
                                    out[:new_tensor.size(0), i, :] = new_tensor
                                    out_encoder_padding_mask[i, :new_tensor.size(0)] = False

                        elif self.compression_mode == "mask":
                            encoder_padding_mask = encoder_padding_mask | (~keep_flag).transpose(0, 1)

                    elif self.compression_metric == "ratio":
                        pass
                        ratio = self.compression_ratio[0]
                        number = int(x.size(0) * ratio)
                        # logger.info("layer: %d max len: %d" % (layer_idx, number))

                        if number > 0:
                            values, indices = torch.topk(blank_prob, k=number, dim=0, largest=False)
                            # new_encoder_padding_mask = encoder_padding_mask.new_zeros(bsz, number).bool()

                            # report error
                            if self.compression_mode == "create":
                                # indices = indices.contiguous().transpose(0, 1).to(x.device)
                                # torch.gather(encoder_padding_mask, dim=1, index=indices, out=new_encoder_padding_mask)
                                # encoder_padding_mask = new_encoder_padding_mask
                                # encoder_padding_mask = encoder_padding_mask.gather(dim=1, index=indices.detach().clone()).contiguous()
                                # encoder_padding_mask = encoder_padding_mask[indices].view(-1, number)
                                # with torch.no_grad():
                                #     encoder_padding_mask = torch.gather(encoder_padding_mask, dim=1, index=indices).contiguous()

                                # out = x.contiguous().transpose(0, 1).gather(dim=1, index=indices.unsqueeze(-1).repeat(1, 1, x.size(-1)))
                                # encoder_padding_mask = encoder_padding_mask.gather(dim=1, index=indices.transpose(0, 1)).contiguous()
                                # embed_positions = self.compression_embed_positions(encoder_padding_mask).transpose(0, 1)
                                # x = out.transpose(0, 1) + embed_positions
                                # out = x[: number, :, :]
                                # x = out.contiguous()
                                # x = out.contiguous().transpose(0, 1)
                                # x = x.contiguous()

                                out = x.gather(dim=0, index=indices.unsqueeze(-1).repeat(1, 1, x.size(-1)))
                                encoder_padding_mask = encoder_padding_mask.gather(dim=1, index=indices.transpose(0, 1)).contiguous()
                                embed_positions = self.compression_embed_positions(encoder_padding_mask). \
                                    transpose(0, 1)
                                x = out.contiguous() + embed_positions

                            elif self.compression_mode == "mask":
                                pass
                    
                    else:
                        out = x
                        out_encoder_padding_mask = encoder_padding_mask

                    if self.compression_stat:
                        self.ctc_stat[layer_idx]["distribution"] = dict()
                        self.ctc_stat[layer_idx]["distribution_ratio"] = dict()
                        self.ctc_stat[layer_idx]["length_ratio"] = []

                        # prob_stat = blank_prob[encoder_padding_mask]
                        # prob_num = len(prob_stat)
                        # prob_list = []
                        # for p in prob_stat:
                        #     key = int(p / 0.01) / 100
                        #     prob_list.append(key)
                        #     if key not in self.ctc_stat[layer_idx]["distribution"]:
                        #         self.ctc_stat[layer_idx]["distribution"][key] = 0
                        #     self.ctc_stat[layer_idx]["distribution"][key] += 1

                        # for p in range(0, 1, 0.01):
                        #     if p in prob_list:
                        #         count = prob_list.count(p)
                        #         if p not in self.ctc_stat[layer_idx]["distribution_ratio"]:
                        #             self.ctc_stat[layer_idx]["distribution_ratio"][p] = []
                        #         self.ctc_stat[layer_idx]["distribution_ratio"][p].append(count/prob_list)

                        # thresholds = [0.7, 0.8, 0.9, 0.95, 0.99]

                        # for i in range(blank_prob.size(0)):
                        #     for j in range(blank_prob.size(1)):
                        #         if not encoder_padding_mask[i, j]:
                        #             prob = 
                        
                    encoder_padding_mask = out_encoder_padding_mask.contiguous()
                    x = out.contiguous()

                    if self.compression_norm:
                        norm = getattr(self, "compression_norm%d" % layer_idx)
                        x = norm(x)
                    if self.compression_pos:
                        embed_positions = self.compression_embed_positions(encoder_padding_mask). \
                            transpose(0, 1)
                        x = x + embed_positions

                    x = x.transpose(0, 1)
                    x.masked_fill_(encoder_padding_mask.unsqueeze(2), 0.0)
                    x = x.transpose(0, 1)

            # gather cosine similarity
            if self.gather_cos_sim:
                cos_sim_idx += 1
                self.add_to_dict(x, dis, cos_sim_idx)

            if self.history is not None:
                self.history.push(x)

        if self.history is not None:
            x = self.history.pop()

        self.show_debug(x, "x after encoding")
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        self.show_debug(x, "x after encoding layer norm")

        if self.use_ctc and ctc_logit is None:
            ctc_logit = self.ctc(x, encoder_padding_mask, "Source output", is_top=True)
            self.show_debug(x, "x after ctc")

        return {
            "encoder_out": [x],  # T x B x C
            "ctc_logit": [] if ctc_logit is None else [ctc_logit],  # T x B x C
            "interleaved_ctc_logits": interleaved_ctc_logits,  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            # "oracle": [oracle, oracle_mask, force_emit],
            "mixup": mixup,
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            [] if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_ctc_logit = (
            [] if len(encoder_out["ctc_logit"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["ctc_logit"] if x is not None]
        )

        new_encoder_padding_mask = (
            [] if len(encoder_out["encoder_padding_mask"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_padding_mask"]]
        )

        new_encoder_embedding = (
            [] if len(encoder_out["encoder_embedding"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "ctc_logit": new_ctc_logit,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }


class TransformerDecoderScriptable(TransformerDecoder):
    def extract_features(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        # call scriptable method from parent class
        x, extra = self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )
        return x, extra

    def get_normalized_probs_scriptable(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None:
            if sample is not None:
                assert "target" in sample
                target = sample["target"]
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output[0]
        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)


@register_model_architecture(model_name="s2t_dynamic_transformer", arch_name="s2t_dynamic_transformer")
def base_architecture(args):
    # Convolutional subsampler
    args.subsampling_type = getattr(args, "subsampling_type", "conv1d")
    args.subsampling_layers = getattr(args, "subsampling_layers", 2)
    args.subsampling_filter = getattr(args, "subsampling_filter", 1024)
    args.subsampling_kernel = getattr(args, "subsampling_kernel", 5)
    args.subsampling_stride = getattr(args, "subsampling_stride", 2)
    args.subsampling_norm = getattr(args, "subsampling_norm", "none")
    args.subsampling_activation = getattr(args, "subsampling_activation", "glu")

    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_type = getattr(args, "encoder_attention_type", "selfattn")
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_type = getattr(args, "decoder_attention_type", "selfattn")
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)

    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.encoder_no_scale_embedding = getattr(args, "encoder_no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    args.encoder_embed_linear = getattr(args, "encoder_embed_linear", False)
    args.encoder_embed_norm = getattr(args, "encoder_embed_norm", False)

    # CTC
    args.ctc_layer = getattr(args, "ctc_layer", 0)
    args.share_ctc_and_embed = getattr(args, "share_ctc_and_embed", False)

    # Conformer
    args.encoder_activation_fn = getattr(args, "encoder_activation_fn", "relu")
    args.macaron_style = getattr(args, "macaron_style", False)
    args.use_cnn_module = getattr(args, "use_cnn_module", False)
    args.cnn_module_kernel = getattr(args, "cnn_module_kernel", 31)
    args.cnn_module_norm = getattr(args, "cnn_module_norm", "batch_norm")

    # settings for DLCL
    args.use_enc_dlcl = getattr(args, "use_enc_dlcl", False)
    args.use_dec_dlcl = getattr(args, "use_dec_dlcl", False)
    args.init_value = getattr(args, 'init_value', 'avg')
    args.weight_type = getattr(args, 'weight_type', 'scalar')
    args.encoder_learnable = getattr(args, 'encoder_learnable', True)
    args.decoder_learnable = getattr(args, 'decoder_learnable', True)
    args.normalize_embed = getattr(args, 'normalize_embed', False)
    args.history_dropout = getattr(args, 'history_dropout', 0.0)
    args.history_window_size = getattr(args, 'history_window_size', -1)

    # Relative position encoding
    args.max_encoder_relative_length = getattr(args, 'max_encoder_relative_length', -1)
    args.max_decoder_relative_length = getattr(args, 'max_decoder_relative_length', -1)
    args.k_only = getattr(args, 'k_only', True)

    # local modeling
    args.hard_mask_window = getattr(args, 'hard_mask_window', 0)
    args.gauss_mask_sigma = getattr(args, 'gauss_mask_sigma', 0)
    args.init_mask_weight = getattr(args, 'init_mask_weight', 0)

    # interleaved CTC
    args.interleaved_ctc_layers = getattr(args, "interleaved_ctc_layers", None)
    args.interleaved_ctc_temperature = getattr(args, "interleaved_ctc_temperature", 1)
    args.interleaved_ctc_drop_prob = getattr(args, "interleaved_ctc_drop_prob", 0)

    # Semantics-augmented Encoding (sae)
    args.sae_adapter = getattr(args, "sae_adapter", "none")
    args.share_sae_and_ctc = getattr(args, "share_sae_and_ctc", False)
    args.sae_embed_norm = getattr(args, "sae_embed_norm", False)
    args.sae_out_norm = getattr(args, "sae_out_norm", False)
    args.sae_drop_prob = getattr(args, "sae_drop_prob", 0)
    args.sae_distribution_cutoff = getattr(args, "sae_distribution_cutoff", None)
    args.sae_distribution_hard = getattr(args, "sae_distribution_hard", False)
    args.sae_gumbel = getattr(args, "sae_gumbel", False)

    # mixup
    args.inter_mixup = getattr(args, "inter_mixup", False)
    args.inter_mixup_layer = getattr(args, "inter_mixup_layer", None)
    args.inter_mixup_beta = getattr(args, "inter_mixup_beta", 0.5)
    args.inter_mixup_prob = getattr(args, "inter_mixup_prob", 1)
    args.inter_mixup_ratio = getattr(args, "inter_mixup_ratio", 0.3)
    args.inter_mixup_keep_org = getattr(args, "inter_mixup_keep_org", False)

    args.compression_metric = getattr(args, "compression_metric", "ratio")
    args.compression_mode = getattr(args, "compression_mode", "create")
    args.compression_layers = getattr(args, "compression_layers", None)
    args.compression_threshold = getattr(args, "compression_threshold", "1.0")
    args.compression_ratio = getattr(args, "compression_ratio", "0.0")
    args.compression_norm = getattr(args, "compression_norm", False)
    args.compression_pos = getattr(args, "compression_pos", False)


@register_model_architecture("s2t_dynamic_transformer", "s2t_dynamic_transformer_s")
def s2t_dynamic_transformer_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)


@register_model_architecture("s2t_dynamic_transformer", "s2t_dynamic_transformer_s_relative")
def s2t_dynamic_transformer_s_relative(args):
    args.max_encoder_relative_length = 100
    args.max_decoder_relative_length = 20
    args.k_only = True
    s2t_dynamic_transformer_s(args)


@register_model_architecture("s2t_dynamic_transformer", "s2t_dynamic_transformer_xs")
def s2t_dynamic_transformer_xs(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 4)
    args.dropout = getattr(args, "dropout", 0.3)
    s2t_dynamic_transformer_s(args)


@register_model_architecture("s2t_dynamic_transformer", "s2t_dynamic_transformer_sp")
def s2t_dynamic_transformer_sp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_dynamic_transformer_s(args)


@register_model_architecture("s2t_dynamic_transformer", "s2t_dynamic_transformer_m")
def s2t_dynamic_transformer_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    base_architecture(args)


@register_model_architecture("s2t_dynamic_transformer", "s2t_dynamic_transformer_mp")
def s2t_dynamic_transformer_mp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_dynamic_transformer_m(args)


@register_model_architecture("s2t_dynamic_transformer", "s2t_dynamic_transformer_l")
def s2t_dynamic_transformer_l(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.2)
    base_architecture(args)


@register_model_architecture("s2t_dynamic_transformer", "s2t_dynamic_transformer_lp")
def s2t_dynamic_transformer_lp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_dynamic_transformer_l(args)
