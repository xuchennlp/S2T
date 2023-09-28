import logging
import math
from typing import Dict, List, Optional, Tuple
import os

import numpy as np
import torch
import torch.nn as nn

from fairseq import checkpoint_utils, utils, tasks
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
from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecCtc
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
    S2TTransformerEncoderLayer,
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


@register_model("s2t_w2v2_transformer")
class S2TW2V2TransformerModel(S2TTransformerModel):
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

        parser.add_argument("--w2v2-model-path", type=str, metavar="N",
                            help="path/to/wav2vec/model, support hdfs")
        parser.add_argument("--freeze-w2v", action="store_true",
                            help="if we want to freeze the w2v features")
        parser.add_argument("--use-asr-finetune-w2v", action="store_true",
                            help="if we want to load wav2vec2.0 asr finetuned data")
        pass

    @classmethod
    def build_encoder(cls, args, task=None, embed_tokens=None):
        encoder = S2TW2V2TransformerEncoder(args, task, embed_tokens)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from, strict=False
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )

        return encoder


class S2TW2V2TransformerEncoder(FairseqEncoder):
    """Speech-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args, task=None, embed_tokens=None):
        super().__init__(None)

        assert args.w2v2_model_path is not None
        self.w2v2_model_path = args.w2v2_model_path
        self.use_asr_finetune_w2v = args.use_asr_finetune_w2v

        ckpt = torch.load(self.w2v2_model_path)
        self.w2v_args = ckpt["args"]

        if not self.use_asr_finetune_w2v:  # if use ssl-trained only
            self.w2v_args = ckpt["args"]
            self.wav2vec_model = Wav2Vec2Model.build_model(ckpt['args'], task=None)
            self.wav2vec_model.load_state_dict(ckpt['model'])
        else:  # wav2vec-ctc model
            ckpt["args"].data = args.data
            if not os.path.exists(os.path.join(ckpt["args"].data, f"dict.{ckpt['args'].labels}.txt")):
                os.system(f"wget -P {ckpt['args'].data} https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt")

            task = tasks.setup_task(ckpt["args"])
            model_finetuned = Wav2VecCtc.build_model(ckpt["args"], task=task)
            model_finetuned.load_state_dict(ckpt['model'])
            self.wav2vec_model = model_finetuned.w2v_encoder.w2v_model
            self.w2v_args = ckpt["args"].w2v_args["model"]

        self.freeze_w2v = args.freeze_w2v
        # w2v_output_dim = 512
        w2v_output_dim = self.w2v_args.encoder_embed_dim

        self.encoder = S2TTransformerEncoder(args, task, embed_tokens)

        del self.encoder.subsample
        self.encoder.subsample = subsampling(args, in_dim=w2v_output_dim)

    def _get_w2v_feature(self, src_tokens, src_lengths):
        """
        :param src_tokens: b x frames
        :param src_lengths: b-dim length
        :return: w2v_feature: b x short_frames x feature-dim;
                w2v_lengths: b-dim tensor
                w2v_padding_mask: b x short_frames x feature-dim T/F tensor
        """
        padding_mask = lengths_to_padding_mask(src_lengths)
        # print("padding mask:", padding_mask.size())
        # print(padding_mask)
        # w2v_feature = self.wav2vec_model.feature_extractor(src_tokens).transpose(1,2)
        w2v_feature, padding_mask = self.wav2vec_model.extract_features(src_tokens, padding_mask)
        # print("after extraction, padding:", padding_mask)
        output_length = (1 - padding_mask.int()).sum(dim=1)
        # output_length = (torch.ones(padding_mask.size()) - padding_mask.int()).sum(dim=1)

        return w2v_feature, padding_mask, output_length

    def forward(self, src_tokens, src_lengths):
        # 1. wav2vec
        if self.freeze_w2v:
            with torch.no_grad():
                w2v_feature, encoder_padding_mask, input_lengths = self._get_w2v_feature(
                    src_tokens, src_lengths)
        else:
            w2v_feature, encoder_padding_mask, input_lengths = self._get_w2v_feature(
                src_tokens, src_lengths)

        return self.encoder.forward(w2v_feature, input_lengths)

    def reorder_encoder_out(self, encoder_out, new_order):
        return self.encoder.reorder_encoder_out(encoder_out, new_order)


@register_model_architecture(model_name="s2t_w2v2_transformer", arch_name="s2t_w2v2_transformer")
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
    args.inter_mixup_layer = getattr(args, "inter_mixup_layer", "-1")
    args.inter_mixup_beta = getattr(args, "inter_mixup_beta", 0.5)
    args.inter_mixup_prob = getattr(args, "inter_mixup_prob", 1)
    args.inter_mixup_ratio = getattr(args, "inter_mixup_ratio", 0.3)
    args.inter_mixup_keep_org = getattr(args, "inter_mixup_keep_org", False)

    # Wav2vec2.0 feature-extractor
    args.w2v2_model_path = getattr(args, "w2v2_model_path", "./wav2vec_small.pt")
    args.freeze_w2v = getattr(args, "freeze_w2v", False)    # default is false, 'store_true'
    args.use_asr_finetune_w2v = getattr(args, "use_asr_finetune_w2v", False)


@register_model_architecture("s2t_w2v2_transformer", "s2t_w2v2_transformer_s")
def s2t_w2v2_transformer_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)


@register_model_architecture("s2t_w2v2_transformer", "s2t_w2v2_transformer_s_relative")
def s2t_w2v2_transformer_s_relative(args):
    args.max_encoder_relative_length = 100
    args.max_decoder_relative_length = 20
    args.k_only = True
    s2t_w2v2_transformer_s(args)


@register_model_architecture("s2t_w2v2_transformer", "s2t_w2v2_transformer_xs")
def s2t_w2v2_transformer_xs(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 4)
    args.dropout = getattr(args, "dropout", 0.3)
    s2t_w2v2_transformer_s(args)


@register_model_architecture("s2t_w2v2_transformer", "s2t_w2v2_transformer_sp")
def s2t_w2v2_transformer_sp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_w2v2_transformer_s(args)


@register_model_architecture("s2t_w2v2_transformer", "s2t_w2v2_transformer_m")
def s2t_w2v2_transformer_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    base_architecture(args)


@register_model_architecture("s2t_w2v2_transformer", "s2t_w2v2_transformer_mp")
def s2t_w2v2_transformer_mp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_w2v2_transformer_m(args)


@register_model_architecture("s2t_w2v2_transformer", "s2t_w2v2_transformer_l")
def s2t_w2v2_transformer_l(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.2)
    base_architecture(args)


@register_model_architecture("s2t_w2v2_transformer", "s2t_w2v2_transformer_lp")
def s2t_w2v2_transformer_lp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_w2v2_transformer_l(args)
