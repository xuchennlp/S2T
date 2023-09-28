import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from random import choice
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import warnings

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
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    LegacyRelPositionalEncoding,
    RelPositionalEncoding,
    S2TTransformerEncoderLayer,
    DynamicLinearCombination,
)
from fairseq.modules.speech_to_text import subsampling

from torch import Tensor

logger = logging.getLogger(__name__)


@register_model("s2t_transformer")
class S2TTransformerModel(FairseqEncoderDecoderModel):
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
        # subsampling
        parser.add_argument(
            "--subsampling-type",
            type=str,
            help="subsampling type, like conv1d and conv2d",
        )
        parser.add_argument(
            "--subsampling-layers",
            type=int,
            help="subsampling layers",
        )
        parser.add_argument(
            "--subsampling-filter",
            type=int,
            help="subsampling filter",
        )
        parser.add_argument(
            "--subsampling-kernel",
            type=int,
            help="subsampling kernel",
        )
        parser.add_argument(
            "--subsampling-stride",
            type=int,
            help="subsampling stride",
        )
        parser.add_argument(
            "--subsampling-norm",
            type=str,
            default="none",
            help="subsampling normalization type",
        )
        parser.add_argument(
            "--subsampling-activation",
            type=str,
            default="none",
            help="subsampling activation function type",
        )
        # Transformer
        parser.add_argument(
            "--activation-fn",
            type=str,
            default="relu",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-type",
            type=str,
            default="selfattn",
            choices=[
                "local",
                "selfattn",
                "reduced",
                "rel_selfattn",
                "relative",
                "rel_pos_legacy",
                "rel_pos",
                "rope",
                "abs",
                "transfer",
                "reduced_rel_pos",
            ],
            help="transformer encoder self-attention layer type",
        )
        # transfer
        parser.add_argument(
            "--relative-pos-enc",
            action="store_true",
            help="use relative position encoding for attention",
        )
        parser.add_argument(
            "--linear-att",
            action="store_true",
            help="use linear attention",
        )

        # reduced attention
        parser.add_argument(
            "--attention-reduced-method",
            type=str,
            default="conv",
            help="reduction method for attention",
        )
        parser.add_argument(
            "--attention-reduced-q",
            action="store_true",
            help="use reduction for query or not",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-type",
            type=str,
            default="selfattn",
            choices=[
                "selfattn",
                "rel_selfattn",
                "relative",
                "local",
            ],
            help="transformer decoder self-attention layer type",
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument(
            "--share-all-embeddings",
            action="store_true",
            help="share encoder, decoder and output embeddings"
            " (requires shared dictionary and embed dim)",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        parser.add_argument(
            "--encoder-no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings in encoder",
        )
        parser.add_argument(
            "--adaptive-softmax-cutoff",
            metavar="EXPR",
            help="comma separated list of adaptive softmax cutoff points. "
            "Must be used with adaptive_loss criterion",
        ),
        parser.add_argument(
            "--adaptive-softmax-dropout",
            type=float,
            metavar="D",
            help="sets adaptive softmax dropout for the tail projections",
        )
        parser.add_argument(
            "--max-encoder-relative-length",
            type=int,
            default=-1,
            help="the max relative length",
        )
        parser.add_argument(
            "--max-decoder-relative-length",
            type=int,
            default=-1,
            help="the max relative length",
        )
        parser.add_argument(
            "--k-only",
            default=False,
            action="store_true",
            help="select the relative mode to map relative position information",
        )
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--load-pretrained-decoder-from",
            type=str,
            metavar="STR",
            help="model to take decoder weights from (for initialization)",
        )
        parser.add_argument(
            "--encoder-freeze-module",
            type=str,
            metavar="STR",
            help="freeze the module of the encoder",
        )
        parser.add_argument(
            "--decoder-freeze-module",
            type=str,
            metavar="STR",
            help="freeze the module of the decoder",
        )
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument(
            "--encoder-layerdrop",
            type=float,
            metavar="D",
            default=0,
            help="LayerDrop probability for encoder",
        )
        parser.add_argument(
            "--decoder-layerdrop",
            type=float,
            metavar="D",
            default=0,
            help="LayerDrop probability for decoder",
        )
        # DLCL
        parser.add_argument(
            "--use-enc-dlcl",
            default=False,
            action="store_true",
            help="use dlcl encoder",
        )
        parser.add_argument(
            "--use-dec-dlcl",
            default=False,
            action="store_true",
            help="use dlcl encoder",
        )
        parser.add_argument(
            "--init-value",
            type=str,
            default="avg",
            choices=["avg", "one"],
            help="how to init the learned weight matrix",
        )
        parser.add_argument(
            "--weight-type",
            type=str,
            default="scalar",
            help="type of learned weight [scalar, scalar_n(n>1), vector]",
        )
        parser.add_argument(
            "--encoder-learnable",
            type=eval,
            default="True",
            help="enable to learn weights for encoder",
        )
        parser.add_argument(
            "--decoder-learnable",
            type=eval,
            default="True",
            help="enable to learn weights for decoder",
        )
        parser.add_argument(
            "--normalize-learned-weight",
            type=eval,
            default="False",
            help="normalize learned weight by softmax",
        )
        parser.add_argument(
            "--normalize-embedding",
            type=eval,
            default="False",
            help="normalize the input of embedding",
        )
        parser.add_argument(
            "--history-dropout",
            type=float,
            default=0.0,
            metavar="D",
            help="dropout for history output",
        )
        parser.add_argument(
            "--history-window-size",
            type=int,
            default="-1",
            help="how many past layers are considered. -1 means all",
        )
        # CTC
        parser.add_argument(
            "--ctc-layer",
            default=0,
            type=int,
            help="the position of the ctc loss",
        )
        parser.add_argument(
            "--share-ctc-and-embed",
            action="store_true",
            help="share the weight of ctc and embedding",
        )

        # local modeling
        parser.add_argument(
            "--hard-mask-window",
            type=float,
            metavar="D",
            default=0,
            help="window size of local mask",
        )
        parser.add_argument(
            "--gauss-mask-sigma",
            type=float,
            metavar="D",
            default=0,
            help="standard deviation of the gauss mask",
        )
        parser.add_argument(
            "--init-mask-weight",
            type=float,
            metavar="D",
            default=0.5,
            help="initialized weight for local mask",
        )
        parser.add_argument(
            "--layer-padding-mask",
            default=False,
            type=bool,
            help="mask the padding to 0 before each layer",
        )
        # Conformer setting
        parser.add_argument(
            "--encoder-activation-fn",
            type=str,
            default="relu",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--macaron-style",
            default=False,
            type=bool,
            help="Whether to use macaron style for positionwise layer",
        )
        # Attention
        parser.add_argument(
            "--zero-triu",
            default=False,
            type=bool,
            help="If true, zero the upper triangular part of attention matrix.",
        )
        # Relative positional encoding
        parser.add_argument(
            "--rel-pos-type",
            type=str,
            default="legacy",
            choices=["legacy", "latest"],
            help="Whether to use the latest relative positional encoding or the legacy one."
            "The legacy relative positional encoding will be deprecated in the future."
            "More Details can be found in https://github.com/espnet/espnet/pull/2816.",
        )
        # CNN module
        parser.add_argument(
            "--use-cnn-module",
            default=False,
            type=bool,
            help="Use convolution module or not",
        )
        parser.add_argument(
            "--cnn-module-norm",
            default="batch_norm",
            type=str,
            help="normalization type of cnn module",
        )
        parser.add_argument(
            "--cnn-module-kernel",
            default=31,
            type=int,
            help="Kernel size of convolution module.",
        )
        parser.add_argument(
            "--encoder-embed-linear",
            action="store_true",
            help="use linear transform after down-sampling",
        )
        parser.add_argument(
            "--encoder-embed-norm",
            action="store_true",
            help="use layer norm after down-sampling",
        )
        parser.add_argument(
            "--layer-out-norm",
            action="store_true",
            help="use layer norm after each layer",
        )
        parser.add_argument(
            "--layer-out-norm-interval",
            default=1,
            type=int,
            help="the interval to use layer norm after each layer",
        )

        # intermediate CTC layers
        parser.add_argument(
            "--inter-ctc-layers",
            default=None,
            type=str,
            help="the position of intermediate ctc layers, separated by comma ",
        )

        parser.add_argument(
            "--inter-ctc-drop-prob",
            default=0,
            type=float,
            help="probability of dropping the followed layers",
        )
        parser.add_argument(
            "--share-inter-ctc",
            action="store_true",
            help="share the weight of all intermediate ctc modules",
        )
        parser.add_argument(
            "--share-inter-ctc-norm",
            action="store_true",
            help="share the weight of layer norm between inter ctc and final norm",
        )

        # CTC Prediction-aware encoding (PAE)
        parser.add_argument(
            "--ctc-pae",
            default="none",
            type=str,
            help="arch type of pae",
        )
        parser.add_argument(
            "--pae-shrink-strategy",
            default="avg",
            type=str,
            help="compress strategy of shrinking, such as avg, weighted, and softmax",
        )
        parser.add_argument(
            "--pae-drop-prob",
            default=0,
            type=float,
            help="dropping one input in pae with a probability",
        )
        parser.add_argument(
            "--pae-distribution-cutoff",
            default=None,
            type=int,
            help="cutoff of the distribution in pae",
        )
        parser.add_argument(
            "--pae-ctc-temperature",
            default=1,
            type=float,
            help="temperature of the CTC probability in pae",
        )
        parser.add_argument(
            "--pae-gumbel",
            action="store_true",
            help="use gumbel softmax in pae",
        )
        parser.add_argument(
            "--pae-distribution-hard",
            action="store_true",
            help="use hard distribution in pae",
        )
        parser.add_argument(
            "--ctc-pae-ground-truth-ratio",
            default=0,
            type=float,
            help="the ratio for ground truth in pae",
        )
        parser.add_argument(
            "--share-pae-and-ctc",
            action="store_true",
            help="share the weight of ctc and pae",
        )
        parser.add_argument(
            "--pae-embed-norm",
            default=False,
            action="store_true",
            help="use the layer norm for embed output",
        )
        parser.add_argument(
            "--pae-out-norm",
            default=False,
            action="store_true",
            help="use the layer norm for final output",
        )
        parser.add_argument(
            "--pae-linear-init",
            default=False,
            action="store_true",
            help="use the linear transform initialization for pae",
        )
        parser.add_argument(
            "--pae-unnorm-input",
            default=False,
            action="store_true",
            help="use the representation before layer norm for pae",
        )

        #  XCTC
        parser.add_argument(
            "--xctc-pae",
            type=str,
            help="adapter type of target pae ",
        )
        parser.add_argument(
            "--xctc-layer",
            default=0,
            type=int,
            help="xctc layer for target sentence",
        )
        parser.add_argument(
            "--axctc-layer",
            default=0,
            type=int,
            help="axctc layer for target sentence",
        )
        parser.add_argument(
            "--share-xctc-and-embed",
            action="store_true",
            help="share the weight of target ctc and embed",
        )
        parser.add_argument(
            "--share-xctc-and-ctc",
            action="store_true",
            help="share the weight of target ctc and ctc",
        )        
        parser.add_argument(
            "--inter-xctc-layers",
            default=None,
            type=str,
            help="intermediate xctc layers for target sentence",
        )
        parser.add_argument(
            "--inter-axctc-layers",
            default=None,
            type=str,
            help="intermediate axctc layers for target sentence",
        )
        parser.add_argument(
            "--share-pae-and-xctc",
            action="store_true",
            help="share the weight of target pae and ctc",
        )
        parser.add_argument(
            "--share-inter-xctc-norm",
            action="store_true",
            help="share the weight of layer norm between inter xctc and final norm",
        )
        parser.add_argument(
            "--xctc-pae-ground-truth-ratio",
            default=0,
            type=float,
            help="the ratio for ground truth in xctc",
        )
        parser.add_argument(
            "--xctc-pae-ground-truth-ratio-decay",
            default=None,
            type=str,
            help="the parameters for decay the ground truth ratio during training",
        )
        parser.add_argument(
            "--xctc-pae-ground-truth-ratio-adaptive",
            action="store_true",
            help="adaptively modify the ground truth ratio during training",
        )
        parser.add_argument(
            "--xctc-pae-ground-truth-only-mistake",
            action="store_true",
            help="only integrate the ground truth where the wrong prediction occurs",
        )
        parser.add_argument(
            "--pae-oracle-smooth",
            action="store_true",
            help="use smoothing prob for oracle",
        )

        # Mixup
        parser.add_argument(
            "--inter-mixup",
            action="store_true",
            help="use mixup or not",
        )
        parser.add_argument(
            "--inter-mixup-layer",
            default="-1",
            type=str,
            help="the layers to apply mixup",
        )
        parser.add_argument(
            "--inter-mixup-decoder-layer",
            default="0",
            type=str,
            help="the layers to apply mixup in the decoder",
        )
        parser.add_argument(
            "--inter-mixup-beta",
            default=0.5,
            type=float,
            help="the coefficient beta of mixup",
        )
        parser.add_argument(
            "--inter-mixup-prob",
            default=1,
            type=float,
            help="the probability of mixup",
        )
        parser.add_argument(
            "--inter-mixup-ratio",
            default=1,
            type=float,
            help="the ratio of mixup",
        )
        parser.add_argument(
            "--inter-mixup-keep-org",
            action="store_true",
            help="keep original batch",
        )
        parser.add_argument(
            "--inter-mixup-decoder-emb",
            action="store_true",
            help="mix the embedding in the decoder",
        )
        parser.add_argument(
            "--inter-mixup-ratio-decay",
            action="store_true",
            help="decay the mixup ratio during training",
        )
        parser.add_argument(
            "--inter-mixup-ratio-decay-params",
            default=None,
            type=str,
            help="the parameters for decay the mixup ratio during training",
        )

        # dynamic compression
        parser.add_argument(
            "--compression-metric",
            default="threshold",
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
            default="0.95",
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
        encoder = S2TTransformerEncoder(args, task, embed_tokens)
        if getattr(args, "load_pretrained_encoder_from", None):
            if not os.path.exists(args.load_pretrained_encoder_from):
                logger.warning(
                    f"No pretrained encoder path: "
                    f"{args.load_pretrained_encoder_from}"
                )
            else:
                encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder,
                    checkpoint=args.load_pretrained_encoder_from,
                    strict=False,
                )
                logger.info(
                    f"loaded pretrained encoder from: "
                    f"{args.load_pretrained_encoder_from}"
                )

        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):

        if getattr(args, "simul", False):
            from examples.simultaneous_translation.models.transformer_monotonic_attention import (
                TransformerMonotonicDecoder,
            )

            decoder = TransformerMonotonicDecoder(
                args, task.target_dictionary, embed_tokens
            )
        else:
            decoder = TransformerDecoderScriptable(
                args, task.target_dictionary, embed_tokens
            )

        if getattr(args, "load_pretrained_decoder_from", None):
            logger.info(
                f"loaded pretrained decoder from: "
                f"{args.load_pretrained_decoder_from}"
            )
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder,
                checkpoint=args.load_pretrained_decoder_from,
                strict=False,
            )

        return decoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )
        encoder = cls.build_encoder(args, task, decoder_embed_tokens)
        if getattr(args, "encoder_freeze_module", None):
            utils.freeze_parameters(encoder, args.encoder_freeze_module)
            logger.info(
                "freeze the encoder module: {}".format(args.encoder_freeze_module)
            )

        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        if getattr(args, "decoder_freeze_module", None):
            utils.freeze_parameters(decoder, args.decoder_freeze_module)
            logger.info(
                "freeze the decoder module: {}".format(args.decoder_freeze_module)
            )
        return cls(encoder, decoder)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(
            src_tokens=src_tokens, src_lengths=src_lengths, **kwargs
        )
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        return decoder_out


class S2TTransformerEncoder(FairseqEncoder):
    """Speech-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args, task=None, embed_tokens=None):
        super().__init__(None)

        dim = args.encoder_embed_dim
        layer_num = args.encoder_layers
        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(dim)
        if args.encoder_no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = 1

        self.subsample = subsampling(args)
        self.encoder_embed_linear = getattr(args, "encoder_embed_linear", False)
        self.encoder_embed_norm = getattr(args, "encoder_embed_norm", False)
        if self.encoder_embed_linear:
            self.linear = nn.Linear(dim, dim)
        if self.encoder_embed_norm:
            self.embed_ln = LayerNorm(dim)

        self.attn_type = getattr(args, "encoder_attention_type", "selfattn")
        if self.attn_type == "rel_pos":
            self.embed_positions = RelPositionalEncoding(
                args.max_source_positions, args.encoder_embed_dim
            )
        elif self.attn_type in ["rel_selfattn", "rel_pos_legacy"]:
            self.embed_positions = LegacyRelPositionalEncoding(
                args.encoder_embed_dim, args.dropout, args.max_source_positions
            )
        elif self.attn_type == "rope":
            self.embed_positions = None
        else:  # Use absolute positional embedding
            self.embed_positions = PositionalEmbedding(
                args.max_source_positions, args.encoder_embed_dim, self.padding_idx
            )

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for _ in range(args.encoder_layers)]
        )
        self.layer_padding_mask = args.layer_padding_mask
        self.layer_out_norm = getattr(args, "layer_out_norm", False)
        self.layer_out_norm_interval = getattr(args, "layer_out_norm_interval", 1)
        if self.layer_out_norm:
            for i in range(args.encoder_layers):
                if i % self.layer_out_norm_interval == 0:
                    ln = LayerNorm(dim)
                    setattr(self, "layer%d_norm" % i, ln)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(dim)
        else:
            self.layer_norm = None

        if args.use_enc_dlcl:
            self.history = DynamicLinearCombination(args, is_encoder=True)
        else:
            self.history = None

        self.pae_ground_truth_ratio = getattr(
            args, "ctc_pae_ground_truth_ratio", 0
        ) + getattr(args, "xctc_pae_ground_truth_ratio", 0)
        # CTC
        self.use_ctc = getattr(args, "ctc_weight", 0) > 0
        if self.use_ctc:
            self.ctc_layer = (
                args.encoder_layers if args.ctc_layer == 0 else args.ctc_layer
            )
            self.inter_ctc = True if self.ctc_layer != args.encoder_layers else False
            logger.info("CTC loss in layer %d." % self.ctc_layer)
            self.ctc = CTC(
                dim,
                dictionary_size=len(task.source_dictionary),
                dropout=args.dropout,
                need_layernorm=True if self.inter_ctc else False,
            )

            if (
                getattr(args, "share_ctc_and_embed", False)
                and task.source_dictionary == task.target_dictionary
                and embed_tokens is not None
                and dim == embed_tokens.embedding_dim
            ):
                self.ctc.ctc_projection.weight = embed_tokens.weight

        self.inter_ctc_drop_prob = args.inter_ctc_drop_prob
        self.share_inter_ctc = getattr(args, "share_inter_ctc", False)
        self.inter_ctc_layers = []
        self.use_inter_ctc = False
        if args.inter_ctc_layers is not None:
            self.use_inter_ctc = True
            self.share_inter_ctc_norm = args.share_inter_ctc_norm
            if self.share_inter_ctc_norm:
                logger.info(
                    "Share layer norm in intermediate CTC %s." % args.inter_ctc_layers
                )
            else:
                logger.info(
                    "Do not Share layer norm in intermediate CTC %s."
                    % args.inter_ctc_layers
                )

            inter_ctc_layers = args.inter_ctc_layers.split(",")
            inter_ctc_mlo = getattr(args, "inter_ctc_mlo", "")
            if inter_ctc_mlo != "":
                assert len(inter_ctc_mlo.split(":")) - 1 == len(inter_ctc_layers), (
                    inter_ctc_mlo,
                    inter_ctc_layers,
                )
                inter_ctc_mlo = [int(x) for x in inter_ctc_mlo.split(":")]
                if self.share_inter_ctc is True:
                    self.share_inter_ctc = False
                    logger.info(
                        "Overwrite the config share_inter_ctc to False for MLO."
                    )

            for layer_idx in inter_ctc_layers:
                layer_idx = int(layer_idx)
                if layer_idx <= 0:
                    layer_idx += args.encoder_layers

                if not self.share_inter_ctc_norm:
                    norm = LayerNorm(dim)
                    setattr(self, "ctc_norm%d" % layer_idx, norm)

                self.inter_ctc_layers.append(layer_idx)

            if not (self.use_ctc and self.share_inter_ctc):
                if not self.share_inter_ctc:
                    inter_idx = -1
                    for layer_idx in self.inter_ctc_layers:
                        inter_idx += 1
                        inter_ctc = CTC(
                            dim,
                            dictionary_size=len(
                                task.get_source_dictionary(
                                    inter_ctc_mlo[inter_idx] - 1
                                    if inter_ctc_mlo != ""
                                    else -1
                                )
                            ),
                            dropout=args.dropout,
                        )
                        setattr(self, "inter_ctc%d" % layer_idx, inter_ctc)
                        # inter_layer_norm = LayerNorm(dim)
                        # setattr(
                        # self, "inter_layer_norm%d" % layer_idx, inter_layer_norm
                        # )
                else:
                    self.ctc = CTC(
                        dim,
                        dictionary_size=len(task.source_dictionary),
                        dropout=args.dropout,
                    )
                    if (
                        getattr(args, "share_ctc_and_embed", False)
                        and task.source_dictionary == task.target_dictionary
                        and embed_tokens is not None
                        and dim == embed_tokens.embedding_dim
                    ):
                        self.ctc.ctc_projection.weight = embed_tokens.weight

            self.ctc_pae_ground_truth_ratio = getattr(
                args, "ctc_pae_ground_truth_ratio", 0
            )
            self.pae_unnorm_input = getattr(args, "pae_unnorm_input", False)
            if self.ctc_pae_ground_truth_ratio != 0:
                logger.info(
                    "CTC ground truth ratio: %.2f." % self.ctc_pae_ground_truth_ratio
                )
            strategy = {
                "embed_norm": getattr(args, "pae_embed_norm", False),
                "out_norm": getattr(args, "pae_out_norm", False),
                "ctc_shrink_strategy": getattr(args, "pae_shrink_strategy", None),
                "ctc_temperature": getattr(args, "pae_ctc_temperature", 1.0),
                "distribution_cutoff": getattr(args, "pae_distribution_cutoff", None),
                "gumbel": getattr(args, "pae_gumbel", False),
                "distribution_hard": getattr(args, "pae_distribution_hard", None),
                "gt_ratio": self.ctc_pae_ground_truth_ratio,
                "drop_prob": getattr(args, "pae_drop_prob", 0),
                "linear_init": getattr(args, "pae_linear_init", False),
            }

            if not self.share_inter_ctc:
                inter_idx = -1
                for layer_idx in self.inter_ctc_layers:
                    inter_idx += 1
                    pae = Adapter(
                        dim,
                        args.ctc_pae,
                        len(
                            task.get_source_dictionary(
                                inter_ctc_mlo[inter_idx] - 1
                                if inter_ctc_mlo != ""
                                else -1
                            )
                        ),
                        strategy=strategy,
                    )
                    inter_ctc = getattr(self, "inter_ctc%d" % layer_idx)
                    if args.share_pae_and_ctc and hasattr(pae, "embed_adapter"):
                        pae.embed_adapter.weight = inter_ctc.ctc_projection.weight
                    setattr(self, "pae%d" % layer_idx, pae)
            else:
                self.pae = Adapter(
                    dim,
                    args.ctc_pae,
                    len(task.source_dictionary),
                    strategy=strategy,
                )
                if args.share_pae_and_ctc and hasattr(self.pae, "embed_adapter"):
                    self.pae.embed_adapter.weight = self.ctc.ctc_projection.weight

        # XCTC
        self.use_xctc = (
            getattr(args, "disable_xctc", False) is False
            and getattr(args, "xctc_weight", 0) > 0
        )
        if self.use_xctc:
            self.xctc_layer = getattr(args, "xctc_layer", layer_num)
            if self.xctc_layer == 0:
                self.xctc_layer = layer_num
            self.inter_xctc = True if self.xctc_layer != layer_num else False
            logger.info("XCTC loss in layer %d" % self.xctc_layer)
            self.xctc = CTC(
                dim,
                dictionary_size=embed_tokens.num_embeddings
                if embed_tokens is not None
                else len(task.target_dictionary),
                dropout=args.dropout,
                need_layernorm=True if self.inter_xctc else False,
            )

            if (
                embed_tokens is not None
                and args.share_xctc_and_embed
                and self.xctc.ctc_projection.weight.size() == embed_tokens.weight.size()
            ):
                self.xctc.ctc_projection.weight = embed_tokens.weight

        self.inter_xctc_layers = []
        self.pae_adaptive_gt = getattr(
            args, "xctc_pae_ground_truth_ratio_adaptive", False
        )
        self.pae_gt_only_mistake = getattr(
            args, "xctc_pae_ground_truth_only_mistake", False
        )
        inter_xctc_layers = getattr(args, "inter_xctc_layers", None)
        if (
            getattr(args, "disable_xctc", False) is False
            and getattr(args, "inter_xctc_weight", 0) > 0
            and inter_xctc_layers is not None
            and inter_xctc_layers != "none"
            and len(inter_xctc_layers.split(",")) > 0
        ):
            self.inter_xctc_drop_prob = args.inter_ctc_drop_prob

            # Dynamic learning
            self.xctc_pae_ground_truth_ratio = getattr(
                args, "xctc_pae_ground_truth_ratio", 0
            )
            self.pae_unnorm_input = getattr(args, "pae_unnorm_input", False)
            self.pae_gt_decay = False
            decay_params = getattr(args, "xctc_pae_ground_truth_ratio_decay", None)

            if self.xctc_pae_ground_truth_ratio != 0:
                if decay_params is not None and len(decay_params.split(":")) == 3:
                    self.pae_gt_decay = True
                    params = [float(item) for item in decay_params.split(":")]
                    self.gt_decay_start_ratio = self.xctc_pae_ground_truth_ratio
                    (
                        self.gt_decay_start_step,
                        self.gt_decay_end_step,
                        self.gt_decay_end_ratio,
                    ) = params
                    self.gt_step_decay = (
                        self.gt_decay_start_ratio - self.gt_decay_end_ratio
                    ) / (self.gt_decay_end_step - self.gt_decay_start_step)
                    logger.info(
                        "PAE GT decay from step %d with ratio of %.2f end step %d with ratio of %.2f."
                        % (
                            self.gt_decay_start_step,
                            self.gt_decay_start_ratio,
                            self.gt_decay_end_step,
                            self.gt_decay_end_ratio,
                        )
                    )
                else:
                    logger.info(
                        "XCTC ground truth ratio: %.2f."
                        % self.xctc_pae_ground_truth_ratio
                    )

            self.share_inter_xctc_norm = args.share_inter_xctc_norm
            if self.share_inter_xctc_norm:
                logger.info(
                    "Share layer norm in intermediate XCTC %s." % inter_xctc_layers
                )
            else:
                logger.info(
                    "Do not Share layer norm in intermediate XCTC %s."
                    % inter_xctc_layers
                )

            inter_xctc_layers = inter_xctc_layers.split(",")
            for layer_idx in inter_xctc_layers:
                layer_idx = int(layer_idx)
                assert layer_idx <= layer_num, (layer_idx, layer_num)

                if layer_idx <= 0:
                    layer_idx += layer_num
                self.inter_xctc_layers.append(layer_idx)

                if not self.share_inter_xctc_norm:
                    xctc_norm = LayerNorm(dim)
                    setattr(self, "xctc_norm%d" % layer_idx, xctc_norm)

            # consider layer norm
            if not hasattr(self, "xctc"):
                self.xctc = CTC(
                    dim,
                    dictionary_size=len(task.target_dictionary),
                    dropout=args.dropout,
                )

                if (
                    embed_tokens is not None
                    and args.share_xctc_and_embed
                    and self.xctc.ctc_projection.weight.size()
                    == embed_tokens.weight.size()
                ):
                    self.xctc.ctc_projection.weight = embed_tokens.weight
                elif getattr(self, "share_xctc_and_ctc", False) and hasattr(self, "ctc"):
                    self.xctc.ctc_projection.weight = self.ctc.ctc_projection.weight

            strategy = {
                "embed_norm": getattr(args, "pae_embed_norm", False),
                "out_norm": getattr(args, "pae_out_norm", False),
                "ctc_shrink_strategy": getattr(args, "pae_shrink_strategy", None),
                "ctc_temperature": getattr(args, "pae_ctc_temperature", 1.0),
                "distribution_cutoff": getattr(args, "pae_distribution_cutoff", None),
                "gumbel": getattr(args, "pae_gumbel", False),
                "distribution_hard": getattr(args, "pae_distribution_hard", None),
                "drop_prob": getattr(args, "pae_drop_prob", 0),
                "gt_ratio": self.xctc_pae_ground_truth_ratio,
                "oracle_smooth": getattr(args, "pae_oracle_smooth", False),
                "linear_init": getattr(args, "pae_linear_init", False),
            }

            self.xctc_pae = Adapter(
                dim,
                args.xctc_pae,
                len(task.target_dictionary),
                strategy=strategy,
            )
            if args.share_pae_and_xctc and hasattr(self.xctc_pae, "embed_adapter"):
                self.xctc_pae.embed_adapter.weight = self.xctc.ctc_projection.weight

        # mixup
        self.mixup = getattr(args, "inter_mixup", False)
        self.mixup_ratio_decay = False
        if self.mixup:
            str_mixup_layer = args.inter_mixup_layer
            if len(str_mixup_layer.split(",")) == 1:
                self.mixup_layer = int(str_mixup_layer)
            else:
                self.mixup_layer = [int(layer) for layer in str_mixup_layer.split(",")]
            self.mixup_prob = args.inter_mixup_prob
            self.mixup_ratio = args.inter_mixup_ratio
            self.mixup_keep_org = args.inter_mixup_keep_org
            self.mixup_decoder_emb = args.inter_mixup_decoder_emb

            beta = args.inter_mixup_beta
            from torch.distributions import Beta

            self.beta = Beta(torch.Tensor([beta]), torch.Tensor([beta]))

            logger.info(
                "Use mixup in layer %s with beta %.2f, prob %.2f, ratio %.2f, keep original data %r."
                % (
                    str_mixup_layer,
                    beta,
                    self.mixup_prob,
                    self.mixup_ratio,
                    self.mixup_keep_org,
                )
            )

            self.mixup_ratio_decay = getattr(args, "inter_mixup_ratio_decay", False)
            if self.mixup_ratio_decay:
                decay_params = getattr(args, "inter_mixup_ratio_decay_params", None)
                if decay_params is not None and len(decay_params.split(",")) == 3:
                    params = [float(item) for item in decay_params.split(",")]
                    self.mixup_decay_start_ratio = self.mixup_ratio
                    (
                        self.mixup_decay_start_step,
                        self.mixup_decay_end_step,
                        self.mixup_decay_end_ratio,
                    ) = params
                    self.mixup_step_decay = (
                        self.mixup_ratio - self.mixup_decay_end_ratio
                    ) / (self.mixup_decay_end_step - self.mixup_decay_start_step)
                    logger.info(
                        "Mixup decay from step %d end step %d with ratio of %.2f."
                        % (
                            self.mixup_decay_start_step,
                            self.mixup_decay_end_step,
                            self.mixup_decay_end_ratio,
                        )
                    )
                else:
                    self.mixup_ratio_decay = False

        self.compression_metric = args.compression_metric
        self.compression_mode = args.compression_mode

        compression_num = (
            len(args.compression_layers.split(","))
            if args.compression_layers is not None
            else 0
        )
        if compression_num > 0:
            self.compression_layers = [
                int(i) for i in args.compression_layers.split(",")
            ]

            thresholds = [float(n) for n in args.compression_threshold.split(",")]
            assert len(thresholds) == 1 or len(thresholds) == compression_num
            compression_threshold = (
                thresholds
                if len(thresholds) == compression_num
                else thresholds * compression_num
            )
            for i, layer in enumerate(self.compression_layers):
                threshold = compression_threshold[i]
                setattr(self, "compression_threshold%d" % layer, threshold)
                logger.info(
                    "Compression with threshold %.3f in layer %d." % (threshold, layer)
                )

            ratios = [float(n) for n in args.compression_ratio.split(",")]
            assert len(ratios) == 1 or len(ratios) == compression_num
            self.compression_ratio = (
                ratios if len(ratios) == compression_num else ratios * compression_num
            )

            self.compression_pos = args.compression_pos
            if self.compression_pos:
                if self.attn_type == "rel_pos":
                    self.compression_embed_positions = RelPositionalEncoding(
                        args.max_source_positions, args.encoder_embed_dim
                    )
                elif self.attn_type in ["rel_selfattn", "rel_pos_legacy"]:
                    self.compression_embed_positions = LegacyRelPositionalEncoding(
                        args.encoder_embed_dim, args.dropout, args.max_source_positions
                    )
                elif self.attn_type == "rope":
                    self.compression_embed_positions = None
                else:  # Use absolute positional embedding
                    self.compression_embed_positions = PositionalEmbedding(
                        args.max_source_positions,
                        args.encoder_embed_dim,
                        self.padding_idx,
                    )

            self.compression_norm = args.compression_norm
            if self.compression_norm:
                for layer in self.compression_layers:
                    norm = LayerNorm(args.encoder_embed_dim)
                    setattr(self, "compression_norm%d" % layer, norm)
        else:
            self.compression_layers = []

        self.compression_stat = False

        # gather cosine similarity
        self.gather_cos_sim = getattr(args, "gather_cos_sim", False)
        self.gather_cos_sim_dis = 2
        self.cos_sim = dict()

        # debug the variance
        self.debug_var = False

        self.update_num = 0
        self.curr_temp = 0

        self.save_rep = False
        self.mixup_infer = False
        self.rep_dict = dict()

    @staticmethod
    def build_encoder_layer(args):
        return S2TTransformerEncoderLayer(args)

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.update_num = num_updates
        if self.mixup_ratio_decay:
            if (
                self.mixup_decay_start_step
                < self.update_num
                < self.mixup_decay_end_step
            ):
                self.mixup_ratio = (
                    self.mixup_decay_start_ratio
                    - self.mixup_step_decay
                    * (self.update_num - self.mixup_decay_start_step)
                )

    def dump(self, fstream, info=""):
        for i, layer in enumerate(self.layers):
            layer.dump(fstream, "%s Layer %d" % (info, i)) if hasattr(
                layer, "dump"
            ) else None

        if self.gather_cos_sim:
            print(
                "\nCosine similarity of distance %d" % self.gather_cos_sim_dis,
                file=fstream,
            )
            for idx, sim in self.cos_sim.items():
                sim = sum(sim) / len(sim) * 100
                print("%s %s: %f" % (info, idx, sim), file=fstream)

        if self.save_rep:
            out_dict = dict()
            for key, value in self.rep_dict.items():
                if key != "mixup":
                    if len(value) == 0:
                        return
                    reps = torch.cat(value, dim=0)
                    out_dict[key] = reps
                else:
                    out_dict["mixup"] = dict()
                    for key, value in self.rep_dict["mixup"].items():
                        if len(value) == 0:
                            return
                        reps = torch.cat(value, dim=0)
                        out_dict["mixup"][key] = reps

            path = os.path.dirname(os.path.realpath(fstream.name))

            x = out_dict
            N = 200
            plt.figure(figsize=(10, 2))
            plt.subplots_adjust(
                left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.5
            )

            keep_idx = [-1, 4, 8, 12]
            fig_idx = 0
            for key, value in x.items():
                if key == "mixup" or value == None:
                    continue
                origin = value.cpu().numpy()
                mixup = x["mixup"][key].cpu().numpy()
                logger.info("Origin sample size: %d." % origin.shape[0])
                logger.info("Mixup sample size: %d." % mixup.shape[0])
                if N != -1 and (N < origin.shape[0] or N < mixup.shape[0]):
                    row_rand_array = np.arange(origin.shape[0])
                    np.random.shuffle(row_rand_array)
                    if N < origin.shape[0]:
                        origin = origin[row_rand_array[0:N]]
                    row_rand_array = np.arange(mixup.shape[0])
                    np.random.shuffle(row_rand_array)
                    if N < mixup.shape[0]:
                        mixup = mixup[row_rand_array[0:N]]
                    logger.info("After random selection")
                    logger.info("Origin sample size: %d." % origin.shape[0])
                    logger.info("Mixup sample size: %d." % mixup.shape[0])
                org_size = origin.shape[0]
                mixup_size = mixup.shape[0]
                all = np.concatenate((origin, mixup), axis=0)
                # label = [0] * org_size + [1] * mixup_size

                tsne = TSNE(n_components=2, init="pca", random_state=0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = tsne.fit_transform(all)
                x_min, x_max = np.min(result, 0), np.max(result, 0)
                data = (result - x_min) / (x_max - x_min)

                def find_center(points):
                    sum_x = 0
                    sum_y = 0
                    for point in points:
                        sum_x += point[0]
                        sum_y += point[1]
                    center_x = sum_x / len(points)
                    center_y = sum_y / len(points)
                    return (center_x, center_y)

                org_center = find_center(data[: org_size])
                mixup_center = find_center(data[org_size:])

                print(
                    "Layer %d L2 distance between interpolated samples and origin samples: %f" % (key, math.sqrt(
                        math.pow(org_center[1] - mixup_center[1], 2) + math.pow(org_center[0] - mixup_center[0], 2)
                    )),
                file=fstream,
                )

                org_color = "teal"
                mix_color = "coral"
                org_shape = "s"
                mix_shape = "o"

                cen_org_color = "darkred"
                cen_mix_color = "darkblue"
                cen_org_shape = "^"
                cen_mix_shape = "v"

                if key in keep_idx:
                    fig_idx += 1
                    plt.subplot(1, 4, fig_idx)
                    for i in range(data.shape[0]):
                        plt.scatter(
                            data[i, 0],
                            data[i, 1],
                            marker=org_shape if i < org_size else mix_shape,
                            color=org_color if i < org_size else mix_color,
                            s=2,
                        )

                    plt.scatter(org_center[0], org_center[1], marker=cen_org_shape, color=cen_org_color, s=100, label='Center')
                    plt.scatter(mixup_center[0], mixup_center[1], marker=cen_mix_shape, color=cen_mix_color, s=100, label='Center')

                    plt.xlim((-0.1, 1.1))
                    plt.ylim((-0.1, 1.1))
                    plt.xticks([0, 0.5, 1])
                    plt.yticks([0, 0.5, 1])

                    if key == -1:
                        title = "Input"
                    elif key == 13:  # or key ==4 or key == 10:
                        title = "Output"
                    else:
                        title = "Layer " + str(key)
                    plt.title(title, fontsize=12)

            save_path = os.path.join(path, "mixup_enc.eps")
            plt.savefig(save_path, dpi=600)
            save_path = os.path.join(path, "mixup_enc.png")
            plt.savefig(save_path, dpi=600)
            logger.info("Saving figure of representation to path:" + save_path)

    def set_flag(self, **kwargs):
        for layer in self.layers:
            if hasattr(layer, "set_flag"):
                layer.set_flag(**kwargs)

        self.save_rep = kwargs.get("save_rep", False)
        self.mixup_infer = kwargs.get("mixup_infer", False)
        self.gather_cos_sim = kwargs.get("gather_cos_sim", False)
        self.gather_cos_sim_dis = kwargs.get("gather_cos_sim_dis", 2)

        if self.mixup_infer:
            self.mixup_keep_org = True

    def set_ctc_infer(
        self, ctc_infer, post_process, src_dict=None, tgt_dict=None, path=None
    ):
        if hasattr(self, "ctc"):
            assert src_dict is not None
            self.ctc.set_infer(
                ctc_infer,
                post_process,
                src_dict,
                path=os.path.splitext(path)[0] + ".ctc" if path is not None else None,
            )
        
        if hasattr(self, "xctc"):
            assert tgt_dict is not None
            self.xctc.set_infer(
                ctc_infer,
                post_process,
                tgt_dict,
                path=os.path.splitext(path)[0] + ".xctc" if path is not None else None,
            )        

    def ctc_valid(self, lprobs, targets, input_lengths, dictionary, lang="source"):
        if hasattr(self, "ctc"):
            return self.ctc.valid(lprobs, targets, input_lengths, dictionary)
        if lang == "target" and hasattr(self, "xctc"):
            return self.xctc.valid(lprobs, targets, input_lengths, dictionary)

        logger.error("No CTC module in the encoder.")

    def set_debug_var(self, debug_var_flag):
        self.debug_var = debug_var_flag

    @staticmethod
    def pooling_ratio():
        return 4

    def add_to_dict(self, x, idx):
        if not self.gather_cos_sim:
            return

        dis = self.gather_cos_sim_dis
        sim = 0
        seq_len = x.size(0)
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        for i in range(dis, seq_len - dis):
            a = x[i, :, :]
            for j in range(-dis, dis + 1):
                if j == 0:
                    continue
                b = x[i + j, :, :]
                sim_j = cos(a, b).mean()
                sim += sim_j
        if sim == 0:
            return

        sim = sim / 2 / dis / (seq_len - 2 * dis)

        if idx not in self.cos_sim:
            self.cos_sim[idx] = []
        self.cos_sim[idx].append(float(sim))

    def apply_mixup(self, x, encoder_padding_mask):

        if encoder_padding_mask is not None:
            mask_pad = encoder_padding_mask.unsqueeze(2)
            x = x.transpose(0, 1)
            x.masked_fill_(mask_pad, 0.0)
            x = x.transpose(0, 1)

        batch = x.size(1)
        org_indices = np.arange(batch)

        mixup_size = int(batch * self.mixup_ratio)
        mixup_flag = []

        if self.mixup_keep_org:
            mixup_index1 = np.random.randint(0, batch, mixup_size)
            mixup_index2 = np.random.randint(0, batch, mixup_size)

            idx1 = np.append(org_indices, mixup_index1)
            idx2 = np.append(org_indices, mixup_index2)
            mixup_flag.extend([0] * len(org_indices))
            mixup_flag.extend([1] * len(mixup_index1))
        else:
            if mixup_size <= batch:
                mixup_index1 = np.random.permutation(mixup_size)
                mixup_index2 = np.random.permutation(mixup_size)
            else:
                mixup_index1 = np.random.randint(0, batch, mixup_size)
                mixup_index2 = np.random.randint(0, batch, mixup_size)

            org_size = batch - mixup_size
            if org_size > 0:
                keep_indices = org_indices[mixup_size:]
                # keep_indices = np.random.permutation(batch)[:org_size]
                idx1 = np.append(keep_indices, mixup_index1)
                idx2 = np.append(keep_indices, mixup_index2)
                mixup_flag.extend([0] * len(keep_indices))
            else:
                idx1 = mixup_index1
                idx2 = mixup_index2
            mixup_flag.extend([1] * len(mixup_index1))

        idx1 = torch.from_numpy(idx1).to(x.device).long()
        idx2 = torch.from_numpy(idx2).to(x.device).long()

        x1 = x[:, idx1]
        x2 = x[:, idx2]

        coef = self.beta.sample([len(idx1)]).to(x.device).type_as(x).view(-1)
        mixup_coef = coef.view(1, -1, 1)
        x = mixup_coef * x1 + (1 - mixup_coef) * x2
        x = x.contiguous()

        pad1 = encoder_padding_mask[idx1]
        pad2 = encoder_padding_mask[idx2]
        encoder_padding_mask = pad1 & pad2
        input_lengths = (~encoder_padding_mask).sum(-1)
        mixup_flag = torch.Tensor(mixup_flag).to(x.device).bool()

        mixup = {
            "ratio": self.mixup_ratio,
            "keep_org": self.mixup_keep_org,
            "coef": coef,
            "index1": idx1,
            "index2": idx2,
            "mixup_flag": mixup_flag,
            "mixup_decoder_emb": self.mixup_decoder_emb,
        }
        return x, encoder_padding_mask, input_lengths, mixup

    def show_debug(self, x, text=None):
        if not self.debug_var:
            return
        if text:
            logger.info("--- Variance of %s: %f." % (text, x.var()))
        else:
            logger.info("--- Variance: %f." % (x.var()))

    def add_rep(self, x, layer_idx, mixup=None):
        if mixup is None:
            self.save_rep = False
        if not self.save_rep:
            return

        rep_dict = self.rep_dict

        if "mixup" not in self.rep_dict:
            self.rep_dict["mixup"] = dict()
        mixup_rep_dict = self.rep_dict["mixup"]

        if layer_idx not in mixup_rep_dict:
            mixup_rep_dict[layer_idx] = []
        else:
            pass
            # return
        if layer_idx not in rep_dict:
            rep_dict[layer_idx] = []

        flag = mixup["mixup_flag"]
        if any(flag):
            mixup_x = x[:, flag, :].mean(0)
            mixup_rep_dict[layer_idx].append(mixup_x)
        if not all(flag):
            org_x = x[:, ~flag, :].mean(0)
            rep_dict[layer_idx].append(org_x)

    def forward(self, src_tokens, src_lengths=None, **kwargs):

        layer_idx = -1
        mixup = None
        if self.mixup:
            if type(self.mixup_layer) is list:
                mixup_layer = choice(self.mixup_layer)
            else:
                mixup_layer = self.mixup_layer

        if self.history is not None:
            self.history.clean()

        # (B, T, D) -> (T, B, D)
        x = src_tokens.transpose(0, 1)
        input_lengths = src_lengths

        if (
            (self.training or self.mixup_infer)
            and self.mixup
            and layer_idx == mixup_layer
        ):
            if torch.rand(1) < self.mixup_prob:
                encoder_padding_mask = lengths_to_padding_mask(input_lengths)
                x, encoder_padding_mask, input_lengths, mixup = self.apply_mixup(
                    x, encoder_padding_mask
                )

        self.show_debug(x, "input x")
        # gather info
        self.add_to_dict(x, "input")
        self.add_rep(x, layer_idx, mixup)

        # down-sampling
        x, input_lengths = self.subsample(x, input_lengths)
        self.show_debug(x, "x after subsampling")

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        if encoder_padding_mask.size(1) < x.size(0):
            bsz = encoder_padding_mask.size(0)
            miss_num = x.size(0) - encoder_padding_mask.size(1)
            miss_pad = (
                torch.Tensor([False])
                .bool()
                .to(x.device)
                .unsqueeze(1)
                .repeat([bsz, miss_num])
            )
            encoder_padding_mask = torch.cat([miss_pad, encoder_padding_mask], dim=1)

        if encoder_padding_mask is not None:
            x = x * (1 - encoder_padding_mask.transpose(0, 1).unsqueeze(-1).type_as(x))
        self.add_to_dict(x, "after sampling")

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
            self.show_debug(positions, "position embedding")
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

        layer_idx += 1
        if self.training and self.mixup and layer_idx == mixup_layer:
            if torch.rand(1) <= self.mixup_prob:
                x, encoder_padding_mask, input_lengths, mixup = self.apply_mixup(
                    x, encoder_padding_mask
                )

        # gather info
        self.add_to_dict(x, "before encoding")
        self.add_rep(x, layer_idx, mixup)
        self.show_debug(x, "x before encoding")

        ctc_logit = None
        inter_ctc_logits = []
        xctc_logit = None
        inter_xctc_logits = []
        # CTC alignment
        ctc_oracle = None
        ctc_oracle_mask = None
        ctc_force_emit = None
        xctc_oracle = None
        xctc_oracle_mask = None
        xctc_force_emit = None

        for layer in self.layers:
            if self.history is not None:
                x = self.history.pop()

            if (
                self.layer_padding_mask
                and encoder_padding_mask is not None
                and not torch.all(encoder_padding_mask)
            ):
                mask_pad = encoder_padding_mask.unsqueeze(2)
                x = x.transpose(0, 1)
                x = x.masked_fill(mask_pad, 0.0)
                x = x.transpose(0, 1)

            if self.attn_type in [
                "rel_pos",
                "rel_pos_legacy",
                "rel_selfattn",
            ] and positions.size(0) != x.size(0):
                positions = self.embed_positions(x)

            # encoder layer
            x = layer(x, encoder_padding_mask, pos_emb=positions)

            if self.layer_out_norm and layer_idx % self.layer_out_norm_interval == 0:
                ln = getattr(self, "layer%d_norm" % layer_idx)
                x = ln(x)
            layer_idx += 1

            # gather info
            self.add_to_dict(x, "layer%s" % layer_idx)
            self.add_rep(x, layer_idx, mixup)
            self.show_debug(x, "x after layer %d" % layer_idx)

            if self.training and self.mixup and layer_idx == mixup_layer:
                if torch.rand(1) < self.mixup_prob:
                    x, encoder_padding_mask, input_lengths, mixup = self.apply_mixup(
                        x, encoder_padding_mask
                    )

            if self.use_ctc and self.inter_ctc and self.ctc_layer == layer_idx:
                ctc_logit = self.ctc(
                    x.clone(),
                    encoder_padding_mask,
                    "CTC Layer %d" % layer_idx,
                    is_top=True,
                )

            if self.use_xctc and self.inter_xctc and self.xctc_layer == layer_idx:
                xctc_logit = self.xctc(
                    x.clone(),
                    encoder_padding_mask,
                    "XCTC layer %d" % layer_idx,
                    is_top=True,
                )

            # Inter CTC
            if layer_idx in self.inter_ctc_layers:
                if self.inter_ctc_drop_prob > 0:
                    p = torch.rand(1).uniform_()
                    if p < self.inter_ctc_drop_prob:
                        break

                if self.share_inter_ctc:
                    inter_ctc = self.ctc
                    pae = self.pae
                else:
                    inter_ctc = getattr(self, "inter_ctc%d" % layer_idx)
                    pae = getattr(self, "pae%d" % layer_idx)

                if self.share_inter_ctc_norm:
                    layer_norm = self.layer_norm
                else:
                    layer_norm = getattr(self, "ctc_norm%d" % layer_idx)
                norm_x = layer_norm(x)
                logit = inter_ctc(
                    norm_x, encoder_padding_mask, "Source Layer %d" % layer_idx
                )

                inter_logit = [logit, encoder_padding_mask]
                if self.ctc_pae_ground_truth_ratio > 0:
                    ctc_alignment_oracle = kwargs.get("ctc_alignment_oracle", None)
                    if (
                        ctc_alignment_oracle is not None
                        and ctc_alignment_oracle.get("ctc", None) is not None
                    ):
                        if ctc_oracle is None:
                            (
                                ctc_oracle,
                                best_aligns_pad,
                                mistake_flag,
                                mistake_ratio,
                            ) = ctc_alignment_oracle["ctc"]
                            
                            if self.pae_adaptive_gt:
                                prob = (
                                    self.ctc_pae_ground_truth_ratio
                                    * mistake_ratio.unsqueeze(-1)
                                )
                            else:
                                prob = self.ctc_pae_ground_truth_ratio
                            ctc_oracle_mask = (
                                torch.rand(
                                    ctc_oracle.size(), device=ctc_oracle.device
                                )
                                < prob
                            ).bool()
                            if self.pae_gt_only_mistake:
                                ctc_oracle_mask.masked_fill_(~mistake_flag, False)
                            ctc_force_emit = best_aligns_pad.masked_fill(
                                ~ctc_oracle_mask, -1
                            )

                        inter_logit = [logit, None, ctc_force_emit]

                pae_input = x if self.pae_unnorm_input else norm_x
                if pae.adapter_type != "none":
                    x, encoder_padding_mask = pae(
                        [pae_input, logit], encoder_padding_mask, ctc_oracle, ctc_oracle_mask
                    )
                    self.show_debug(x, "x after pae")

                inter_ctc_logits.append(inter_logit)

                if layer_idx in self.compression_layers:
                    ctc_prob = utils.softmax(logit, dim=-1)  # (T B C)
                    blank_prob = ctc_prob[:, :, 0]
                    bsz = x.size(1)

                    if self.compression_metric == "threshold":
                        threshold = getattr(self, "compression_threshold%d" % layer_idx)
                        keep_flag = blank_prob < threshold
                        keep_flag = keep_flag.masked_fill(
                            encoder_padding_mask.clone().transpose(0, 1), False
                        )
                        max_len = max(keep_flag.sum(0))
                        min_len = min(keep_flag.sum(0))

                        if self.compression_mode == "create":
                            if min_len > 0 and max_len > 0 and not keep_flag.all():
                                max_len = max(keep_flag.sum(0))
                                out = x.new_zeros(max_len, bsz, x.size(2))
                                out_encoder_padding_mask = (
                                    encoder_padding_mask.new_ones(bsz, max_len).bool()
                                )

                                for i in range(bsz):
                                    item_flag = keep_flag[:, i]
                                    org_tensor = x[:, i, :]
                                    new_tensor = org_tensor[item_flag]
                                    out[: new_tensor.size(0), i, :] = new_tensor
                                    out_encoder_padding_mask[
                                        i, : new_tensor.size(0)
                                    ] = False

                                encoder_padding_mask = (
                                    out_encoder_padding_mask.contiguous()
                                )
                                x = out.contiguous()
                        elif self.compression_mode == "mask":
                            encoder_padding_mask = encoder_padding_mask | (
                                ~keep_flag
                            ).transpose(0, 1)

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

                    if self.compression_norm:
                        norm = getattr(self, "compression_norm%d" % layer_idx)
                        x = norm(x)
                    if self.compression_pos:
                        if self.attn_type in [
                            "rel_pos",
                            "rel_pos_legacy",
                            "rel_selfattn",
                        ]:
                            positions = self.compression_embed_positions(x)

                        elif self.attn_type == "rope":
                            positions = None

                        else:
                            positions = self.compression_embed_positions(
                                encoder_padding_mask
                            ).transpose(0, 1)
                            x += positions
                            positions = None

                    x = x.transpose(0, 1)
                    x = x.masked_fill(encoder_padding_mask.unsqueeze(2), 0.0)
                    x = x.transpose(0, 1)

            # Inter XCTC
            if layer_idx in self.inter_xctc_layers:
                if self.inter_xctc_drop_prob > 0:
                    p = torch.rand(1).uniform_()
                    if p < self.inter_xctc_drop_prob:
                        break

                if self.share_inter_xctc_norm:
                    norm_x = self.layer_norm(x)
                else:
                    norm = getattr(self, "xctc_norm%d" % layer_idx)
                    norm_x = norm(x)

                logit = self.xctc(
                    norm_x, encoder_padding_mask, "Inter XCTC layer %d" % layer_idx
                )

                inter_logit = logit
                # CTC alignment
                if self.xctc_pae_ground_truth_ratio > 0:
                    ctc_alignment_oracle = kwargs.get("ctc_alignment_oracle", None)
                    if (
                        ctc_alignment_oracle is not None
                        and ctc_alignment_oracle.get("xctc", None) is not None
                    ):
                        if xctc_oracle is None:
                            (
                                xctc_oracle,
                                best_aligns_pad,
                                mistake_flag,
                                mistake_ratio,
                            ) = ctc_alignment_oracle["xctc"]
                            if self.pae_adaptive_gt:
                                prob = (
                                    self.xctc_pae_ground_truth_ratio
                                    * mistake_ratio.unsqueeze(-1)
                                )
                            else:
                                prob = self.xctc_pae_ground_truth_ratio
                            xctc_oracle_mask = (
                                torch.rand(
                                    xctc_oracle.size(), device=xctc_oracle.device
                                )
                                < prob
                            ).bool()
                            if self.pae_gt_only_mistake:
                                xctc_oracle_mask.masked_fill_(~mistake_flag, False)
                            xctc_force_emit = best_aligns_pad.masked_fill(
                                ~xctc_oracle_mask, -1
                            )
                        inter_logit = [logit, None, xctc_force_emit]

                pae_input = x if self.pae_unnorm_input else norm_x
                if self.xctc_pae.adapter_type != "none":
                    x, encoder_padding_mask = self.xctc_pae(
                        [pae_input, logit],
                        encoder_padding_mask,
                        xctc_oracle,
                        xctc_oracle_mask,
                    )

                inter_xctc_logits.append(inter_logit)

            if self.history is not None:
                self.history.push(x)

        if self.history is not None:
            x = self.history.pop()

        self.show_debug(x, "x after encoding")
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        layer_idx += 1

        self.add_rep(x, layer_idx, mixup)
        self.show_debug(x, "x after encoding layer norm")

        if self.training and self.mixup and layer_idx == mixup_layer:
            if torch.rand(1) < self.mixup_prob:
                x, encoder_padding_mask, input_lengths, mixup = self.apply_mixup(
                    x, encoder_padding_mask
                )

        if self.use_ctc and ctc_logit is None:
            ctc_logit = self.ctc(x, encoder_padding_mask, "Encoder output", is_top=True)
            self.show_debug(x, "x after ctc")

        if self.use_xctc and xctc_logit is None:
            xctc_logit = self.xctc(
                x, encoder_padding_mask, "Encoder output", is_top=True
            )
            self.show_debug(x, "x after xctc")

        if ctc_force_emit is not None:
            ctc_logit = [ctc_logit, None, ctc_force_emit]

        if xctc_force_emit is not None:
            xctc_logit = [xctc_logit, None, xctc_force_emit]

        return {
            "encoder_out": [x],  # T x B x C
            "ctc_logit": [] if ctc_logit is None else [ctc_logit],  # T x B x C
            "inter_ctc_logits": inter_ctc_logits,  # T x B x C
            "xctc_logit": [] if xctc_logit is None else [xctc_logit],  # B x T x C
            "inter_xctc_logits": inter_xctc_logits,  # B x T x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "mixup": mixup,
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            []
            if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_ctc_logit = (
            []
            if len(encoder_out["ctc_logit"]) == 0
            else [
                x.index_select(1, new_order)
                for x in encoder_out["ctc_logit"]
                if x is not None
            ]
        )
        new_xctc_logit = (
            []
            if len(encoder_out["xctc_logit"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["xctc_logit"]]
        )
        new_encoder_padding_mask = (
            []
            if len(encoder_out["encoder_padding_mask"]) == 0
            else [
                x.index_select(0, new_order)
                for x in encoder_out["encoder_padding_mask"]
            ]
        )

        new_encoder_embedding = (
            []
            if len(encoder_out["encoder_embedding"]) == 0
            else [
                x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]
            ]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "ctc_logit": new_ctc_logit,  # T x B x C
            "xctc_logit": new_xctc_logit,
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


@register_model_architecture(model_name="s2t_transformer", arch_name="s2t_transformer")
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
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)

    args.layer_padding_mask = getattr(args, "layer_padding_mask", False)

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
    args.init_value = getattr(args, "init_value", "avg")
    args.weight_type = getattr(args, "weight_type", "scalar")
    args.encoder_learnable = getattr(args, "encoder_learnable", True)
    args.decoder_learnable = getattr(args, "decoder_learnable", True)
    args.normalize_embed = getattr(args, "normalize_embed", False)
    args.history_dropout = getattr(args, "history_dropout", 0.0)
    args.history_window_size = getattr(args, "history_window_size", -1)

    # Relative position encoding
    args.max_encoder_relative_length = getattr(args, "max_encoder_relative_length", -1)
    args.max_decoder_relative_length = getattr(args, "max_decoder_relative_length", -1)
    args.k_only = getattr(args, "k_only", True)

    # local modeling
    args.hard_mask_window = getattr(args, "hard_mask_window", 0)
    args.gauss_mask_sigma = getattr(args, "gauss_mask_sigma", 0)
    args.init_mask_weight = getattr(args, "init_mask_weight", 0)

    args.layer_out_norm = getattr(args, "layer_out_norm", False)
    args.layer_out_norm_interval = getattr(args, "layer_out_norm_interval", False)

    # intermediate CTC
    args.inter_ctc_layers = getattr(args, "inter_ctc_layers", None)
    args.share_inter_ctc_norm = getattr(args, "share_inter_ctc_norm", False)
    args.pae_ctc_temperature = getattr(args, "pae_ctc_temperature", 1)
    args.inter_ctc_drop_prob = getattr(args, "inter_ctc_drop_prob", 0)
    args.ctc_pae_ground_truth_ratio = getattr(args, "ctc_pae_ground_truth_ratio", 0)

    # Prediction-aware encoding (pae)
    args.ctc_pae = getattr(args, "ctc_pae", "none")
    args.pae_shrink_strategy = getattr(args, "pae_shrink_strategy", "avg")
    args.share_pae_and_ctc = getattr(args, "share_pae_and_ctc", False)
    args.pae_embed_norm = getattr(args, "pae_embed_norm", False)
    args.pae_out_norm = getattr(args, "pae_out_norm", False)
    args.pae_drop_prob = getattr(args, "pae_drop_prob", 0)
    args.pae_distribution_cutoff = getattr(args, "pae_distribution_cutoff", None)
    args.pae_distribution_hard = getattr(args, "pae_distribution_hard", False)
    args.pae_gumbel = getattr(args, "pae_gumbel", False)
    args.pae_linear_init = getattr(args, "pae_linear_init", False)
    args.pae_unnorm_input = getattr(args, "pae_unnorm_input", False)

    # XCTC
    args.xctc_layer = getattr(args, "xctc_layer", 0)
    args.share_xctc_and_embed = getattr(args, "share_xctc_and_embed", False)
    args.share_xctc_and_ctc = getattr(args, "share_xctc_and_ctc", False)
    args.xctc_pae = getattr(args, "xctc_pae", args.ctc_pae)
    args.axctc_pae = getattr(args, "axctc_pae", args.xctc_pae)
    args.share_pae_and_xctc = getattr(args, "share_pae_and_xctc", False)
    args.inter_xctc_layers = getattr(args, "inter_xctc_layers", None)
    args.axctc_layer = getattr(args, "axctc_layer", None)
    args.inter_axctc_layers = getattr(args, "inter_axctc_layers", None)
    args.share_inter_xctc_norm = getattr(args, "share_inter_xctc_norm", False)
    args.share_inter_axctc_norm = getattr(
        args, "share_inter_axctc_norm", args.share_inter_xctc_norm
    )
    args.xctc_pae_ground_truth_ratio = getattr(args, "xctc_pae_ground_truth_ratio", 0)

    # mixup
    args.inter_mixup = getattr(args, "inter_mixup", False)
    args.inter_mixup_layer = getattr(args, "inter_mixup_layer", "-1")
    args.inter_mixup_decoder_layer = getattr(args, "inter_mixup_decoder_layer", "0")
    args.inter_mixup_beta = getattr(args, "inter_mixup_beta", 0.5)
    args.inter_mixup_prob = getattr(args, "inter_mixup_prob", 1)
    args.inter_mixup_ratio = getattr(args, "inter_mixup_ratio", 0.3)
    args.inter_mixup_keep_org = getattr(args, "inter_mixup_keep_org", False)
    args.inter_mixup_decoder_emb = getattr(args, "inter_mixup_decoder_emb", False)

    # compression
    args.compression_metric = getattr(args, "compression_metric", "ratio")
    args.compression_mode = getattr(args, "compression_mode", "create")
    args.compression_layers = getattr(args, "compression_layers", None)
    args.compression_threshold = getattr(args, "compression_threshold", "1.0")
    args.compression_ratio = getattr(args, "compression_ratio", "0.0")
    args.compression_norm = getattr(args, "compression_norm", False)
    args.compression_pos = getattr(args, "compression_pos", False)


@register_model_architecture("s2t_transformer", "s2t_transformer_s")
def s2t_transformer_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)


@register_model_architecture("s2t_transformer", "s2t_transformer_s_relative")
def s2t_transformer_s_relative(args):
    args.max_encoder_relative_length = 100
    args.max_decoder_relative_length = 20
    args.k_only = True
    s2t_transformer_s(args)


@register_model_architecture("s2t_transformer", "s2t_transformer_xs")
def s2t_transformer_xs(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 4)
    args.dropout = getattr(args, "dropout", 0.3)
    s2t_transformer_s(args)


@register_model_architecture("s2t_transformer", "s2t_transformer_sp")
def s2t_transformer_sp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_transformer_s(args)


@register_model_architecture("s2t_transformer", "s2t_transformer_m")
def s2t_transformer_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    base_architecture(args)


@register_model_architecture("s2t_transformer", "s2t_transformer_mp")
def s2t_transformer_mp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_transformer_m(args)


@register_model_architecture("s2t_transformer", "s2t_transformer_l")
def s2t_transformer_l(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.2)
    base_architecture(args)


@register_model_architecture("s2t_transformer", "s2t_transformer_lp")
def s2t_transformer_lp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_transformer_l(args)
