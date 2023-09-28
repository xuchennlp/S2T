import logging
import math
import copy
import os

import torch
import torch.nn as nn
from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models.transformer import Embedding
from fairseq.models import (
    FairseqEncoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text import (
    S2TTransformerModel,
    S2TTransformerEncoder,
    PDSS2TTransformerModel,
    PDSS2TTransformerEncoder,
)
from fairseq.modules.speech_to_text import Adapter, CTC
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
    TransformerS2EncoderLayer,
    S2TTransformerEncoderLayer,
    S2TTransformerS2EncoderLayer,
    DynamicLinearCombination,
)

logger = logging.getLogger(__name__)


@register_model("s2t_sate")
class S2TSATEModel(S2TTransformerModel):
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
        PDSS2TTransformerModel.add_specific_args(parser)
        S2TSATEModel.add_specific_args(parser)

    @staticmethod
    def add_specific_args(parser):
        # SATE setting
        parser.add_argument(
            "--acoustic-encoder",
            default="transformer",
            type=str,
            help="the architecture of the acoustic encoder",
        )
        parser.add_argument(
            "--text-encoder-layers",
            default=6,
            type=int,
            help="layers of the text encoder",
        )
        parser.add_argument(
            "--text-attention-type",
            default="selfattn",
            type=str,
            help="attention type of the textual encoder",
        )
        parser.add_argument(
            "--textual-encoder-embed-norm",
            action="store_true",
            help="use layer norm after down-sampling in the textual encoder",
        )
        parser.add_argument(
            "--textual-encoder-no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings in the textual encoder",
        )
        parser.add_argument(
            "--text-use-s2t-layer",
            action="store_true",
            help="if True, use s2t_transformer_layer in textual encoder",
        )
        parser.add_argument(
            "--text-no-pos-emb",
            action="store_true",
            help="if True, do not use position embedding in textual encoder",
        )
        # adapter
        parser.add_argument(
            "--adapter",
            default="league",
            type=str,
            help="adapter type",
        )
        parser.add_argument(
            "--ctc-shrink-strategy",
            default="avg",
            type=str,
            help="compress strategy of shrinking, such as avg, weighted, and softmax",
        )
        parser.add_argument(
            "--share-adapter-and-ctc",
            default=False,
            action="store_true",
            help="share the projection weights of the adapter and ctc",
        )
        parser.add_argument(
            "--share-adapter-and-embed",
            default=False,
            action="store_true",
            help="share the projection weights of the adapter and embed",
        )
        parser.add_argument(
            "--adapter-temperature",
            default=1.0,
            type=float,
            help="temperature of the CTC softmax in adapter",
        )
        parser.add_argument(
            "--adapter-embed-norm",
            default=False,
            action="store_true",
            help="use the layer norm for embed output",
        )
        parser.add_argument(
            "--adapter-out-norm",
            default=False,
            action="store_true",
            help="use the layer norm for final output",
        )
        parser.add_argument(
            "--adapter-gumbel", default=False, action="store_true", help="gumbel"
        )
        parser.add_argument(
            "--adapter-distribution-hard",
            default=False,
            action="store_true",
            help="hard distribution",
        )
        parser.add_argument(
            "--adapter-ground-truth-ratio",
            default=0,
            type=float,
            help="the ratio for ground truth in adapter",
        )

        # Pre-training
        parser.add_argument(
            "--load-pretrained-acoustic-encoder-from",
            type=str,
            metavar="STR",
            help="model to take acoustic encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--load-pretrained-text-encoder-from",
            type=str,
            metavar="STR",
            help="model to take text encoder weights from (for initialization)",
        )

        # Cross Attention for XCTC
        parser.add_argument(
            "--xctc-cross-attn",
            action="store_true",
            help="use the cross attention for target ctc",
        )
        parser.add_argument(
            "--cross-attn-start-layer",
            default=None,
            type=int,
            help="add cross attn for target ctc in the subsequent layers",
        )
        parser.add_argument(
            "--cross-attn-layer",
            default=None,
            type=int,
            help="the layer representation for cross attention",
        )
        parser.add_argument(
            "--cross-attn-ctc-logit",
            action="store_true",
            help="use the ctc logit for cross attention",
        )
        parser.add_argument(
            "--cross-attn-collaboration-mode",
            default="serial",
            type=str,
            help="how to calculate attention during league in encoder",
        )
        parser.add_argument(
            "--cross-attn-league-s1-ratio",
            default=0.5,
            type=float,
            help="league ratio of the s1 representation",
        )
        parser.add_argument(
            "--cross-attn-league-s2-ratio",
            default=0.5,
            type=float,
            help="league ratio of the s2 representation",
        )
        parser.add_argument(
            "--cross-attn-league-out-norm",
            action="store_true",
            help="layer normalization before league in the cross attention",
        )
        parser.add_argument(
            "--cross-attn-league-gated",
            action="store_true",
            help="league with the gated mechanism in the cross attention",
        )
        parser.add_argument(
            "--cross-attn-league-drop-net",
            action="store_true",
            help="drop one input during league",
        )
        parser.add_argument(
            "--cross-attn-league-drop-net-prob",
            default=0.0,
            type=float,
            help="probability of dropping one representations",
        )
        parser.add_argument(
            "--cross-attn-league-drop-net-mix",
            action="store_true",
            help="mix the two input with any probability",
        )
        # freeze
        parser.add_argument(
            "--freeze-acoustic-encoder",
            action="store_true",
            help="freeze the parameters of the acoustic encoder",
        )
        parser.add_argument(
            "--freeze-textual-encoder",
            action="store_true",
            help="freeze the parameters of the acoustic encoder",
        )
        parser.add_argument(
            "--freeze-decoder",
            action="store_true",
            help="freeze the parameters of the decoder",
        )

    @classmethod
    def build_encoder(cls, args, task=None, decoder_embed_tokens=None):
        encoder = S2TSATEEncoder(args, task, decoder_embed_tokens)

        if getattr(args, "load_pretrained_encoder_from", None):
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder,
                checkpoint=args.load_pretrained_encoder_from,
                strict=False,
            )

        if getattr(args, "load_pretrained_acoustic_encoder_from", None):
            logger.info(
                f"loaded pretrained acoustic encoder from: "
                f"{args.load_pretrained_acoustic_encoder_from}"
            )
            encoder.acoustic_encoder = (
                checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder.acoustic_encoder,
                    checkpoint=args.load_pretrained_acoustic_encoder_from,
                    strict=False,
                )
            )

        if getattr(args, "load_pretrained_text_encoder_from", None):
            logger.info(
                f"loaded pretrained text encoder from: "
                f"{args.load_pretrained_text_encoder_from}"
            )
            encoder.textual_encoder = (
                checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder.textual_encoder,
                    checkpoint=args.load_pretrained_text_encoder_from,
                    strict=False,
                )
            )
        if args.share_adapter_and_ctc and hasattr(encoder.adapter, "embed_adapter"):
            encoder.adapter.embed_adapter.weight = (
                encoder.acoustic_encoder.ctc.ctc_projection.weight
            )

        return encoder

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
            logging.info(
                "freeze the encoder module: {}".format(args.encoder_freeze_module)
            )

        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        if getattr(args, "decoder_freeze_module", None):
            utils.freeze_parameters(decoder, args.decoder_freeze_module)
            logging.info(
                "freeze the decoder module: {}".format(args.decoder_freeze_module)
            )

        # if args.share_adapter_and_embed and hasattr(encoder.adapter, "embed_adapter"):
        # encoder.adapter.embed_adapter.weight = decoder_embed_tokens.weight

        return cls(encoder, decoder)

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
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return decoder_out


class TextualEncoder(FairseqEncoder):
    def __init__(self, args, task, embed_tokens=None):

        super().__init__(None)

        self.register_buffer("version", torch.Tensor([3]))  # for consistent
        embed_dim = args.encoder_embed_dim
        layer_num = args.text_encoder_layers
        self.layer_num = layer_num
        self.embed_tokens = embed_tokens

        self.embed_scale = math.sqrt(embed_dim)
        if args.textual_encoder_no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = task.source_dictionary.pad_index
        self.text_no_pos_emb = getattr(args, "text_no_pos_emb", False)

        self.encoder_embed_norm = getattr(args, "textual_encoder_embed_norm", False)
        if self.encoder_embed_norm:
            self.embed_ln = LayerNorm(embed_dim)

        if not self.text_no_pos_emb:
            self.dropout_module = FairseqDropout(
                p=args.dropout, module_name=self.__class__.__name__
            )

            self.embed_positions = PositionalEmbedding(
                args.max_source_positions, args.encoder_embed_dim, self.padding_idx
            )

        self.use_s2t_layer = getattr(args, "text_use_s2t_layer", False)
        if self.use_s2t_layer:
            layer_module = S2TTransformerEncoderLayer
        else:
            layer_module = TransformerEncoderLayer

        self.layers = nn.ModuleList(
            [layer_module(args) for _ in range(layer_num)]
        )
        if args.encoder_normalize_before and layer_num > 0:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

        # XCTC
        self.use_xctc = getattr(args, "xctc_weight", 0) > 0
        if self.use_xctc:
            self.xctc_layer = getattr(args, "xctc_layer", layer_num)
            if self.xctc_layer == 0:
                self.xctc_layer = layer_num
            self.inter_xctc = True if self.xctc_layer != layer_num else False
            logger.info("XCTC loss in layer %d" % self.xctc_layer)
            self.xctc = CTC(
                embed_dim,
                dictionary_size=embed_tokens.num_embeddings
                if embed_tokens is not None
                else len(task.target_dictionary),
                dropout=args.dropout,
            )
            if self.inter_xctc:
                self.xctc_norm = LayerNorm(embed_dim)

            if (
                embed_tokens is not None
                and args.share_xctc_and_embed
                and self.xctc.ctc_projection.weight.size() == embed_tokens.weight.size()
            ):
                self.xctc.ctc_projection.weight = embed_tokens.weight

        self.inter_xctc_drop_prob = args.inter_ctc_drop_prob
        self.pae_ground_truth_ratio = getattr(args, "xctc_pae_ground_truth_ratio", 0)
        self.pae_unnorm_input = getattr(args, "pae_unnorm_input", False)
        self.pae_gt_decay = False
        decay_params = getattr(args, "xctc_pae_ground_truth_ratio_decay", None)
        self.gt_ratio = self.pae_ground_truth_ratio
        self.pae_adaptive_gt = getattr(args, "xctc_pae_ground_truth_ratio_adaptive", False)
        self.pae_gt_only_mistake = getattr(args, "xctc_pae_ground_truth_only_mistake", False)
        if self.pae_ground_truth_ratio != 0:
            if decay_params is not None and len(decay_params.split(":")) == 3:
                self.pae_gt_decay = True
                params = [float(item) for item in decay_params.split(":")]
                self.gt_decay_start_ratio = self.pae_ground_truth_ratio
                self.gt_decay_start_step, self.gt_decay_end_step, self.gt_decay_end_ratio = params
                self.gt_step_decay = (self.gt_decay_start_ratio - self.gt_decay_end_ratio) / (self.gt_decay_end_step - self.gt_decay_start_step)
                logger.info("PAE GT decay from step %d with ratio of %.2f end step %d with ratio of %.2f." % (
                    self.gt_decay_start_step, self.gt_decay_start_ratio, self.gt_decay_end_step, self.gt_decay_end_ratio
                ))
            else:
                logger.info("XCTC ground truth ratio: %.2f." % self.gt_ratio)

        self.inter_xctc_layers = []
        inter_xctc_layers = getattr(args, "inter_xctc_layers", None)
        if (
            getattr(args, "inter_xctc_weight", 0) > 0
            and inter_xctc_layers is not None
            and inter_xctc_layers != "none"
            and len(inter_xctc_layers.split(",")) > 0
        ):
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
                    xctc_norm = LayerNorm(embed_dim)
                    setattr(self, "xctc_norm%d" % layer_idx, xctc_norm)

                # consider layer norm
            if not hasattr(self, "xctc"):
                self.xctc = CTC(
                    embed_dim,
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

            strategy = {
                "embed_norm": getattr(args, "pae_embed_norm", False),
                "out_norm": getattr(args, "pae_out_norm", False),
                "ctc_shrink_strategy": getattr(args, "ctc_shrink_strategy", None),
                "ctc_temperature": getattr(args, "pae_ctc_temperature", 1.0),
                "distribution_cutoff": getattr(args, "pae_distribution_cutoff", None),
                "gumbel": getattr(args, "pae_gumbel", False),
                "distribution_hard": getattr(args, "pae_distribution_hard", None),
                "drop_prob": getattr(args, "pae_drop_prob", 0),
                "gt_ratio": self.pae_ground_truth_ratio,
                "oracle_smooth": getattr(args, "pae_oracle_smooth", False),
                "linear_init": getattr(args, "pae_linear_init", False)
            }

            self.xctc_pae = Adapter(
                embed_dim,
                args.xctc_pae,
                len(task.target_dictionary),
                strategy=strategy,
            )
            if args.share_pae_and_xctc and hasattr(self.xctc_pae, "embed_adapter"):
                self.xctc_pae.embed_adapter.weight = self.xctc.ctc_projection.weight

        # AXCTC
        self.use_axctc = getattr(args, "axctc_weight", 0) > 0
        if self.use_axctc:
            self.axctc_layer = getattr(args, "axctc_layer", None)
            if self.axctc_layer is None:
                logger.error("Need the layer of aligned XCTC.")

            assert hasattr(self, "xctc"), "No XCTC for AXCTC in textual encoder"
            logger.info("AXCTC loss in layer %d" % self.axctc_layer)
            self.axctc_norm = LayerNorm(embed_dim)

        self.inter_axctc_layers = []
        inter_axctc_layers = getattr(args, "inter_axctc_layers", None)
        if (
            getattr(args, "inter_axctc_weight", 0) > 0
            and inter_axctc_layers is not None
            and inter_axctc_layers != "none"
            and len(inter_axctc_layers.split(",")) > 0
        ):
            assert hasattr(self, "xctc_pae"), "No PAE for AXCTC in textual encoder"
            self.share_inter_axctc_norm = args.share_inter_axctc_norm
            if self.share_inter_axctc_norm:
                logger.info(
                    "Share layer norm in intermediate AXCTC %s." % inter_axctc_layers
                )
            else:
                logger.info(
                    "Do not Share layer norm in intermediate AXCTC %s."
                    % inter_axctc_layers
                )

            inter_axctc_layers = inter_axctc_layers.split(",")
            for layer_idx in inter_axctc_layers:
                layer_idx = int(layer_idx)
                assert layer_idx <= layer_num, (layer_idx, layer_num)

                if layer_idx <= 0:
                    layer_idx += layer_num
                self.inter_axctc_layers.append(layer_idx)

                if not self.share_inter_axctc_norm:
                    axctc_norm = LayerNorm(embed_dim)
                    setattr(self, "axctc_norm%d" % layer_idx, axctc_norm)

        self.use_cross_attn = False
        self.xctc_cross_attn = getattr(args, "xctc_cross_attn", False)
        if self.xctc_cross_attn:
            self.cross_attn_start_layer = getattr(args, "cross_attn_start_layer", None)
            self.cross_attn_layer = getattr(args, "cross_attn_layer", None)
            self.cross_attn_ctc_logit = getattr(args, "cross_attn_ctc_logit", False)
            if (
                self.cross_attn_start_layer is not None
                and self.cross_attn_layer is not None
            ):
                del self.layers
                self.use_cross_attn = True
                self.attn_norm = LayerNorm(embed_dim)

                update_args = copy.deepcopy(args)
                setattr(
                    update_args,
                    "encoder_collaboration_mode",
                    args.cross_attn_collaboration_mode,
                )
                setattr(
                    update_args,
                    "encoder_league_s1_ratio",
                    args.cross_attn_league_s1_ratio,
                )
                setattr(
                    update_args,
                    "encoder_league_s2_ratio",
                    args.cross_attn_league_s2_ratio,
                )
                setattr(
                    update_args,
                    "encoder_league_drop_net",
                    args.cross_attn_league_drop_net,
                )
                setattr(
                    update_args,
                    "encoder_league_drop_net_prob",
                    args.cross_attn_league_drop_net_prob,
                )
                setattr(
                    update_args,
                    "encoder_league_drop_net_mix",
                    args.cross_attn_league_drop_net_mix,
                )
                setattr(
                    update_args,
                    "encoder_league_out_norm",
                    args.cross_attn_league_out_norm,
                )
                setattr(
                    update_args, "encoder_league_gated", args.cross_attn_league_gated
                )

                if self.use_s2t_layer:
                    s2_layer_module = S2TTransformerS2EncoderLayer
                else:
                    s2_layer_module = TransformerS2EncoderLayer 
                
                layers = [
                    layer_module(args)
                    for _ in range(self.cross_attn_start_layer - 1)
                ]
                layers.extend(
                    [
                        s2_layer_module(update_args)
                        for _ in range(layer_num - self.cross_attn_start_layer + 1)
                    ]
                )
                self.layers = nn.ModuleList(layers)

    def dump(self, fstream, info=""):
        for i, layer in enumerate(self.layers):
            layer.dump(fstream, "%s Layer %d" % (info, i))

    def set_flag(self, **kwargs):
        for layer in self.layers:
            if hasattr(layer, "set_flag"):
                layer.set_flag(**kwargs)

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.update_num = num_updates
        if self.pae_gt_decay:
            if self.gt_decay_start_step < self.update_num < self.gt_decay_end_step:
                self.gt_ratio = self.gt_decay_start_ratio - self.gt_step_decay * (self.update_num - self.gt_decay_start_step)

    def forward(self, x, encoder_padding_mask=None, history=None, **kwargs):

        if self.encoder_embed_norm:
            x = self.embed_ln(x)
        x = self.embed_scale * x

        if not self.text_no_pos_emb:
            positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
            x = positions + x
            x = self.dropout_module(x)
        else:
            positions = None

        xctc_logit = None
        inter_xctc_logits = []
        axctc_logit = None
        inter_axctc_logits = []
        layer_idx = 0
        attn_x = None

        # CTC alignment
        xctc_oracle = None
        xctc_oracle_mask = None
        xctc_force_emit = None
        axctc_oracle = None
        axctc_oracle_mask = None
        axctc_force_emit = None
        for layer in self.layers:
            if history is not None:
                x = history.pop()

            if self.use_cross_attn and layer_idx >= self.cross_attn_start_layer - 1:
                x = layer(
                    x,
                    encoder_padding_mask,
                    s2=attn_x,
                    s2_encoder_padding_mask=encoder_padding_mask,
                    pos_emb=positions,
                )
            else:
                x = layer(x, encoder_padding_mask, pos_emb=positions)
            layer_idx += 1

            if self.use_cross_attn and layer_idx == self.cross_attn_layer:
                attn_x = self.attn_norm(x)

            if self.use_axctc and self.axctc_layer == layer_idx:
                axctc_logit = self.xctc(
                    x.clone(), encoder_padding_mask, "AXCTC layer %d" % layer_idx
                )

            if self.use_xctc and self.inter_xctc and self.xctc_layer == layer_idx:
                norm_x = self.xctc_norm(x)
                xctc_logit = self.xctc(
                    x, encoder_padding_mask, "XCTC layer %d" % layer_idx, is_top=True
                )

            if layer_idx in self.inter_axctc_layers:
                if self.inter_xctc_drop_prob > 0:
                    p = torch.rand(1).uniform_()
                    if p < self.inter_xctc_drop_prob:
                        break

                if self.share_inter_axctc_norm:
                    norm_x = self.layer_norm(x)
                else:
                    norm = getattr(self, "axctc_norm%d" % layer_idx)
                    norm_x = norm(x)

                if self.use_cross_attn and self.cross_attn_ctc_logit:
                    attn_x = norm_x

                logit = self.xctc(
                    norm_x, encoder_padding_mask, "Inter AXCTC layer %d" % layer_idx
                )

                inter_logit = logit
                if self.gt_ratio > 0:
                    ctc_alignment_oracle = kwargs.get("ctc_alignment_oracle", None)
                    if (
                        ctc_alignment_oracle is not None
                        and ctc_alignment_oracle.get("axctc", None) is not None
                    ):
                        if axctc_oracle is None:
                            axctc_oracle, best_aligns_pad = ctc_alignment_oracle[
                                "axctc"
                            ]
                            axctc_oracle_mask = (
                                torch.rand(
                                    axctc_oracle.size(), device=axctc_oracle.device
                                )
                                < self.gt_ratio
                            ).bool()
                            axctc_force_emit = best_aligns_pad.masked_fill(
                                ~axctc_oracle_mask, -1
                            )
                        else:
                            inter_logit = [logit, None, axctc_force_emit]

                pae_input = x if self.pae_unnorm_input else norm_x
                if self.xctc_pae.adapter_type != "none":
                    x, encoder_padding_mask = self.xctc_pae(
                        [pae_input, logit],
                        encoder_padding_mask,
                        axctc_oracle,
                        axctc_oracle_mask,
                    )

                inter_axctc_logits.append(inter_logit)

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

                if self.use_cross_attn and self.cross_attn_ctc_logit:
                    attn_x = norm_x

                logit = self.xctc(
                    norm_x, encoder_padding_mask, "Inter XCTC layer %d" % layer_idx
                )

                inter_logit = logit
                # CTC alignment
                if self.gt_ratio > 0:
                    ctc_alignment_oracle = kwargs.get("ctc_alignment_oracle", None)
                    if (
                        ctc_alignment_oracle is not None
                        and ctc_alignment_oracle.get("xctc", None) is not None
                    ):
                        if xctc_oracle is None:
                            xctc_oracle, best_aligns_pad, mistake_flag, mistake_ratio = ctc_alignment_oracle["xctc"]
                            if self.pae_adaptive_gt:
                                prob = self.gt_ratio * mistake_ratio.unsqueeze(-1)
                            else:
                                prob = self.gt_ratio
                            xctc_oracle_mask = (
                                torch.rand(
                                    xctc_oracle.size(), device=xctc_oracle.device
                                )
                                < prob
                            ).bool()
                            if self.pae_gt_only_mistake:
                                xctc_oracle_mask.masked_fill_(
                                    ~mistake_flag, False
                                )
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

            if history is not None:
                history.push(x)

        if history is not None:
            x = history.pop()

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if self.use_xctc and xctc_logit is None:
            xctc_logit = self.xctc(
                x, encoder_padding_mask, "Textual encoder output", is_top=True
            )
            
        if xctc_force_emit is not None:
            xctc_logit = [xctc_logit, None, xctc_force_emit]

        return x, xctc_logit, inter_xctc_logits, axctc_logit, inter_axctc_logits

    def reorder_encoder_out(self, encoder_out, new_order):
        pass


class S2TSATEEncoder(FairseqEncoder):
    """Speech-to-text Conformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args, task=None, decoder_embed_tokens=None):
        super().__init__(None)

        # acoustic encoder
        acoustic_encoder_type = args.acoustic_encoder
        if args.text_encoder_layers > 0:
            setattr(args, "disable_xctc", True)
        if acoustic_encoder_type == "transformer":
            self.acoustic_encoder = S2TTransformerEncoder(
                args, task, decoder_embed_tokens
            )
        elif acoustic_encoder_type == "pds":
            self.acoustic_encoder = PDSS2TTransformerEncoder(
                args, task, decoder_embed_tokens
            )
        else:
            logging.error("Unsupported model arch {}!".format(acoustic_encoder_type))

        self.freeze_acoustic_encoder = getattr(args, "freeze_acoustic_encoder", False)
        self.freeze_textual_encoder = getattr(args, "freeze_textual_encoder", False)
        self.adapter_ground_truth_ratio = getattr(args, "adapter_ground_truth_ratio", 0)
        self.pae_ground_truth_ratio = (
            getattr(args, "ctc_pae_ground_truth_ratio", 0)
            + getattr(args, "adapter_ground_truth_ratio", 0)
            + getattr(args, "xctc_pae_ground_truth_ratio", 0)
        )
        if self.adapter_ground_truth_ratio != 0:
            logger.info(
                "Adapter ground truth ratio: %.2f." % self.adapter_ground_truth_ratio
            )

        # adapter
        strategy = {
            "embed_norm": getattr(args, "adapter_embed_norm", False),
            "out_norm": getattr(args, "adapter_out_norm", False),
            "shrink_strategy": getattr(args, "adapter_shrink_strategy", None),
            "ctc_temperature": getattr(args, "adapter_temperature", 1.0),
            "gumbel": getattr(args, "adapter_gumbel", False),
            "distribution_hard": getattr(args, "adapter_distribution_hard", None),
            "drop_prob": getattr(args, "adapter_drop_prob", 0),
            "gt_ratio": self.adapter_ground_truth_ratio,
        }

        self.adapter = Adapter(
            args.encoder_embed_dim,
            args.adapter,
            len(task.source_dictionary),
            strategy=strategy,
        )

        assert not (
            args.share_adapter_and_ctc and args.share_adapter_and_embed
        ), "Can not be True at the same time"
        if args.share_adapter_and_ctc and hasattr(self.adapter, "embed_adapter"):
            self.adapter.embed_adapter.weight = (
                self.acoustic_encoder.ctc.ctc_projection.weight
            )
        if (
            args.share_adapter_and_embed
            and hasattr(self.adapter, "embed_adapter")
            and task.source_dictionary == task.target_dictionary
        ):
            self.adapter.embed_adapter.weight = decoder_embed_tokens.weight

        acoustic_encoder_attention_type = args.encoder_attention_type
        args.encoder_attention_type = args.text_attention_type
        # textual encoder
        self.textual_encoder = TextualEncoder(args, task, decoder_embed_tokens)

        args.encoder_attention_type = acoustic_encoder_attention_type

        if getattr(args, "use_enc_dlcl", False):
            layer_num = args.encoder_layers + args.text_encoder_layers + 2
            self.history = DynamicLinearCombination(
                args, is_encoder=True, layer_num=layer_num
            )
        else:
            self.history = None

    def dump(self, fstream, info=""):
        self.acoustic_encoder.dump(fstream, "%s Acoustic Encoder" % info) if hasattr(
            self.acoustic_encoder, "dump"
        ) else None
        self.textual_encoder.dump(fstream, "%s Textual Encoder" % info) if hasattr(
            self.textual_encoder, "dump"
        ) else None

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.update_num = num_updates

        self.acoustic_encoder.set_num_updates(num_updates) if hasattr(self.acoustic_encoder, "set_num_updates") else None
        self.textual_encoder.set_num_updates(num_updates) if hasattr(self.textual_encoder, "set_num_updates") else None

    def set_flag(self, **kwargs):
        self.acoustic_encoder.set_flag(**kwargs) if hasattr(self.acoustic_encoder, "set_flag") else None
        self.textual_encoder.set_flag(**kwargs) if hasattr(self.textual_encoder, "set_flag") else None

    def set_ctc_infer(
        self, ctc_infer, post_process, src_dict=None, tgt_dict=None, path=None
    ):
        if hasattr(self.acoustic_encoder, "ctc"):
            assert src_dict is not None
            logger.info("Acoustic Encoder CTC Inference")
            self.acoustic_encoder.ctc.set_infer(
                ctc_infer,
                post_process,
                src_dict,
                path=os.path.splitext(path)[0] + ".ctc" if path is not None else None,
            )
        if hasattr(self.textual_encoder, "xctc"):
            assert tgt_dict is not None
            logger.info("Textual Encoder CTC Inference")
            self.textual_encoder.xctc.set_infer(
                ctc_infer,
                post_process,
                tgt_dict,
                path=os.path.splitext(path)[0] + ".xctc" if path is not None else None,
            )

    def ctc_valid(self, lprobs, targets, input_lengths, dictionary, lang="source"):
        if lang == "source":
            if hasattr(self.acoustic_encoder, "ctc"):
                return self.acoustic_encoder.ctc.valid(
                    lprobs, targets, input_lengths, dictionary
                )
            else:
                logger.error("No ctc module in textual encoder")
        else:
            if hasattr(self.textual_encoder, "xctc"):
                return self.textual_encoder.xctc.valid(
                    lprobs, targets, input_lengths, dictionary
                )
            else:
                logger.error("No xctc module in textual encoder")

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        if self.history is not None:
            self.history.clean()

        ctc_logit = None
        inter_ctc_logits = []
        mixup = None

        if (isinstance(self.acoustic_encoder.layers, int) and self.acoustic_encoder.layers == 0) \
            or (isinstance(self.acoustic_encoder.layers, list) and len(self.acoustic_encoder.layers) == 0):
            x = src_tokens.transpose(0, 1)
            x, input_lengths = self.acoustic_encoder.subsample(x, src_lengths)
            encoder_padding_mask = lengths_to_padding_mask(input_lengths)
            ctc_padding_mask = encoder_padding_mask
        else:
            if self.freeze_acoustic_encoder:
                with torch.no_grad():
                    acoustic_encoder_out = self.acoustic_encoder(
                        src_tokens, src_lengths, **kwargs
                    )
            else:
                acoustic_encoder_out = self.acoustic_encoder(
                    src_tokens, src_lengths, **kwargs
                )

            encoder_out = acoustic_encoder_out["encoder_out"][0]
            encoder_padding_mask = acoustic_encoder_out["encoder_padding_mask"][0]
            ctc_padding_mask = encoder_padding_mask
            if "mixup" in acoustic_encoder_out:
                mixup = acoustic_encoder_out["mixup"]

            inter_ctc_logits = acoustic_encoder_out.get("inter_ctc_logits", [])
            xctc_logits = acoustic_encoder_out.get("xctc_logits", [])
            inter_xctc_logits = acoustic_encoder_out.get("inter_xctc_logits", [])

            if (
                "ctc_logit" in acoustic_encoder_out
                and len(acoustic_encoder_out["ctc_logit"]) > 0
            ):
                ctc_logit = acoustic_encoder_out["ctc_logit"][0]
                if type(ctc_logit) == list and len(ctc_logit) > 1:
                    logit = ctc_logit[0]
                else:
                    logit = ctc_logit
            else:
                logit = None

            x = (encoder_out, logit)
            x, encoder_padding_mask = self.adapter(x, encoder_padding_mask)

        if self.history is not None:
            acoustic_history = self.acoustic_encoder.history
            layer_num = acoustic_history.layer_num
            idx = (
                torch.arange(layer_num)
                .unsqueeze(0)
                .T.repeat(1, layer_num)
                .to(x.device)
                .unsqueeze(2)
            )
            self.history.weight.scatter(0, idx, acoustic_history.weight)
            self.history.layers.extend(acoustic_history.layers)
            self.history.count = acoustic_history.count

            self.history.push(x)

        if self.freeze_textual_encoder:
            with torch.no_grad():
                (
                    x,
                    xctc_logit,
                    inter_xctc_logits,
                    axctc_logit,
                    inter_axctc_logits,
                ) = self.textual_encoder(
                    x, encoder_padding_mask, self.history, **kwargs
                )
        else:
            (
                x,
                xctc_logit,
                inter_xctc_logits,
                axctc_logit,
                inter_axctc_logits,
            ) = self.textual_encoder(x, encoder_padding_mask, self.history, **kwargs)

        
        return {
            "encoder_out": [x],  # T x B x C
            "ctc_logit": [] if ctc_logit is None else [ctc_logit],  # T x B x C
            "inter_ctc_logits": inter_ctc_logits,  # B x T x C
            "xctc_logit": xctc_logits if xctc_logit is None else [xctc_logit],  # B x T x C
            "inter_xctc_logits": inter_xctc_logits,  # B x T x C
            "axctc_logit": [] if axctc_logit is None else [axctc_logit],  # B x T x C
            "inter_axctc_logits": inter_axctc_logits,  # B x T x C
            "ctc_padding_mask": [ctc_padding_mask],  # B x T
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "mixup": mixup,
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        new_ctc_logit = (
            []
            if len(encoder_out["ctc_logit"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["ctc_logit"]]
        )
        new_xctc_logit = (
            []
            if len(encoder_out["xctc_logit"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["xctc_logit"]]
        )
        new_encoder_out = (
            []
            if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
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
            "ctc_logit": new_ctc_logit,
            "xctc_logit": new_xctc_logit,
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }


@register_model_architecture(model_name="s2t_sate", arch_name="s2t_sate")
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

    # intermediate CTC
    args.inter_ctc_layers = getattr(args, "inter_ctc_layers", None)
    args.share_inter_ctc_norm = getattr(args, "share_inter_ctc_norm", False)
    args.pae_ctc_temperature = getattr(args, "pae_ctc_temperature", 1)
    args.inter_ctc_drop_prob = getattr(args, "inter_ctc_drop_prob", 0)

    # Prediction-aware Encoding (pae)
    args.ctc_pae = getattr(args, "ctc_pae", "none")
    args.share_pae_and_ctc = getattr(args, "share_pae_and_ctc", False)
    args.pae_drop_prob = getattr(args, "pae_drop_prob", 0)
    args.pae_distribution_cutoff = getattr(args, "pae_distribution_cutoff", None)
    args.pae_distribution_hard = getattr(args, "pae_distribution_hard", False)
    args.pae_gumbel = getattr(args, "pae_gumbel", False)
    args.pae_linear_init = getattr(args, "pae_linear_init", False)
    args.pae_unnorm_input = getattr(args, "pae_unnorm_input", False)

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

    # PDS
    args.pds_stages = getattr(args, "pds_stages", None)
    args.pds_layers = getattr(args, "pds_layers", None)
    args.pds_ratios = getattr(args, "pds_ratios", None)

    args.pds_ds_method = getattr(args, "pds_ds_method", "conv")
    args.pds_embed_dims = getattr(args, "pds_embed_dims", None)
    args.pds_embed_norm = getattr(args, "pds_embed_norm", False)
    args.pds_position_embed = getattr(args, "pds_position_embed", None)

    args.pds_attn_heads = getattr(args, "pds_attn_heads", None)
    args.pds_ffn_ratios = getattr(args, "pds_ffn_ratios", None)
    args.pds_cnn_kernel_sizes = getattr(args, "pds_cnn_kernel_sizes", None)

    args.pds_attn_ds_ratios = getattr(args, "pds_attn_ds_ratios", None)
    args.pds_conv_strides = getattr(args, "pds_conv_strides", None)
    args.pds_attn_strides = getattr(args, "pds_attn_strides", None)

    args.ctc_layer = getattr(args, "ctc_layer", 0)
    args.pds_dropout = getattr(args, "pds_dropout", 0)

    args.pds_fusion = getattr(args, "pds_fusion", False)
    args.pds_fusion_method = getattr(args, "pds_fusion_method", "all_conv")

    # SATE
    args.acoustic_encoder = getattr(args, "acoustic_encoder", "transformer")
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 6)
    args.text_attention_type = getattr(args, "text_attention_type", "selfattn")
    args.textual_encoder_embed_norm = getattr(args, "textual_encoder_embed_norm", False)
    args.textual_encoder_no_scale_embedding = getattr(
        args, "textual_encoder_no_scale_embedding", False
    )
    args.text_no_pos_emb = getattr(args, "text_no_pos_emb", False)
    args.text_use_s2t_layer = getattr(args, "text_use_s2t_layer", False)
    args.freeze_acoustic_encoder = getattr(args, "freeze_acoustic_encoder", False)
    args.freeze_textual_encoder = getattr(args, "freeze_textual_encoder", False)

    # Adapter
    args.adapter = getattr(args, "adapter", "inter_league")
    args.ctc_shrink_strategy = getattr(args, "ctc_shrink_strategy", "avg")
    args.adapter_temperature = getattr(args, "adapter_temperature", 1.0)
    args.adapter_distribution_hard = getattr(args, "adapter_distribution_hard", False)
    args.share_adapter_and_ctc = getattr(args, "share_adapter_and_ctc", False)
    args.share_adapter_and_embed = getattr(args, "share_adapter_and_embed", False)
    args.adapter_embed_norm = getattr(args, "adapter_embed_norm", False)
    args.adapter_out_norm = getattr(args, "adapter_out_norm", False)
    args.adapter_gumbel = getattr(args, "adapter_gumbel", False)
    args.adapter_out_norm = getattr(args, "adapter_out_norm", False)
    args.ctc_pae_ground_truth_ratio = getattr(args, "ctc_pae_ground_truth_ratio", 0)
    args.adapter_ground_truth_ratio = getattr(args, "adapter_ground_truth_ratio", 0)

    # XCTC
    args.share_xctc_and_embed = getattr(args, "share_xctc_and_embed", False)
    args.xctc_pae = getattr(args, "xctc_pae", args.ctc_pae)
    args.axctc_pae = getattr(args, "axctc_pae", args.xctc_pae)
    args.share_pae_and_xctc = getattr(args, "share_pae_and_xctc", False)
    args.xctc_layer = getattr(args, "xctc_layer", 0)
    args.inter_xctc_layers = getattr(args, "inter_xctc_layers", None)
    args.axctc_layer = getattr(args, "axctc_layer", None)
    args.inter_axctc_layers = getattr(args, "inter_axctc_layers", None)
    args.share_inter_xctc_norm = getattr(args, "share_inter_xctc_norm", False)
    args.share_inter_axctc_norm = getattr(
        args, "share_inter_axctc_norm", args.share_inter_xctc_norm
    )
    args.xctc_pae_ground_truth_ratio = getattr(args, "xctc_pae_ground_truth_ratio", 0)

    # XCTC cross attn
    args.xctc_cross_attn = getattr(args, "xctc_cross_attn", False)
    args.cross_attn_start_layer = getattr(args, "cross_attn_start_layer", 4)
    args.cross_attn_layer = getattr(args, "cross_attn_layer", 3)
    args.cross_attn_ctc_logit = getattr(args, "cross_attn_ctc_logit", False)
    args.cross_attn_collaboration_mode = getattr(
        args, "cross_attn_collaboration_mode", "none"
    )
    args.cross_attn_league_s1_ratio = getattr(args, "cross_attn_league_s1_ratio", 0.5)
    args.cross_attn_league_s2_ratio = getattr(args, "cross_attn_league_s2_ratio", 0.5)
    args.cross_attn_league_out_norm = getattr(args, "cross_attn_league_out_norm", False)
    args.cross_attn_league_gated = getattr(args, "cross_attn_league_gated", False)
    args.cross_attn_league_drop_net = getattr(args, "cross_attn_league_drop_net", False)
    args.cross_attn_league_drop_net_prob = getattr(
        args, "cross_attn_league_drop_net_prob", 0.0
    )
    args.cross_attn_league_drop_net_mix = getattr(
        args, "cross_attn_league_drop_net_mix", False
    )


@register_model_architecture("s2t_sate", "s2t_sate_s")
def s2t_sate_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)


@register_model_architecture("s2t_sate", "s2t_sate_s_relative")
def s2t_sate_s_relative(args):
    args.encoder_attention_type = "relative"
    args.decoder_attention_type = "relative"
    args.max_encoder_relative_length = 100
    args.max_decoder_relative_length = 20
    args.k_only = True
    s2t_sate_s(args)


@register_model_architecture("s2t_sate", "s2t_sate_xs")
def s2t_sate_xs(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 3)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 4)
    args.dropout = getattr(args, "dropout", 0.3)
    s2t_sate_s(args)


@register_model_architecture("s2t_sate", "s2t_sate_m")
def s2t_sate_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    base_architecture(args)


@register_model_architecture("s2t_sate", "s2t_sate_l")
def s2t_sate_l(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.2)
    base_architecture(args)
