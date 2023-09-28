import logging
import math
from functools import reduce
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils, utils
from fairseq.models import (
    FairseqEncoder,
    register_model,
    register_model_architecture,
)
from .s2t_transformer import S2TTransformerModel
from fairseq.modules.speech_to_text import CTC, Adapter

from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    RelPositionalEncoding,
    LegacyRelPositionalEncoding,
    PDSTransformerEncoderLayer,
    DownSampleConvolutionModule,
    ConvolutionModule,
)
from fairseq.modules.speech_to_text import subsampling

logger = logging.getLogger(__name__)


def lengths_to_padding_mask_with_maxlen(lens, max_length):
    bsz = lens.size(0)
    mask = torch.arange(max_length).to(lens.device).view(1, max_length)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_length)
    return mask


class Permute120(nn.Module):
    @staticmethod
    def forward(x):
        return x.permute(1, 2, 0)


class Permute201(nn.Module):
    @staticmethod
    def forward(x):
        return x.permute(2, 0, 1)


class Downsampling(nn.Module):
    # down-sampling module
    def __init__(
        self,
        reduced_way: str,
        embed_norm: bool,
        in_channels: int,
        out_channels: int,
        kernel_sizes: int,
        stride: int,
        padding: int,
    ):
        super().__init__()

        self.stride = stride
        self.reduced_way = reduced_way

        if stride == 0:
            return
        # default conv
        if self.reduced_way == "conv":
            self.conv = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_sizes,
                    stride=stride,
                    padding=padding,
                ),
            )
        elif self.reduced_way == "proj":
            self.conv = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_sizes,
                    stride=stride,
                    padding=padding,
                ),
                nn.ReLU(),
            )
        else:
            logger.error("Unsupported reduced way!")

        self.embed_norm = embed_norm
        if self.embed_norm:
            self.norm = LayerNorm(out_channels)

    def forward(self, x, lengths):
        if self.stride == 0:
            return x, lengths

        seq_len, bsz, dim = x.size()
        assert (
            seq_len % self.stride == 0
        ), "The sequence length %d must be a multiple of %d." % (seq_len, self.stride)

        # mask batch padding
        if not torch.all(lengths == seq_len):
            padding_mask = lengths_to_padding_mask_with_maxlen(
                lengths, seq_len
            )  # bsz, seq_len
            mask_pad = padding_mask.unsqueeze(2)
            if mask_pad is not None:
                x = x.transpose(0, 1)
                x = x.masked_fill(mask_pad, 0.0)
                x = x.transpose(0, 1)

        lengths = ((lengths.float() - 1) / self.stride + 1).floor().long()
        out_seq_len = max(lengths).item()
        if self.reduced_way == "proj":
            x = x.permute(1, 2, 0)  # bsz, dim, seq_len
            x = nn.functional.adaptive_avg_pool1d(x, out_seq_len)
            x = self.conv(self.in_norm(x))
            x = x.permute(2, 0, 1)  # seq_len, bsz, dim
        else:
            x = x.permute(1, 2, 0)  # B * D * T
            x = self.conv(x)
            x = x.permute(2, 0, 1)  # T * B * D
        if self.embed_norm:
            x = self.norm(x)

        # mask batch padding
        if not torch.all(lengths == x.size(-1)):
            padding_mask = lengths_to_padding_mask_with_maxlen(lengths, x.size(0))
            mask_pad = padding_mask.unsqueeze(2)
            if mask_pad is not None:
                x = x.transpose(0, 1)
                x = x.masked_fill(mask_pad, 0.0)
                x = x.transpose(0, 1)

        return x, lengths


@register_model("pdss2t_transformer")
class PDSS2TTransformerModel(S2TTransformerModel):
    """Progressive down-sampling for acoustic encoding."""

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        S2TTransformerModel.add_args(parser)
        PDSS2TTransformerModel.add_specific_args(parser)

    @staticmethod
    def add_specific_args(parser):
        """Add specific arguments to the parser."""
        # PDS setting
        parser.add_argument(
            "--pds-stages",
            type=int,
            help="the number of the stage",
        )
        parser.add_argument(
            "--pds-layers",
            type=str,
            help="the number of the encoder layers in each stage",
        )
        parser.add_argument(
            "--pds-ratios",
            type=str,
            help="the ratio of the down-sampling in each stage",
        )
        parser.add_argument(
            "--pds-ds-method",
            type=str,
            choices=["glu", "conv", "proj", "fusion"],
            help="the down-sampling method",
        )
        parser.add_argument(
            "--pds-embed-dims",
            type=str,
            help="the embedding dimension in each stage",
        )
        parser.add_argument(
            "--pds-kernel-sizes",
            type=str,
            help="the kernel size of the down-sampling module in each stage",
        )
        parser.add_argument(
            "--pds-embed-norm",
            action="store_true",
            help="use layer norm in the down-sampling module",
        )
        parser.add_argument(
            "--pds-position-embed",
            type=str,
            help="use the position embedding or not before each encoding",
        )
        parser.add_argument(
            "--pds-attn-heads",
            type=str,
            help="the number of the attention heads in each stage",
        )
        parser.add_argument(
            "--pds-attn-ds-ratios",
            type=str,
            help="the ratios of the down-sampling in the self attention module",
        )
        parser.add_argument(
            "--pds-ffn-ratios",
            type=str,
            help="the ratio of the ffn in each stage",
        )
        parser.add_argument(
            "--pds-cnn-kernel-sizes",
            type=str,
            help="the kernel size of convolutional modules in Conformer",
        )
        parser.add_argument(
            "--pds-conv-strides",
            type=str,
            help="the strides of the convolutional module (conformer) in each stage",
        )
        parser.add_argument(
            "--pds-attn-strides",
            type=str,
            help="the strides of the attention module (conformer) in each stage",
        )
        parser.add_argument(
            "--pds-fusion",
            action="store_true",
            help="use the representation fusion method",
        )
        parser.add_argument(
            "--pds-fusion-layers",
            type=str,
            help="use fusion in each stage",
        )
        parser.add_argument(
            "--pds-fusion-mask",
            action="store_true",
            help="use the representation fusion method",
        )
        parser.add_argument(
            "--pds-fusion-no-prenorm",
            action="store_true",
            help="use the representation fusion method",
        )
        parser.add_argument(
            "--pds-fusion-method",
            type=str,
            help="the fusion method",
        )
        parser.add_argument(
            "--pds-fusion-weight",
            type=str,
            help="the specify fusion weight",
        )
        parser.add_argument(
            "--pds-dropout",
            type=float,
            help="dropout in each stage",
        )
        parser.add_argument(
            "--pds-ctc",
            type=str,
            help="use the ctc after each stage",
        )
        parser.add_argument(
            "--pds-xctc",
            type=str,
            help="use the xctc after each stage",
        )        
        parser.add_argument(
            "--pds-final-layers",
            default=0,
            type=int,
            help="encoding with n layers after fusion",
        )

    @classmethod
    def build_encoder(cls, args, task=None, embed_tokens=None):
        encoder = PDSS2TTransformerEncoder(args, task, embed_tokens)
        if getattr(args, "load_pretrained_encoder_from", None):
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


class PDSS2TTransformerEncoder(FairseqEncoder):
    """Progressive Down-sampling for Acoustic Encoding"""

    def __init__(self, args, task=None, embed_tokens=None):
        super().__init__(None)

        self.padding_idx = 1
        self.attn_type = getattr(args, "encoder_attention_type", "selfattn")
        self.embed_dim = args.encoder_embed_dim

        self.dropout = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.pds_dropout = FairseqDropout(
            p=getattr(args, "pds_dropout", args.dropout),
            module_name=self.__class__.__name__,
        )

        self.pds_stages = getattr(args, "pds_stages", 4)
        self.pds_layers = [int(n) for n in args.pds_layers.split("_")]
        self.layers = sum(self.pds_layers)
        self.pds_ratios = [int(n) for n in args.pds_ratios.split("_")]

        # down-sampling module
        self.pds_ds_method = args.pds_ds_method
        self.pds_embed_dims = [int(n) for n in args.pds_embed_dims.split("_")]
        self.pds_kernel_sizes = [int(n) for n in args.pds_kernel_sizes.split("_")]
        self.pds_embed_norm = args.pds_embed_norm
        self.pds_position_embed = [int(n) for n in args.pds_position_embed.split("_")]
        self.pds_attn_heads = [int(n) for n in args.pds_attn_heads.split("_")]
        self.pds_ffn_ratios = [int(n) for n in args.pds_ffn_ratios.split("_")]
        self.pds_cnn_kernel_sizes = (
            [int(n) for n in args.pds_cnn_kernel_sizes.split("_")]
            if getattr(args, "pds_cnn_kernel_sizes", None) is not None
            else None
        )
        self.pds_attn_ds_ratios = (
            [int(n) for n in args.pds_attn_ds_ratios.split("_")]
            if args.pds_attn_ds_ratios is not None
            else None
        )

        self.pds_conv_strides = (
            [int(n) for n in args.pds_conv_strides.split("_")]
            if args.pds_conv_strides is not None
            else None
        )
        self.pds_attn_strides = (
            [int(n) for n in args.pds_attn_strides.split("_")]
            if args.pds_attn_strides is not None
            else None
        )

        # fusion
        self.pds_fusion = args.pds_fusion
        pds_fusion_layers = getattr(args, "pds_fusion_layers", None)
        self.pds_fusion_layers = (
            [int(n) for n in pds_fusion_layers.split("_")]
            if pds_fusion_layers is not None
            else [1] * self.pds_stages
        )
        self.fusion_mask = getattr(args, "pds_fusion_mask", False)
        self.fusion_no_prenorm = getattr(args, "pds_fusion_no_prenorm", False)
        self.pds_fusion_method = args.pds_fusion_method
        if self.pds_fusion_method is None or self.pds_fusion_method == "none":
            self.pds_fusion = False

        pds_fusion_weight = getattr(args, "pds_fusion_weight", None)

        self.pds_fusion_transform = "conv"
        if len(self.pds_fusion_method.split("_")) == 2:
            items = self.pds_fusion_method.split("_")
            self.pds_fusion_method = items[0]
            self.pds_fusion_transform = items[1]

        fusion_stages_num = 0
        if self.pds_fusion:
            if self.pds_fusion_method == "all":
                fusion_stages_num = self.pds_stages
            elif self.pds_fusion_method == "same":
                for dim in self.pds_embed_dims:
                    if dim == self.embed_dim:
                        fusion_stages_num += 1
            else:
                logger.error("Unsupported fusion!")
            if fusion_stages_num == 1:
                fusion_stages_num = 0
        self.fusion_stages_num = min(fusion_stages_num, sum(self.pds_fusion_layers))

        # PDS CTC
        args.pds_ctc = getattr(args, "pds_ctc", None)
        self.pds_ctc = (
            [int(n) for n in args.pds_ctc.split("_")]
            if args.pds_ctc is not None
            else None
        )
        self.share_inter_ctc = getattr(args, "share_inter_ctc", False)
                
        # Multi-level objective (MLO)
        inter_ctc_mlo = getattr(args, "inter_ctc_mlo", "")
        if inter_ctc_mlo != "":
            inter_ctc_mlo = [int(x) for x in inter_ctc_mlo.split(":")]
            # assert len(inter_ctc_mlo.split(":")) - 1 == self.pds_stages, (
            #     inter_ctc_mlo,
            #     self.pds_stages,
            # )
            if self.share_inter_ctc is True:
                self.share_inter_ctc = False
                logger.info("Overwrite the config share_inter_ctc to False for MLO.")
        
        # PDS XCTC
        args.pds_xctc = getattr(args, "pds_xctc", None)
        self.pds_xctc = (
            [int(n) for n in args.pds_xctc.split("_")]
            if args.pds_xctc is not None and getattr(args, "disable_xctc", False) is False
            else None
        )
        self.share_inter_xctc = getattr(args, "share_inter_xctc", False)

        ctc_idx = 0
        inter_ctc = None
        inter_ctc_pae = None

        xctc_idx = 0
        inter_xctc = None
        inter_xctc_pae = None

        self.pae_unnorm_input = getattr(args, "pae_unnorm_input", False)
        self.ctc_pae_ground_truth_ratio = getattr(args, "ctc_pae_ground_truth_ratio", 0)
        if self.ctc_pae_ground_truth_ratio != 0:
            logger.info("CTC ground truth ratio: %.2f." % self.ctc_pae_ground_truth_ratio)
        self.xctc_pae_ground_truth_ratio = getattr(args, "xctc_pae_ground_truth_ratio", 0)
        if self.xctc_pae_ground_truth_ratio != 0:
            logger.info("XCTC ground truth ratio: %.2f." % self.xctc_pae_ground_truth_ratio)

        ctc_pae_strategy = {
            "embed_norm": getattr(args, "pae_embed_norm", False),
            "out_norm": getattr(args, "pae_out_norm", False),
            "ctc_shrink_strategy": getattr(args, "pae_shrink_strategy", None),
            "ctc_temperature": getattr(args, "pae_ctc_temperature", 1.0),
            "distribution_cutoff": getattr(args, "pae_distribution_cutoff", None),
            "gumbel": getattr(args, "pae_gumbel", False),
            "distribution_hard": getattr(args, "pae_distribution_hard", None),
            "gt_ratio": self.ctc_pae_ground_truth_ratio,
            "drop_prob": getattr(args, "pae_drop_prob", 0),
            "linear_init": getattr(args, "pae_linear_init", False)
        }

        xctc_pae_strategy = {
            "embed_norm": getattr(args, "pae_embed_norm", False),
            "out_norm": getattr(args, "pae_out_norm", False),
            "ctc_shrink_strategy": getattr(args, "pae_shrink_strategy", None),
            "ctc_temperature": getattr(args, "pae_ctc_temperature", 1.0),
            "distribution_cutoff": getattr(args, "pae_distribution_cutoff", None),
            "gumbel": getattr(args, "pae_gumbel", False),
            "distribution_hard": getattr(args, "pae_distribution_hard", None),
            "gt_ratio": self.xctc_pae_ground_truth_ratio,
            "drop_prob": getattr(args, "pae_drop_prob", 0),
            "linear_init": getattr(args, "pae_linear_init", False)
        }

        for i in range(self.pds_stages):
            num_layers = self.pds_layers[i]
            ds_ratio = self.pds_ratios[i]

            embed_dim = self.pds_embed_dims[i]
            kernel_size = self.pds_kernel_sizes[i]
            use_pos_embed = self.pds_position_embed[i]
            use_ctc = self.pds_ctc[i] if self.pds_ctc is not None else False
            use_xctc = self.pds_xctc[i] if self.pds_xctc is not None else False

            num_head = self.pds_attn_heads[i]
            ffn_ratio = self.pds_ffn_ratios[i]
            cnn_kernel_size = (
                self.pds_cnn_kernel_sizes[i]
                if self.pds_cnn_kernel_sizes is not None
                else None
            )
            attn_ds_ratio = (
                self.pds_attn_ds_ratios[i]
                if self.pds_conv_strides is not None and self.attn_type == "reduced"
                else 1
            )
            conv_stride = (
                self.pds_conv_strides[i] if self.pds_conv_strides is not None else 1
            )
            attn_stride = (
                self.pds_attn_strides[i] if self.pds_attn_strides is not None else 1
            )
            if conv_stride != 1 or attn_stride != 1:
                expand_embed_dim = (
                    embed_dim
                    if i == self.pds_stages - 1
                    else self.pds_embed_dims[i + 1]
                )
            else:
                expand_embed_dim = None
            
            fusion = self.pds_fusion_layers[i]

            logger.info(
                "The stage {}: layer {}, down-sample ratio {}, embed dim {}, "
                "kernel size {}, position embed {}, ffn ratio {}, num head {}, "
                "attn down-sample ratio {}, conv stride {}, attn stride {}, use ctc {}, use xctc {}, "
                "fusion {}, fusion method {}, fusion transformer {}.".format(
                    i,
                    num_layers,
                    ds_ratio,
                    embed_dim,
                    kernel_size,
                    use_pos_embed,
                    ffn_ratio,
                    num_head,
                    attn_ds_ratio,
                    conv_stride,
                    attn_stride,
                    use_ctc,
                    use_xctc,
                    fusion,
                    self.pds_fusion_method,
                    self.pds_fusion_transform,
                )
            )

            if i == 0:
                self.embed_scale = math.sqrt(embed_dim)
                if args.no_scale_embedding:
                    self.embed_scale = 1.0

            # down-sampling
            if ds_ratio == -1:
                # downsampling = subsampling(args, embed_dim)
                downsampling = subsampling(args)
            else:
                downsampling = Downsampling(
                    self.pds_ds_method,
                    self.pds_embed_norm,
                    args.input_feat_per_channel * args.input_channels
                    if i == 0
                    else self.pds_embed_dims[i - 1],
                    embed_dim,
                    kernel_sizes=kernel_size,
                    stride=ds_ratio,
                    padding=(kernel_size - 1) // 2,
                )

            # position encoding
            if use_pos_embed:
                if self.attn_type in ["rel_pos", "reduced_rel_pos"]:
                    pos_embed = RelPositionalEncoding(
                        args.max_source_positions, embed_dim
                    )
                elif self.attn_type in ["rel_selfattn", "rel_pos_legacy"]:
                    pos_embed = LegacyRelPositionalEncoding(
                        embed_dim, args.dropout, args.max_source_positions
                    )
                elif self.attn_type == "rope":
                    pos_embed = None
                else:  # Use absolute positional embedding
                    pos_embed = PositionalEmbedding(
                        args.max_source_positions, embed_dim, self.padding_idx
                    )
            else:
                pos_embed = None

            stage = nn.ModuleList(
                [
                    PDSTransformerEncoderLayer(
                        args,
                        embed_dim,
                        ffn_ratio,
                        num_head,
                        attn_ds_ratio,
                        conv_stride=conv_stride if layer_idx == num_layers - 1 else 1,
                        attn_stride=attn_stride if layer_idx == num_layers - 1 else 1,
                        expand_embed_dim=expand_embed_dim
                        if layer_idx == num_layers - 1
                        else None,
                        cnn_kernel_size=cnn_kernel_size,
                    )
                    for layer_idx in range(num_layers)
                ]
            )

            # representation fusion
            fusion_pre_layer_norm = None
            fusion_post_layer_norm = None
            fusion_downsampling = None
            if fusion and fusion_stages_num != 0:
                if self.pds_fusion_method == "all" or (
                    self.pds_fusion_method == "same" and self.embed_dim == embed_dim
                ):
                    if i != self.pds_stages - 1:
                        ratio = reduce(lambda a, b: a * b, self.pds_ratios[i + 1 :])
                    else:
                        ratio = 1

                    fusion_pre_layer_norm = LayerNorm(embed_dim)
                    fusion_post_layer_norm = LayerNorm(self.embed_dim)
                    # default conv
                    if self.pds_fusion_transform == "conv":
                        fusion_downsampling = nn.Sequential(
                            Permute120(),
                            nn.Conv1d(
                                embed_dim,
                                self.embed_dim,
                                kernel_size=ratio,
                                stride=ratio,
                            ),
                            nn.BatchNorm1d(self.embed_dim),
                            nn.ReLU(),
                            Permute201(),
                        )
                    elif self.pds_fusion_transform == "pool":
                        fusion_downsampling = nn.Sequential(
                            nn.Conv1d(embed_dim, self.embed_dim, kernel_size=1),
                            nn.BatchNorm1d(self.embed_dim),
                            nn.ReLU(),
                            Permute201(),
                        )
                    elif self.pds_fusion_transform == "conv2":
                        fusion_downsampling = DownSampleConvolutionModule(
                            self.embed_dim,
                            kernel_size=ratio,
                            stride=ratio,
                            input_channels=embed_dim,
                        )
                    elif self.pds_fusion_transform == "conv3":
                        fusion_downsampling = ConvolutionModule(
                            embed_dim,
                            self.embed_dim,
                            depthwise_kernel_size=ratio,
                            stride=ratio,
                            padding=0,
                            dropout=args.dropout
                        )
                    else:
                        logger.error("Unsupported fusion transform!")

            # Inter CTC
            if use_ctc:
                ctc_norm = LayerNorm(embed_dim)
                if not self.share_inter_ctc:
                    ctc = CTC(
                        embed_dim,
                        dictionary_size=len(
                            task.get_source_dictionary(
                                inter_ctc_mlo[ctc_idx] - 1
                                if inter_ctc_mlo != ""
                                else -1
                            )
                        ),
                        dropout=args.dropout,
                    )
                    inter_ctc = ctc
                else:
                    ctc = CTC(
                        embed_dim,
                        dictionary_size=len(task.source_dictionary),
                        dropout=args.dropout,
                    )

                    if (
                        getattr(args, "share_ctc_and_embed", False)
                        and task.source_dictionary == task.target_dictionary
                        and embed_tokens is not None
                        and embed_dim == embed_tokens.embedding_dim
                    ):
                        ctc.ctc_projection.weight = embed_tokens.weight

                    if (
                        inter_ctc is not None
                        and ctc.ctc_projection.weight.shape
                        == inter_ctc.ctc_projection.weight.shape
                    ):
                        ctc = inter_ctc
                    else:
                        inter_ctc = ctc

                if i != self.pds_stages - 1:
                    if not self.share_inter_ctc:
                        ctc_pae = Adapter(
                            embed_dim,
                            args.ctc_pae,
                            len(
                                task.get_source_dictionary(
                                    inter_ctc_mlo[ctc_idx] - 1
                                    if inter_ctc_mlo != ""
                                    else -1
                                )
                            ),
                            strategy=ctc_pae_strategy,
                        )
                    else:
                        ctc_pae = Adapter(
                            embed_dim,
                            args.ctc_pae,
                            len(task.get_source_dictionary(i)),
                            strategy=ctc_pae_strategy,
                        )
                        if (
                            inter_ctc_pae is not None
                            and ctc_pae.dim == inter_ctc_pae.dim
                            and ctc_pae.dict_size == inter_ctc_pae.dict_size
                        ):
                            ctc_pae = inter_ctc_pae
                        else:
                            inter_ctc_pae = ctc_pae
                else:
                    ctc_pae = None
                ctc_idx += 1
            else:
                ctc = None
                ctc_norm = None
                ctc_pae = None

            # Inter XCTC
            if use_xctc:
                xctc_norm = LayerNorm(embed_dim)
                if not self.share_inter_xctc:
                    xctc = CTC(
                        embed_dim,
                        dictionary_size=len(task.target_dictionary),
                        dropout=args.dropout,
                    )
                    inter_xctc = xctc
                else:
                    xctc = CTC(
                        embed_dim,
                        dictionary_size=len(task.target_dictionary),
                        dropout=args.dropout,
                    )

                    if (
                        getattr(args, "share_xctc_and_embed", False)
                        and embed_tokens is not None
                        and embed_dim == embed_tokens.embedding_dim
                    ):
                        xctc.ctc_projection.weight = embed_tokens.weight

                    if (
                        inter_xctc is not None
                        and xctc.ctc_projection.weight.shape
                        == inter_xctc.ctc_projection.weight.shape
                    ):
                        xctc = inter_xctc
                    else:
                        inter_xctc = xctc

                if i != self.pds_stages - 1:
                    if not self.share_inter_xctc:
                        pae = Adapter(
                            embed_dim,
                            args.xctc_pae,
                            len(task.target_dictionary),
                            strategy=xctc_pae_strategy,
                        )
                    else:
                        pae = Adapter(
                            embed_dim,
                            args.xctc_pae,
                            len(task.target_dictionary),
                            strategy=xctc_pae_strategy,
                        )
                        if (
                            inter_xctc_pae is not None
                            and xctc_pae.dim == inter_xctc_pae.dim
                            and pae.dict_size == inter_xctc_pae.dict_size
                        ):
                            xctc_pae = inter_xctc_pae
                        else:
                            inter_xctc_pae = xctc_pae
                else:
                    xctc_pae = None
                xctc_idx += 1
            else:
                xctc = None
                xctc_norm = None
                xctc_pae = None

            setattr(self, f"downsampling{i + 1}", downsampling)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"stage{i + 1}", stage)
            setattr(self, f"fusion_downsampling{i + 1}", fusion_downsampling)
            setattr(self, f"fusion_pre_layer_norm{i + 1}", fusion_pre_layer_norm)
            setattr(self, f"fusion_post_layer_norm{i + 1}", fusion_post_layer_norm)

            setattr(self, f"ctc{i + 1}", ctc)
            setattr(self, f"ctc_norm{i + 1}", ctc_norm)
            setattr(self, f"ctc_pae{i + 1}", ctc_pae)

            setattr(self, f"xctc{i + 1}", xctc)
            setattr(self, f"xctc_norm{i + 1}", xctc_norm)
            setattr(self, f"xctc_pae{i + 1}", xctc_pae)

        if self.fusion_stages_num != 0:
            if pds_fusion_weight is None:
                self.fusion_weight = nn.Parameter(
                    torch.Tensor(self.fusion_stages_num).fill_(1.0)
                )
                self.fusion_weight.data = (
                    self.fusion_weight.data / self.fusion_weight.data.sum(0, keepdim=True)
                )
            else:
                pds_fusion_weight = [float(item) for item in pds_fusion_weight.split("_")]
                self.fusion_weight = torch.Tensor(pds_fusion_weight).float()

        if getattr(args, "pds_final_layers", 0) > 0:
            self.final_layers = nn.ModuleList(
                [
                    PDSTransformerEncoderLayer(
                        args,
                        self.embed_dim,
                        self.pds_ffn_ratios[-1],
                        self.pds_attn_heads[-1],
                        cnn_kernel_size=(
                            self.pds_cnn_kernel_sizes[-1]
                            if self.pds_cnn_kernel_sizes is not None
                            else None
                        ),
                    )
                    for _ in range(args.pds_final_layers)
                ]
            )
        else:
            self.final_layers = None

        self.use_ctc = getattr(args, "ctc_weight", 0) > 0
        if self.use_ctc:
            self.ctc_layer = args.ctc_layer
            self.inter_ctc = True if self.ctc_layer != 0 else False
            if self.inter_ctc:
                logger.info("Intermediate CTC loss in layer %d" % self.ctc_layer)

            embed_dim = self.embed_dim
            if self.inter_ctc:
                ctc_layer = self.ctc_layer
                for i in range(self.pds_stages):
                    ctc_layer -= self.pds_layers[i]
                    if ctc_layer <= 0:
                        embed_dim = self.pds_embed_dims[i]
                        break
            self.ctc = CTC(
                embed_dim,
                dictionary_size=len(task.source_dictionary),
                dropout=args.dropout,
                need_layernorm=True if self.inter_ctc else False,
            )

            if (
                getattr(args, "share_ctc_and_embed", False)
                and task.source_dictionary == task.target_dictionary
                and embed_tokens is not None
                and embed_dim == embed_tokens.embedding_dim
            ):
                self.ctc.ctc_projection.weight = embed_tokens.weight

            if (
                inter_ctc is not None
                and self.ctc.ctc_projection.weight.shape
                == inter_ctc.ctc_projection.weight.shape
            ):
                self.ctc.ctc_projection = inter_ctc.ctc_projection

        # XCTC
        self.use_xctc = getattr(args, "disable_xctc", False) is False and getattr(args, "xctc_weight", 0) > 0
        if self.use_xctc:
            self.xctc_layer = args.xctc_layer
            self.inter_xctc = True if self.xctc_layer != 0 else False
            if self.inter_xctc:
                logger.info("Intermediate XCTC loss in layer %d" % self.xctc_layer)

            embed_dim = self.embed_dim
            if self.inter_xctc:
                xctc_layer = self.xctc_layer
                for i in range(self.pds_stages):
                    xctc_layer -= self.pds_layers[i]
                    if xctc_layer <= 0:
                        embed_dim = self.pds_embed_dims[i]
                        break
            self.xctc = CTC(
                embed_dim,
                dictionary_size=len(task.target_dictionary),
                dropout=args.dropout,
                need_layernorm=True if self.inter_xctc else False,
            )

            if (
                getattr(args, "share_xctc_and_embed", False)
                and embed_tokens is not None
                and embed_dim == embed_tokens.embedding_dim
            ):
                self.xctc.ctc_projection.weight = embed_tokens.weight

            if (
                inter_xctc is not None
                and self.xctc.ctc_projection.weight.shape
                == inter_xctc.ctc_projection.weight.shape
            ):
                self.xctc.ctc_projection = inter_xctc.ctc_projection

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(self.embed_dim)
        else:
            self.layer_norm = None

        # Mixup
        self.mixup = getattr(args, "inter_mixup", False)
        if self.mixup:
            self.mixup_layer = int(args.inter_mixup_layer)
            self.mixup_prob = float(getattr(args, "inter_mixup_prob", 1.0))
            self.mixup_ratio = float(getattr(args, "inter_mixup_ratio", 1.0))
            beta = float(args.inter_mixup_beta)

            from torch.distributions import Beta

            self.beta = Beta(torch.Tensor([beta]), torch.Tensor([beta]))
            logger.info(
                "Use mixup in layer %d with beta %.2f, prob %.2f, ratio %.2f."
                % (self.mixup_layer, beta, self.mixup_prob, self.mixup_ratio)
            )

        # gather cosine similarity
        self.gather_cos_sim = getattr(args, "gather_cos_sim", False)
        self.gather_cos_sim_dis = 2
        self.cos_sim = dict()

    def set_flag(self, **kwargs):
        for i in range(self.pds_stages):
            stage = getattr(self, f"stage{i + 1}")
            for layer in stage:
                if hasattr(layer, "set_flag"):
                    layer.set_flag(**kwargs)

        self.save_rep = kwargs.get("save_rep", False)
        self.mixup_infer = kwargs.get("mixup_infer", False)
        self.gather_cos_sim = kwargs.get("gather_cos_sim", False)
        self.gather_cos_sim_dis = kwargs.get("gather_cos_sim_dis", 2)

        # if self.mixup_infer:
            # self.mixup_keep_org = True

    def dump(self, fstream, info=""):
        for i in range(self.pds_stages):
            idx = 0
            stage = getattr(self, f"stage{i + 1}")
            for layer in stage:
                layer.dump(fstream, "%s Stage %d Layer %d" % (info, i, idx)) if hasattr(layer, "dump") else None
                idx += 1
        
        if self.gather_cos_sim:
            print("Cosine similarity of distance %d" % self.gather_cos_sim_dis, file=fstream)
            for idx, sim in self.cos_sim.items():
                sim = sum(sim) / len(sim) * 100
                print("%s %s: %f" % (info, idx, sim), file=fstream)

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
        sim = sim / 2 / dis / (seq_len - 2 * dis)

        if idx not in self.cos_sim:
            self.cos_sim[idx] = []
        self.cos_sim[idx].append(float(sim))

    def apply_mixup(self, x, encoder_padding_mask):
        batch = x.size(1)
        indices = np.random.permutation(batch)
        if self.mixup_ratio == 1:
            if len(indices) % 2 != 0:
                indices = np.append(indices, (indices[-1]))
            idx1 = indices[0::2]
            idx2 = indices[1::2]

        else:
            mix_size = int(max(2, batch * self.mixup_ratio // 2 * 2))
            mix_indices = indices[:mix_size]
            idx1 = np.append(mix_indices[0::2], (indices[mix_size:]))
            idx2 = np.append(mix_indices[1::2], (indices[mix_size:]))

        idx1 = torch.from_numpy(idx1).to(x.device)
        idx2 = torch.from_numpy(idx2).to(x.device)

        x1 = x[:, idx1]
        x2 = x[:, idx2]

        coef = self.beta.sample().to(x.device).type_as(x)
        x = coef * x1 + (1 - coef) * x2

        pad1 = encoder_padding_mask[idx1]
        pad2 = encoder_padding_mask[idx2]
        encoder_padding_mask = pad1 & pad2
        input_lengths = (~encoder_padding_mask).sum(-1)

        mixup = {
            "coef": coef,
            "index1": idx1,
            "index2": idx2,
        }
        return x, encoder_padding_mask, input_lengths, mixup

    def set_ctc_infer(
        self, ctc_infer, post_process, src_dict=None, tgt_dict=None, path=None
    ):
        if hasattr(self, "ctc"):
            assert src_dict is not None
            self.ctc.set_infer(
                ctc_infer,
                post_process,
                src_dict,
                path=path + ".ctc" if path is not None else None,
            )

    def ctc_valid(self, lprobs, targets, input_lengths, dictionary, lang="source"):
        if lang == "target" and hasattr(self, "xctc"):
            return self.xctc.valid(lprobs, targets, input_lengths, dictionary)
        if hasattr(self, "ctc"):
            return self.ctc.valid(lprobs, targets, input_lengths, dictionary)
        for i in torch.arange(self.pds_stages - 1, -1, -1):
            ctc = getattr(self, f"ctc{i + 1}")
            if ctc is not None:
                return ctc.valid(lprobs, targets, input_lengths, dictionary)

        logger.error("No ctc module in the encoder")

    def forward(self, src_tokens, src_lengths, **kwargs):

        batch = src_tokens.size(0)

        x = src_tokens.transpose(0, 1)
        input_lengths = src_lengths

        # padding to the multiply of 2
        max_len = x.size(0)
        length = reduce(lambda a, b: max(1, a) * max(1, b), self.pds_ratios)
        padding_to_len = length - max_len % length
        if length > 1 and padding_to_len > 0:
            padding_for_pds = x.new_zeros((padding_to_len, batch, x.size(2)))
            x = torch.cat([x, padding_for_pds], dim=0)

        encoder_padding_mask = lengths_to_padding_mask_with_maxlen(
            input_lengths, x.size(0)
        )

        # gather cosine similarity
        self.add_to_dict(x, "input")

        layer_idx = 0
        mixup = None
        prev_state = []
        prev_padding = []

        ctc_logit = None
        inter_ctc_logits = []
        xctc_logit = None
        inter_xctc_logits = []
        for i in range(self.pds_stages):
            downsampling = getattr(self, f"downsampling{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            stage = getattr(self, f"stage{i + 1}")
            ctc = getattr(self, f"ctc{i + 1}")
            ctc_norm = getattr(self, f"ctc_norm{i + 1}")
            ctc_pae = getattr(self, f"ctc_pae{i + 1}")

            xctc = getattr(self, f"xctc{i + 1}")
            xctc_norm = getattr(self, f"xctc_norm{i + 1}")
            xctc_pae = getattr(self, f"xctc_pae{i + 1}")

            if self.training and self.mixup and layer_idx == self.mixup_layer:
                x, encoder_padding_mask, input_lengths, mixup = self.apply_mixup(
                    x, encoder_padding_mask
                )

            x, input_lengths = downsampling(x, input_lengths)
            encoder_padding_mask = lengths_to_padding_mask_with_maxlen(
                input_lengths, x.size(0)
            )

            # gather cosine similarity
            self.add_to_dict(x, "Stage %d input" % i)

            # add the position encoding and dropout
            if pos_embed:
                if self.attn_type in [
                    "rel_pos",
                    "rel_pos_legacy",
                    "rel_selfattn",
                    "reduced_rel_pos",
                ]:
                    positions = pos_embed(x)

                elif self.attn_type == "rope":
                    positions = None

                else:
                    positions = pos_embed(encoder_padding_mask).transpose(0, 1)
                    x += positions
                    positions = None
            else:
                positions = None

            if i == 0:
                x = self.dropout(x)
            else:
                x = self.pds_dropout(x)

            for layer in stage:
                x = layer(x, encoder_padding_mask, pos_emb=positions)
                layer_idx += 1

                if layer.conv_stride > 1:

                    # Stride Mask (B, 1, T // S, T // S)
                    if encoder_padding_mask is not None:
                        encoder_padding_mask = encoder_padding_mask[
                            :, :: layer.conv_stride
                        ]

                    # Update Seq Lengths
                    if input_lengths is not None:
                        input_lengths = (
                            torch.div(
                                input_lengths - 1,
                                layer.conv_stride,
                                rounding_mode="floor",
                            )
                            + 1
                        )

                # gather cosine similarity
                self.add_to_dict(x, "Stage %d Layer %d" % (i, layer_idx))

                if self.use_ctc and self.inter_ctc and self.ctc_layer == layer_idx:
                    ctc_logit = self.ctc(
                        x.clone(), encoder_padding_mask, "Source Layer %d" % layer_idx
                    )
                if self.use_xctc and self.inter_xctc and self.xctc_layer == layer_idx:
                    xctc_logit = self.xctc(
                        x.clone(), encoder_padding_mask, "Source Layer %d" % layer_idx
                    )                

            prev_state.append(x.clone())
            prev_padding.append(encoder_padding_mask)

            # Inter CTC
            if ctc is not None:
                norm_x = ctc_norm(x)
                logit = ctc(norm_x, encoder_padding_mask, "Source Layer %d" % layer_idx)

                pae_input = x if self.pae_unnorm_input else norm_x
                if ctc_pae is not None and ctc_pae.adapter_type != "none":
                    x, encoder_padding_mask = ctc_pae(
                        [pae_input, logit], encoder_padding_mask
                    )

                inter_ctc_logits.append([logit.clone(), encoder_padding_mask])

            # Inter XCTC
            if xctc is not None:
                norm_x = xctc_norm(x)
                logit = xctc(norm_x, encoder_padding_mask, "Source Layer %d" % layer_idx)

                pae_input = x if self.pae_unnorm_input else norm_x
                if xctc_pae is not None and xctc_pae.adapter_type != "none":
                    x, encoder_padding_mask = xctc_pae(
                        [pae_input, logit], encoder_padding_mask
                    )

                inter_xctc_logits.append([logit.clone(), encoder_padding_mask])

        if self.fusion_stages_num != 0:
            fusion_state = []
            i = -1
            seq_len = x.size(0)
            for state in prev_state:
                i += 1

                if self.fusion_mask:
                    padding = prev_padding[i]
                    if padding is not None:
                        zero_padding = padding.unsqueeze(2)
                        state = state.transpose(0, 1)
                        state = state.masked_fill(zero_padding, 0.0)
                        state = state.transpose(0, 1)

                fusion_downsampling = getattr(self, f"fusion_downsampling{i + 1}")
                fusion_pre_layer_norm = getattr(self, f"fusion_pre_layer_norm{i + 1}")
                fusion_post_layer_norm = getattr(self, f"fusion_post_layer_norm{i + 1}")

                if (
                    fusion_pre_layer_norm is not None
                    or fusion_pre_layer_norm is not None
                ):
                    if not self.fusion_no_prenorm:
                        state = fusion_pre_layer_norm(state)

                    if self.pds_fusion_transform == "conv":
                        state = fusion_downsampling(state)
                    elif self.pds_fusion_transform == "conv2":
                        state = fusion_downsampling(state, prev_padding[i])
                    elif self.pds_fusion_transform == "conv3":
                        state = state.transpose(0, 1)
                        state = fusion_downsampling(state, prev_padding[i])
                        state = state.transpose(0, 1)
                    elif self.pds_fusion_transform == "pool":
                        state = state.permute(1, 2, 0)  # bsz, dim, seq_len
                        if i != self.pds_stages - 1:
                            state = nn.functional.adaptive_max_pool1d(state, seq_len)
                        state = fusion_downsampling(state)

                    state = fusion_post_layer_norm(state)
                    fusion_state.append(state)
            if self.fusion_weight.device != x.device:
                self.fusion_weight = self.fusion_weight.to(x.device).to(x.dtype)
            x = (
                torch.stack(fusion_state, dim=0) * self.fusion_weight.view(-1, 1, 1, 1)
            ).sum(0)

        if self.final_layers is not None:
            # add the position encoding and dropout
            if pos_embed:
                if self.attn_type in [
                    "rel_pos",
                    "rel_pos_legacy",
                    "rel_selfattn",
                    "reduced_rel_pos",
                ]:
                    positions = pos_embed(x)

                elif self.attn_type == "rope":
                    positions = None

                else:
                    positions = pos_embed(encoder_padding_mask).transpose(0, 1)
                    x += positions
                    positions = None
            else:
                positions = None
            x = self.pds_dropout(x)

            for layer in self.final_layers:
                x = layer(x, encoder_padding_mask, pos_emb=positions)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if self.use_ctc and ctc_logit is None:
            ctc_logit = self.ctc(x, encoder_padding_mask, is_top=True)

        if self.use_xctc and xctc_logit is None:
            xctc_logit = self.xctc(x, encoder_padding_mask, is_top=True)

        return {
            "encoder_out": [x],  # T x B x C
            "ctc_logit": [] if ctc_logit is None else [ctc_logit],  # T x B x C
            "inter_ctc_logits": inter_ctc_logits,  # T x B x C
            "xctc_logit": [] if xctc_logit is None else [xctc_logit],  # T x B x C
            "inter_xctc_logits": inter_ctc_logits,  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "mixup": mixup,
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def get_loss(self):
        if not self.pds_fusion:
            return 0

        weight = self.fusion_weight
        loss = 0
        for i in range(self.fusion_stages_num - 1):
            sub = weight[i] - weight[i + 1]
            if sub > 0:
                loss += sub
            if weight[i] < 0:
                loss += weight[i]
        loss += (0.5 * (weight.sum() - 1.0) ** 2).mean()

        return loss

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
            else [
                x.index_select(1, new_order)
                for x in encoder_out["xctc_logit"]
                if x is not None
            ]
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
            "xctc_logit": new_xctc_logit,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }


@register_model_architecture(
    model_name="pdss2t_transformer", arch_name="pdss2t_transformer"
)
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
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    args.embed_linear = getattr(args, "embed_linear", False)

    # CTC
    args.ctc_layer = getattr(args, "ctc_layer", 0)
    args.share_ctc_and_embed = getattr(args, "share_ctc_and_embed", False)

    # Conformer
    args.encoder_activation_fn = getattr(args, "encoder_activation_fn", "relu")
    args.macaron_style = getattr(args, "macaron_style", False)
    args.use_cnn_module = getattr(args, "use_cnn_module", False)
    args.cnn_module_kernel = getattr(args, "cnn_module_kernel", 31)
    args.cnn_module_norm = getattr(args, "cnn_module_norm", "batch_norm")

    # Relative position encoding
    args.max_encoder_relative_length = getattr(args, "max_encoder_relative_length", -1)
    args.max_decoder_relative_length = getattr(args, "max_decoder_relative_length", -1)
    args.k_only = getattr(args, "k_only", True)

    # intermediate CTC
    args.inter_ctc_layers = getattr(args, "inter_ctc_layers", None)
    args.share_inter_ctc_norm = getattr(args, "share_inter_ctc_norm", False)
    args.pae_ctc_temperature = getattr(args, "pae_ctc_temperature", 1)
    args.inter_ctc_drop_prob = getattr(args, "inter_ctc_drop_prob", 0)
    args.ctc_pae_ground_truth_ratio = getattr(args, "ctc_pae_ground_truth_ratio", 0)

    # Prediction-aware encoding (pae)
    args.ctc_pae = getattr(args, "ctc_pae", "none")
    args.share_pae_and_ctc = getattr(args, "share_pae_and_ctc", False)
    args.pae_embed_norm = getattr(args, "pae_embed_norm", False)
    args.pae_out_norm = getattr(args, "pae_out_norm", False)
    args.pae_drop_prob = getattr(args, "pae_drop_prob", 0)
    args.pae_distribution_cutoff = getattr(args, "pae_distribution_cutoff", None)
    args.pae_distribution_hard = getattr(args, "pae_distribution_hard", False)
    args.pae_gumbel = getattr(args, "pae_gumbel", False)
    args.pae_unnorm_input = getattr(args, "pae_unnorm_input", False)

    # XCTC
    args.xctc_layer = getattr(args, "xctc_layer", 0)
    args.share_xctc_and_embed = getattr(args, "share_xctc_and_embed", False)
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
    args.inter_mixup_layer = getattr(args, "inter_mixup_layer", None)
    args.inter_mixup_beta = getattr(args, "inter_mixup_beta", 0.5)
    args.inter_mixup_prob = getattr(args, "inter_mixup_prob", 1)
    args.inter_mixup_ratio = getattr(args, "inter_mixup_ratio", 1)

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

    args.pds_dropout = getattr(args, "pds_dropout", 0)
    args.pds_fusion = getattr(args, "pds_fusion", False)
    args.pds_fusion_layers = getattr(args, "pds_fusion_layers", None)
    args.pds_fusion_method = getattr(args, "pds_fusion_method", "none")
    args.pds_final_layers = getattr(args, "pds_final_layers", 0)


def set_pds_base_8(args):
    args.pds_stages = getattr(args, "pds_stages", 4)
    args.pds_ratios = getattr(args, "pds_ratios", "2_2_1_2")
    args.pds_layers = getattr(args, "pds_layers", "3_3_3_3")
    args.pds_kernel_sizes = getattr(args, "pds_kernel_sizes", "5_5_5_5")
    args.pds_position_embed = getattr(args, "pds_position_embed", "1_1_1_1")


def set_pds_base_16(args):
    args.pds_stages = getattr(args, "pds_stages", 4)
    args.pds_ratios = getattr(args, "pds_ratios", "2_2_2_2")
    args.pds_layers = getattr(args, "pds_layers", "2_2_6_2")
    args.pds_kernel_sizes = getattr(args, "pds_kernel_sizes", "5_5_5_5")
    args.pds_position_embed = getattr(args, "pds_position_embed", "1_1_1_1")


def set_pds_base_32(args):
    args.pds_stages = getattr(args, "pds_stages", 5)
    args.pds_ratios = getattr(args, "pds_ratios", "2_2_2_2_2")
    args.pds_layers = getattr(args, "pds_layers", "2_2_3_3_2")
    args.pds_kernel_sizes = getattr(args, "pds_kernel_sizes", "5_5_5_5_5")
    args.pds_position_embed = getattr(args, "pds_position_embed", "1_1_1_1_1")


def set_pds_deep_8(args):
    args.pds_stages = getattr(args, "pds_stages", 4)
    args.pds_ratios = getattr(args, "pds_ratios", "2_2_1_2")
    args.pds_layers = getattr(args, "pds_layers", "7_7_7_9")
    args.pds_kernel_sizes = getattr(args, "pds_kernel_sizes", "5_5_5_5")
    args.pds_position_embed = getattr(args, "pds_position_embed", "1_1_1_1")


def set_pds_deep_16(args):
    args.pds_stages = getattr(args, "pds_stages", 4)
    args.pds_ratios = getattr(args, "pds_ratios", "2_2_2_2")
    args.pds_layers = getattr(args, "pds_layers", "5_5_12_8")
    args.pds_kernel_sizes = getattr(args, "pds_kernel_sizes", "5_5_5_5")
    args.pds_position_embed = getattr(args, "pds_position_embed", "1_1_1_1")


def set_pds_deep_32(args):
    args.pds_stages = getattr(args, "pds_stages", 5)
    args.pds_ratios = getattr(args, "pds_ratios", "2_2_2_2_2")
    args.pds_layers = getattr(args, "pds_layers", "5_5_7_7_6")
    args.pds_kernel_sizes = getattr(args, "pds_kernel_sizes", "5_5_5_5_5")
    args.pds_position_embed = getattr(args, "pds_position_embed", "1_1_1_1_1")


@register_model_architecture("pdss2t_transformer", "pdss2t_transformer_s")
def pdss2t_transformer_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)

    # PDS
    set_pds_base_16(args)
    args.pds_embed_dims = getattr(args, "pds_embed_dims", "256_256_256_256")
    args.pds_attn_heads = getattr(args, "pds_attn_heads", "4_4_4_4")
    args.pds_ffn_ratios = getattr(args, "pds_ffn_ratios", "8_8_8_8")

    base_architecture(args)


@register_model_architecture("pdss2t_transformer", "pdss2t_transformer_s_8")
def pdss2t_transformer_s_8(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)

    # PDS
    set_pds_base_8(args)
    args.pds_embed_dims = getattr(args, "pds_embed_dims", "256_256_256_256")
    args.pds_attn_heads = getattr(args, "pds_attn_heads", "4_4_4_4")
    args.pds_ffn_ratios = getattr(args, "pds_ffn_ratios", "8_8_8_8")

    base_architecture(args)


@register_model_architecture("pdss2t_transformer", "pdss2t_transformer_s_16")
def pdss2t_transformer_s_16(args):
    pdss2t_transformer_s(args)


@register_model_architecture("pdss2t_transformer", "pdss2t_transformer_s_32")
def pdss2t_transformer_s_32(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)

    # PDS
    set_pds_base_32(args)
    args.pds_embed_dims = getattr(args, "pds_embed_dims", "256_256_256_256")
    args.pds_attn_heads = getattr(args, "pds_attn_heads", "4_4_4_4_4")
    args.pds_ffn_ratios = getattr(args, "pds_ffn_ratios", "8_8_8_8_8")

    base_architecture(args)


@register_model_architecture("pdss2t_transformer", "pdss2t_transformer_sd")
def pdss2t_transformer_sd(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)

    # PDS
    set_pds_deep_16(args)
    args.pds_embed_dims = getattr(args, "pds_embed_dims", "256_256_256_256")
    args.pds_attn_heads = getattr(args, "pds_attn_heads", "4_4_4_4")
    args.pds_ffn_ratios = getattr(args, "pds_ffn_ratios", "8_8_8_8")

    base_architecture(args)


@register_model_architecture("pdss2t_transformer", "pdss2t_transformer_sd_8")
def pdss2t_transformer_sd_8(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)

    # PDS
    set_pds_deep_8(args)
    args.pds_embed_dims = getattr(args, "pds_embed_dims", "256_256_256_256")
    args.pds_attn_heads = getattr(args, "pds_attn_heads", "4_4_4_4")
    args.pds_ffn_ratios = getattr(args, "pds_ffn_ratios", "8_8_8_8")

    base_architecture(args)


@register_model_architecture("pdss2t_transformer", "pdss2t_transformer_sd_16")
def pdss2t_transformer_sd_16(args):
    pdss2t_transformer_sd(args)


@register_model_architecture("pdss2t_transformer", "pdss2t_transformer_sd_32")
def pdss2t_transformer_sd_32(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)

    # PDS
    set_pds_deep_32(args)
    args.pds_embed_dims = getattr(args, "pds_embed_dims", "256_256_256_256")
    args.pds_attn_heads = getattr(args, "pds_attn_heads", "4_4_4_4")
    args.pds_ffn_ratios = getattr(args, "pds_ffn_ratios", "8_8_8_8")

    base_architecture(args)


@register_model_architecture("pdss2t_transformer", "pdss2t_transformer_m")
def pdss2t_transformer_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)

    # PDS
    set_pds_base_16(args)
    args.pds_embed_dims = getattr(args, "pds_embed_dims", "512_512_512_512")
    args.pds_attn_heads = getattr(args, "pds_attn_heads", "8_8_8_8")
    args.pds_ffn_ratios = getattr(args, "pds_ffn_ratios", "4_4_4_4")

    base_architecture(args)


@register_model_architecture("pdss2t_transformer", "pdss2t_transformer_m_8")
def pdss2t_transformer_m_8(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)

    # PDS
    set_pds_base_8(args)
    args.pds_embed_dims = getattr(args, "pds_embed_dims", "512_512_512_512")
    args.pds_attn_heads = getattr(args, "pds_attn_heads", "8_8_8_8")
    args.pds_ffn_ratios = getattr(args, "pds_ffn_ratios", "4_4_4_4")

    base_architecture(args)


@register_model_architecture("pdss2t_transformer", "pdss2t_transformer_m_16")
def pdss2t_transformer_m_16(args):
    pdss2t_transformer_m(args)


@register_model_architecture("pdss2t_transformer", "pdss2t_transformer_m_32")
def pdss2t_transformer_m_32(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)

    # PDS
    set_pds_base_32(args)
    args.pds_embed_dims = getattr(args, "pds_embed_dims", "512_512_512_512")
    args.pds_attn_heads = getattr(args, "pds_attn_heads", "8_8_8_8")
    args.pds_ffn_ratios = getattr(args, "pds_ffn_ratios", "4_4_4_4")

    base_architecture(args)


@register_model_architecture("pdss2t_transformer", "pdss2t_transformer_l")
def pdss2t_transformer_l(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.2)

    # PDS
    set_pds_base_16(args)
    args.pds_embed_dims = getattr(args, "pds_embed_dims", "1024_1024_1024_1024")
    args.pds_attn_heads = getattr(args, "pds_attn_heads", "16_16_16_16")
    args.pds_ffn_ratios = getattr(args, "pds_ffn_ratios", "4_4_4_4")

    base_architecture(args)
