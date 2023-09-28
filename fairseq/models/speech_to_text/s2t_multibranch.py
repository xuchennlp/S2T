import copy
from email.policy import default
import math
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from fairseq import checkpoint_utils, utils

from fairseq.models.speech_to_text import (
    S2TTransformerModel,
    S2TTransformerEncoder,
    PDSS2TTransformerModel,
    PDSS2TTransformerEncoder,
    S2TSATEModel,
)
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    LegacyRelPositionalEncoding,
    RelPositionalEncoding,
    S2TTransformerS2EncoderLayer,
)
from fairseq.modules.speech_to_text import Adapter, build_adapter, CTC
from fairseq.models.transformer_s2 import (
    Embedding,
    TransformerS2Encoder,
    TransformerS2Decoder,
)

logger = logging.getLogger(__name__)


@register_model("s2t_multibranch")
class S2TMultiBranchModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        S2TTransformerModel.add_args(parser)
        PDSS2TTransformerModel.add_specific_args(parser)
        S2TSATEModel.add_specific_args(parser)
        S2TMultiBranchModel.add_specific_args(parser)

    @staticmethod
    def add_specific_args(parser):
        # multibranch
        parser.add_argument(
            "--junior-acoustic-encoder",
            default="transformer",
            choices=["transformer", "pds", "sate", "wav2vec"],
            type=str,
            help="the architecture of the junior acoustic encoder",
        )
        parser.add_argument(
            "--senior-acoustic-encoder",
            default="transformer",
            choices=["transformer", "pds", "sate", "wav2vec"],
            type=str,
            help="the architecture of the senior acoustic ASR encoder",
        )
        parser.add_argument(
            "--textual-encoder",
            default="transformer",
            type=str,
            help="the architecture of the MT encoder",
        )
        parser.add_argument(
            "--textual-encoder-dim",
            type=int,
            help="the dimension of the textual encoder",
        )
        parser.add_argument(
            "--junior-acoustic-encoder-layers",
            default=6,
            type=int,
            help="the layers of the senior acoustic encoder",
        )
        parser.add_argument(
            "--senior-acoustic-encoder-layers",
            default=6,
            type=int,
            help="the layers of the senior acoustic encoder",
        )
        parser.add_argument(
            "--textual-encoder-layers",
            default=6,
            type=int,
            help="the layers of the textual encoder",
        )
        parser.add_argument(
            "--acoustic-adapter",
            default="none",
            type=str,
            help="acoustic adapter",
        )
        parser.add_argument(
            "--textual-adapter",
            default="none",
            type=str,
            help="textual adapter",
        )
        # collaboration
        parser.add_argument(
            "--collaboration-direction",
            default="none",
            type=str,
            help="direction of collaboration",
        )
        parser.add_argument(
            "--collaboration-start",
            default="0:0",
            type=str,
            help="start collaboration in two encoders",
        )
        parser.add_argument(
            "--collaboration-step",
            default="1:1",
            type=str,
            help="collaboration step in two encoders",
        )
        parser.add_argument(
            "--encoder-collaboration-mode",
            default="serial",
            type=str,
            help="how to calculate attention during league in encoder",
        )
        parser.add_argument(
            "--decoder-collaboration-mode",
            default="serial",
            type=str,
            help="how to calculate attention during league in encoder",
        )
        # league
        parser.add_argument(
            "--encoder-league-s1-ratio",
            default=0.5,
            type=float,
            help="league ratio of the s1 representation",
        )
        parser.add_argument(
            "--encoder-league-s2-ratio",
            default=0.5,
            type=float,
            help="league ratio of the s2 representation",
        )
        parser.add_argument(
            "--encoder-league-out-norm",
            action="store_true",
            help="layer normalization before league in the encoder",
        )
        parser.add_argument(
            "--encoder-league-gated",
            action="store_true",
            help="league with the gated mechanism in the encoder",
        )
        parser.add_argument(
            "--encoder-league-drop-net",
            action="store_true",
            help="drop one input during league",
        )
        parser.add_argument(
            "--encoder-league-drop-net-prob",
            default=0.0,
            type=float,
            help="probability of dropping one representations",
        )
        parser.add_argument(
            "--encoder-league-drop-net-mix",
            action="store_true",
            help="mix the two input with any probability",
        )

        parser.add_argument(
            "--decoder-league-s1-ratio",
            default=0.5,
            type=float,
            help="league ratio of the s1 representation",
        )
        parser.add_argument(
            "--decoder-league-s2-ratio",
            default=0.5,
            type=float,
            help="league ratio of the s2 representation",
        )
        parser.add_argument(
            "--decoder-league-out-norm",
            action="store_true",
            help="layer normalization before league in the decoder",
        )
        parser.add_argument(
            "--decoder-league-gated",
            action="store_true",
            help="league with the gated mechanism in the decoder",
        )
        parser.add_argument(
            "--decoder-league-drop-net",
            action="store_true",
            help="drop one input during league",
        )
        parser.add_argument(
            "--decoder-league-drop-net-prob",
            default=0.0,
            type=float,
            help="probability of dropping one representations",
        )
        parser.add_argument(
            "--decoder-league-drop-net-mix",
            action="store_true",
            help="mix the two input with any probability",
        )

        parser.add_argument(
            "--load-pretrained-junior-acoustic-encoder-from",
            type=str,
            metavar="STR",
            help="model to take junior acoustic encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--load-pretrained-senior-acoustic-encoder-from",
            type=str,
            metavar="STR",
            help="model to take senior acoustic encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--load-pretrained-textual-encoder-from",
            type=str,
            metavar="STR",
            help="model to take textual encoder weights from (for initialization)",
        )
        # multi-modality modeling
        parser.add_argument(
            "--use-raw-text",
            action="store_true",
            help="encoding by the raw text in the textual encoder",
        )
        parser.add_argument(
            "--modality-switch",
            action="store_true",
            help="encoding by the raw text in the textual encoder",
        )
        parser.add_argument(
            "--text-to-hidden-progress",
            default=None,
            type=str,
            help="encoding by the raw text in the textual encoder",
        )
        pass

    @classmethod
    def build_encoder(cls, args, task=None, embed_tokens=None):
        encoder = S2TMultiBranchEncoder(args, task, embed_tokens)

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
        if getattr(args, "load_pretrained_junior_encoder_from", None):
            encoder.junior_acoustic_encoder = (
                checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder.junior_acoustic_encoder,
                    checkpoint=args.load_pretrained_junior_encoder_from,
                    strict=False,
                )
            )
            logger.info(
                f"loaded pretrained junior acoustic encoder from: "
                f"{args.load_pretrained_junior_encoder_from}"
            )
        if getattr(args, "load_pretrained_senior_encoder_from", None):
            encoder.senior_acoustic_encoder = (
                checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder.senior_acoustic_encoder,
                    checkpoint=args.load_pretrained_senior_encoder_from,
                    strict=False,
                )
            )
            logger.info(
                f"loaded pretrained senior acoustic encoder from: "
                f"{args.load_pretrained_senior_encoder_from}"
            )
        if getattr(args, "load_pretrained_textual_encoder_from", None):
            encoder.textual_encoder = (
                checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder.textual_encoder,
                    checkpoint=args.load_pretrained_textual_encoder_from,
                    strict=False,
                )
            )
            logger.info(
                f"loaded pretrained textual encoder from: "
                f"{args.load_pretrained_textual_encoder_from}"
            )
        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        decoder = TransformerS2Decoder(args, task.target_dictionary, embed_tokens)

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

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            encoder_embed_tokens = build_embedding(src_dict, args.encoder_embed_dim)
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(src_dict, args.encoder_embed_dim)
            decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim)

        encoder = cls.build_encoder(args, task, encoder_embed_tokens)
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
        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        return decoder_out


class S2TMultiBranchEncoder(FairseqEncoder):
    """Speech-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args, task=None, embed_tokens=None):
        super().__init__(None)
        self.padding_idx = 1
        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )

        # junior acoustic encoder
        jae_args = copy.deepcopy(args)
        setattr(jae_args, "encoder_layers", args.junior_acoustic_encoder_layers)

        junior_encoder_type = args.junior_acoustic_encoder
        if junior_encoder_type == "transformer":
            self.junior_acoustic_encoder = S2TTransformerEncoder(
                args, task, embed_tokens
            )
        elif junior_encoder_type == "pds":
            self.junior_acoustic_encoder = PDSS2TTransformerEncoder(
                args, task, embed_tokens
            )
        else:
            logger.error(
                "Unsupported junior acoustic architecture: %s." % junior_encoder_type
            )

        # senior acoustic encoder
        sae_args = copy.deepcopy(args)
        self.senior_acoustic_encoder_layer_num = args.senior_acoustic_encoder_layers
        setattr(sae_args, "encoder_layers", args.senior_acoustic_encoder_layers)
        if self.senior_acoustic_encoder_layer_num > 0:
            # adapter for acoustic encoder
            self.ae_adapter = build_adapter(
                args,
                args.acoustic_adapter,
                args.encoder_embed_dim,
                task.source_dictionary,
            )
            assert not (
                args.share_adapter_and_ctc and args.share_adapter_and_embed
            ), "Can not be True at the same time"
            if (
                args.share_adapter_and_ctc
                and hasattr(self.ae_adapter, "embed_adapter")
                and hasattr(self.junior_acoustic_encoder, "ctc")
            ):
                self.ae_adapter.embed_adapter.weight = (
                    self.junior_acoustic_encoder.ctc.ctc_projection.weight
                )
            if args.share_adapter_and_embed and hasattr(self.adapter, "embed_adapter"):
                self.ae_adapter.embed_adapter.weight = embed_tokens.weight

            self.senior_acoustic_encoder = S2TTransformerS2Encoder(
                sae_args, task.source_dictionary, embed_tokens
            )
        else:
            self.senior_acoustic_encoder = None
            self.ae_adapter = None

        # adapter for textual encoder
        self.te_adapter = build_adapter(
            args, args.textual_adapter, args.encoder_embed_dim, task.source_dictionary
        )
        assert not (
            args.share_adapter_and_ctc and args.share_adapter_and_embed
        ), "Can not be True at the same time"
        if (
            args.share_adapter_and_ctc
            and hasattr(self.te_adapter, "embed_adapter")
            and hasattr(self.junior_acoustic_encoder, "ctc")
        ):
            self.te_adapter.embed_adapter.weight = (
                self.junior_acoustic_encoder.ctc.ctc_projection.weight
            )
        if args.share_adapter_and_embed and hasattr(self.adapter, "embed_adapter"):
            self.te_adapter.embed_adapter.weight = embed_tokens.weight

        # textual encoder
        self.textual_encoder_layer_num = args.textual_encoder_layers
        te_args = copy.deepcopy(args)
        setattr(te_args, "encoder_attention_type", "selfattn")
        setattr(te_args, "encoder_layers", args.textual_encoder_layers)
        self.textual_encoder = TransformerS2Encoder(
            te_args, task.source_dictionary, embed_tokens
        )

        # collaboration
        collaboration_step = args.collaboration_step
        self.collaboration_direction = args.collaboration_direction

        collaboration_start = args.collaboration_start
        if len(collaboration_start.split(":")) == 2:
            self.collaboration_start = [int(s) for s in collaboration_start.split(":")]
        elif len(collaboration_start.split(":")) == 1:
            self.collaboration_start = [
                int(collaboration_start),
                int(collaboration_start),
            ]
        else:
            self.collaboration_start = [0, 0]

        if len(collaboration_step.split(":")) == 2:
            self.collaboration_step = [int(s) for s in collaboration_step.split(":")]
        elif len(collaboration_step.split(":")) == 1:
            self.collaboration_step = [int(collaboration_step), int(collaboration_step)]
        else:
            self.collaboration_step = [1, 1]

        # multi-modality modeling
        self.use_raw_text = args.use_raw_text
        self.modality_switch = False
        if args.modality_switch and args.text_to_hidden_progress is not None:
            self.modality_switch = args.modality_switch
            progress_items = args.text_to_hidden_progress.split(":")
            assert len(progress_items) == 3
            self.switch_start, self.switch_end, self.all_step = [
                float(item) for item in progress_items
            ]
            self.switch_step = (self.switch_start - self.switch_end) / self.all_step
            self.text_curr_ratio = self.switch_start

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.update_num = num_updates
        if self.modality_switch and num_updates <= self.all_step:
            self.text_curr_ratio = self.switch_start - self.switch_step * num_updates

    def set_ctc_infer(
        self, ctc_infer, post_process, src_dict=None, tgt_dict=None, path=None
    ):
        self.junior_acoustic_encoder.set_ctc_infer(
            ctc_infer, post_process, src_dict=src_dict, tgt_dict=tgt_dict, path=path
        )

    def ctc_valid(self, lprobs, targets, input_lengths, dictionary, lang="source"):
        return self.junior_acoustic_encoder.ctc_valid(
            lprobs, targets, input_lengths, dictionary, lang
        )

    def forward(
        self,
        src_tokens,
        src_lengths=None,
        text_src_tokens=None,
        text_src_lengths=None,
        **kwargs,
    ):
        # junior acoustic encoder
        jae_out = self.junior_acoustic_encoder(src_tokens, src_lengths, **kwargs)
        jae_x = jae_out["encoder_out"][0]
        jae_padding_mask = jae_out["encoder_padding_mask"][0]
        jae_out["ctc_padding_mask"] = [jae_padding_mask]

        if "ctc_logit" in jae_out and len(jae_out["ctc_logit"]) > 0:
            ctc_logit = jae_out["ctc_logit"][0]
        else:
            ctc_logit = None

        # acoustic input
        if self.senior_acoustic_encoder is not None:
            ae_x, ae_padding_mask = self.ae_adapter(
                (jae_x, ctc_logit), jae_padding_mask
            )
            ae_x = self.senior_acoustic_encoder.forward_input(ae_x, ae_padding_mask)
        else:
            ae_x = None
            ae_padding_mask = None

        # textual input
        if self.use_raw_text and text_src_tokens is not None:
            te_token_x = self.textual_encoder.forward_embedding(text_src_tokens)[0]
            te_token_padding_mask = text_src_tokens.eq(self.padding_idx)
            if self.modality_switch:
                te_adapter_x, te_adapter_padding_mask = self.te_adapter(
                    (jae_x, ctc_logit), jae_padding_mask
                )
                te_adapter_x = te_adapter_x.transpose(
                    0, 1
                ) + self.textual_encoder.embed_positions(te_adapter_padding_mask)
                te_adapter_x = self.dropout_module(te_adapter_x)

                if self.text_curr_ratio == 1:
                    te_x = te_token_x.transpose(0, 1)
                    te_padding_mask = te_token_padding_mask
                elif 0 < self.text_curr_ratio < 1:
                    if te_token_x.size(1) != te_adapter_x.size(1):
                        bsz = te_token_x.size(0)
                        length_diff = int(
                            math.fabs(te_token_x.size(1) - te_adapter_x.size(1))
                        )

                        zero_embed = (
                            torch.Tensor([0])
                            .type_as(te_token_x)
                            .to(te_token_x.device)
                            .unsqueeze(1)
                            .unsqueeze(2)
                            .repeat([bsz, length_diff, int(te_token_x.size(2))])
                        )
                        zero_padding = (
                            torch.Tensor([True])
                            .bool()
                            .to(te_token_padding_mask.device)
                            .unsqueeze(1)
                            .repeat([bsz, length_diff])
                        )
                        if te_token_x.size(1) > te_adapter_x.size(1):
                            te_adapter_x = torch.cat([te_adapter_x, zero_embed], dim=1)
                            te_adapter_padding_mask = torch.cat(
                                [te_adapter_padding_mask, zero_padding], dim=1
                            )
                        else:
                            te_token_x = torch.cat([te_token_x, zero_embed], dim=1)
                            te_token_padding_mask = torch.cat(
                                [te_token_padding_mask, zero_padding], dim=1
                            )

                        te_x = te_token_x * self.text_curr_ratio + te_adapter_x * (
                            1 - self.text_curr_ratio
                        )
                        te_x = te_x.transpose(0, 1)
                        te_padding_mask = (
                            te_adapter_padding_mask & te_token_padding_mask
                        )
                elif self.text_curr_ratio <= 0:
                    te_x = te_adapter_x.transpose(0, 1)
                    te_padding_mask = te_adapter_padding_mask
            else:
                te_x = te_token_x.transpose(0, 1)
                te_padding_mask = te_token_padding_mask
        else:
            # textual adapter
            te_x, te_padding_mask = self.te_adapter(
                (jae_x, ctc_logit), jae_padding_mask
            )
            te_x = te_x + self.textual_encoder.embed_positions(
                te_padding_mask
            ).transpose(0, 1)
            te_x = self.dropout_module(te_x)

        if ae_padding_mask is not None:
            ae_x = ae_x * (
                1 - ae_padding_mask.transpose(0, 1).unsqueeze(-1).type_as(ae_x)
            )
        if te_padding_mask is not None:
            te_x = te_x * (
                1 - te_padding_mask.transpose(0, 1).unsqueeze(-1).type_as(te_x)
            )

        senior_acoustic_encoder_idx = -1
        textual_encoder_idx = -1

        while True:
            if self.collaboration_direction == "acoustic":
                for _ in range(self.collaboration_step[1]):
                    textual_encoder_idx += 1
                    if textual_encoder_idx < self.textual_encoder_layer_num:
                        te_x = self.textual_encoder.layers[textual_encoder_idx](
                            te_x,
                            encoder_padding_mask=te_padding_mask,
                        )
                for _ in range(self.collaboration_step[0]):
                    senior_acoustic_encoder_idx += 1
                    if senior_acoustic_encoder_idx < self.senior_acoustic_encoder_layer_num:
                        ae_x = self.senior_acoustic_encoder.layers[
                            senior_acoustic_encoder_idx
                        ](
                            ae_x,
                            encoder_padding_mask=ae_padding_mask,
                            s2=te_x
                            if senior_acoustic_encoder_idx >= self.collaboration_start[0]
                            else None,
                            s2_encoder_padding_mask=te_padding_mask
                            if senior_acoustic_encoder_idx >= self.collaboration_start[0]
                            else None,
                            s2_need_norm=True
                        )
            elif self.collaboration_direction == "textual":
                for _ in range(self.collaboration_step[0]):
                    senior_acoustic_encoder_idx += 1
                    if senior_acoustic_encoder_idx < self.senior_acoustic_encoder_layer_num:
                        ae_x = self.senior_acoustic_encoder.layers[
                            senior_acoustic_encoder_idx
                        ](
                            ae_x,
                            encoder_padding_mask=ae_padding_mask,
                        )
                for _ in range(self.collaboration_step[1]):
                    textual_encoder_idx += 1
                    if textual_encoder_idx < self.textual_encoder_layer_num: 
                        te_x = self.textual_encoder.layers[textual_encoder_idx](
                            te_x,
                            encoder_padding_mask=te_padding_mask,
                            s2=ae_x
                            if textual_encoder_idx >= self.collaboration_start[1]
                            else None,
                            s2_encoder_padding_mask=ae_padding_mask
                            if textual_encoder_idx >= self.collaboration_start[1]
                            else None,
                            s2_need_norm=True
                        )
            elif self.collaboration_direction == "both":
                for _ in range(self.collaboration_step[0]):
                    senior_acoustic_encoder_idx += 1
                    if senior_acoustic_encoder_idx < self.senior_acoustic_encoder_layer_num:
                        ae_x = self.senior_acoustic_encoder.layers[
                            senior_acoustic_encoder_idx
                        ](
                            ae_x,
                            encoder_padding_mask=ae_padding_mask,
                            s2=te_x
                            if senior_acoustic_encoder_idx >= self.collaboration_start[0]
                            else None,
                            s2_encoder_padding_mask=te_padding_mask
                            if senior_acoustic_encoder_idx >= self.collaboration_start[0]
                            else None,
                            s2_need_norm=True
                        )
                for _ in range(self.collaboration_step[1]):
                    textual_encoder_idx += 1
                    if textual_encoder_idx < self.textual_encoder_layer_num:
                        te_x = self.textual_encoder.layers[textual_encoder_idx](
                            te_x,
                            encoder_padding_mask=te_padding_mask,
                            s2=ae_x
                            if textual_encoder_idx >= self.collaboration_start[1]
                            else None,
                            s2_encoder_padding_mask=ae_padding_mask
                            if textual_encoder_idx >= self.collaboration_start[1]
                            else None,
                            s2_need_norm=True,
                        )
            elif self.collaboration_direction == "none":
                for _ in range(self.collaboration_step[0]):
                    senior_acoustic_encoder_idx += 1
                    if senior_acoustic_encoder_idx < self.senior_acoustic_encoder_layer_num:
                        ae_x = self.senior_acoustic_encoder.layers[
                            senior_acoustic_encoder_idx
                        ](
                            ae_x,
                            encoder_padding_mask=ae_padding_mask,
                        )
                for _ in range(self.collaboration_step[1]):
                    textual_encoder_idx += 1
                    if textual_encoder_idx < self.textual_encoder_layer_num:
                        te_x = self.textual_encoder.layers[textual_encoder_idx](
                            te_x,
                            encoder_padding_mask=te_padding_mask,
                        )
            if (
                senior_acoustic_encoder_idx
                >= self.senior_acoustic_encoder_layer_num - 1
                and textual_encoder_idx >= self.textual_encoder_layer_num - 1
            ):
                break

        if ae_x is not None:
            ae_x = self.senior_acoustic_encoder.layer_norm(ae_x)
        if te_x is not None:
            te_x = self.textual_encoder.layer_norm(te_x)

        encoder_out = jae_out
        encoder_out["encoder_out"] = [ae_x] if ae_x is not None else [jae_x]
        encoder_out["encoder_padding_mask"] = [ae_padding_mask] if ae_padding_mask is not None else [jae_padding_mask]
        encoder_out["s2_encoder_out"] = [te_x]
        encoder_out["s2_encoder_padding_mask"] = [te_padding_mask]

        return encoder_out

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["s2_encoder_out"]) == 0:
            new_s2_encoder_out = []
        else:
            new_s2_encoder_out = [
                encoder_out["s2_encoder_out"][0].index_select(1, new_order)
            ]
        if len(encoder_out["s2_encoder_padding_mask"]) == 0:
            new_s2_encoder_padding_mask = []
        else:
            new_s2_encoder_padding_mask = [
                encoder_out["s2_encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "s2_encoder_out": new_s2_encoder_out,  # T x B x C
            "s2_encoder_padding_mask": new_s2_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }


class S2TTransformerS2Encoder(FairseqEncoder):
    def __init__(self, args, dictionary=None, embed_tokens=None):
        super().__init__(None)

        embed_dim = args.encoder_embed_dim
        layer_num = args.encoder_layers
        self.embed_tokens = embed_tokens

        self.embed_scale = math.sqrt(embed_dim)
        if args.encoder_no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = dictionary.pad_index

        self.encoder_embed_norm = getattr(args, "encoder_embed_norm", False)
        if self.encoder_embed_norm:
            self.embed_ln = LayerNorm(embed_dim)

        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )

        self.attn_type = getattr(args, "encoder_attention_type", "selfattn")
        if self.attn_type == "rel_pos":
            self.embed_positions = RelPositionalEncoding(
                args.max_source_positions, args.encoder_embed_dim
            )
        elif self.attn_type in ["rel_selfattn", "rel_pos_legacy"]:
            self.embed_positions = LegacyRelPositionalEncoding(
                args.encoder_embed_dim, args.dropout, args.max_source_positions
            )
        else:  # Use absolute positional embedding
            self.embed_positions = PositionalEmbedding(
                args.max_source_positions, args.encoder_embed_dim, self.padding_idx
            )

        self.layers = nn.ModuleList(
            [S2TTransformerS2EncoderLayer(args) for _ in range(layer_num)]
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

    def forward_input(self, x, encoder_padding_mask):
        if encoder_padding_mask is not None:
            x = x * (1 - encoder_padding_mask.transpose(0, 1).unsqueeze(-1).type_as(x))

        if self.encoder_embed_norm:
            x = self.embed_ln(x)

        # embedding scaling
        x = self.embed_scale * x

        # position embedding
        if self.attn_type in ["rel_pos", "rel_pos_legacy", "rel_selfattn"]:
            positions = self.embed_positions(x)

        elif self.attn_type == "rope":
            positions = None

        else:
            positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
            x += positions
            positions = None

        x = self.dropout_module(x)

        return x


@register_model_architecture(model_name="s2t_multibranch", arch_name="s2t_multibranch")
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
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
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
    args.normalize_embed = getattr(args, "normalize_embed", False)
    args.history_dropout = getattr(args, "history_dropout", 0.0)
    args.history_window_size = getattr(args, "history_window_size", -1)

    # Relative position encoding
    args.max_encoder_relative_length = getattr(args, "max_encoder_relative_length", -1)
    args.k_only = getattr(args, "k_only", True)

    # local modeling
    args.hard_mask_window = getattr(args, "hard_mask_window", 0)
    args.gauss_mask_sigma = getattr(args, "gauss_mask_sigma", 0)
    args.init_mask_weight = getattr(args, "init_mask_weight", 0)

    # interleaved CTC
    args.interleaved_ctc_layers = getattr(args, "interleaved_ctc_layers", None)
    args.interleaved_ctc_temperature = getattr(args, "interleaved_ctc_temperature", 1)
    args.interleaved_ctc_drop_prob = getattr(args, "interleaved_ctc_drop_prob", 0)

    # Semantics-augmented Encoding (sae)
    args.sae_adapter = getattr(args, "sae_adapter", "none")
    args.target_sae_adapter = getattr(args, "target_sae_adapter", args.sae_adapter)
    args.share_sae_and_ctc = getattr(args, "share_sae_and_ctc", False)
    args.share_target_sae_and_ctc = getattr(args, "share_target_sae_and_ctc", False)
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

    args.pds_dropout = getattr(args, "pds_dropout", args.dropout)
    args.pds_fusion = getattr(args, "pds_fusion", False)
    args.pds_fusion_method = getattr(args, "pds_fusion_method", "all_conv")

    # multibranch
    args.junior_acoustic_encoder = getattr(
        args, "junior_acoustic_encoder", "transformer"
    )
    args.senior_acoustic_encoder = getattr(
        args, "senior_acoustic_encoder", "transformer"
    )
    args.textual_encoder = getattr(args, "textual_encoder", "transformer")
    args.textual_encoder_dim = getattr(args, "textual_encoder", args.encoder_embed_dim)

    args.junior_acoustic_encoder_layers = getattr(
        args, "junior_acoustic_encoder_layers", 12
    )
    args.senior_acoustic_encoder_layers = getattr(
        args, "senior_acoustic_encoder_layers", 6
    )
    args.textual_encoder_layers = getattr(args, "textual_encoder_layers", 6)
    args.acoustic_adapter = getattr(args, "acoustic_adapter", "none")
    args.textual_adapter = getattr(args, "textual_adapter", "none")

    args.collaboration_direction = getattr(args, "collaboration_direction", "none")
    args.collaboration_step = getattr(args, "collaboration_step", "1:1")
    args.collaboration_start = getattr(args, "collaboration_start", "0:0")
    args.encoder_collaboration_mode = getattr(
        args, "encoder_collaboration_mode", "serial"
    )
    args.decoder_collaboration_mode = getattr(
        args, "decoder_collaboration_mode", "serial"
    )

    args.encoder_league_s1_ratio = getattr(args, "encoder_league_s1_ratio", 0.5)
    args.encoder_league_s2_ratio = getattr(args, "encoder_league_s2_ratio", 0.5)
    args.encoder_league_out_norm = getattr(args, "encoder_league_out_norm", False)
    args.encoder_league_gated = getattr(args, "encoder_league_gated", False)
    args.encoder_league_drop_net = getattr(args, "encoder_league_drop_net", False)
    args.encoder_league_drop_net_prob = getattr(
        args, "encoder_league_drop_net_prob", 0.0
    )
    args.encoder_league_drop_net_mix = getattr(
        args, "encoder_league_drop_net_mix", False
    )

    args.decoder_league_s1_ratio = getattr(args, "decoder_league_s1_ratio", 0.5)
    args.decoder_league_s2_ratio = getattr(args, "decoder_league_s2_ratio", 0.5)
    args.decoder_league_out_norm = getattr(args, "decoder_league_out_norm", False)
    args.decoder_league_gated = getattr(args, "decoder_league_gated", False)
    args.decoder_league_drop_net = getattr(args, "decoder_league_drop_net", False)
    args.decoder_league_drop_net_prob = getattr(
        args, "decoder_league_drop_net_prob", 0.0
    )
    args.decoder_league_drop_net_mix = getattr(
        args, "decoder_league_drop_net_mix", False)

    # args.encoder_asr_ratio = getattr(args, "encoder_asr_ratio", 1.0)
    # args.encoder_mt_ratio = getattr(args, "encoder_mt_ratio", 1.0)
    # args.encoder_drop_net = getattr(args, "encoder_drop_net", False)
    # args.encoder_drop_net_prob = getattr(args, "encoder_drop_net_prob", 1.0)
    # args.encoder_drop_net_mix = getattr(args, "encoder_drop_net_mix", False)

    args.use_raw_text = getattr(args, "use_raw_text", False)
    args.modality_switch = getattr(args, "modality_switch", False)
    args.text_to_hidden_progress = getattr(args, "text_to_hidden_progress", None)


@register_model_architecture("s2t_multibranch", "s2t_multibranch_s")
def s2t_multibranch_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)


@register_model_architecture("s2t_multibranch", "s2t_multibranch_s_relative")
def s2t_multibranch_s_relative(args):
    args.max_encoder_relative_length = 100
    args.k_only = True
    s2t_multibranch_s(args)


@register_model_architecture("s2t_multibranch", "s2t_multibranch_xs")
def s2t_multibranch_xs(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 4)
    args.dropout = getattr(args, "dropout", 0.3)
    s2t_multibranch_s(args)


@register_model_architecture("s2t_multibranch", "s2t_multibranch_sp")
def s2t_multibranch_sp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_multibranch_s(args)


@register_model_architecture("s2t_multibranch", "s2t_multibranch_m")
def s2t_multibranch_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    base_architecture(args)


@register_model_architecture("s2t_multibranch", "s2t_multibranch_mp")
def s2t_multibranch_mp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_multibranch_m(args)


@register_model_architecture("s2t_multibranch", "s2t_multibranch_l")
def s2t_multibranch_l(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.2)
    base_architecture(args)


@register_model_architecture("s2t_multibranch", "s2t_multibranch_lp")
def s2t_multibranch_lp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_multibranch_l(args)
