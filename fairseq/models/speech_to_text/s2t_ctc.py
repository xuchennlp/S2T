import logging
from typing import Dict, Optional

import torch

from fairseq import checkpoint_utils, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import Embedding
from fairseq.models.speech_to_text import (
    S2TTransformerModel,
    S2TTransformerEncoder,
    PDSS2TTransformerModel,
    PDSS2TTransformerEncoder,
    S2TSATEModel,
    S2TSATEEncoder,
)

from torch import Tensor

logger = logging.getLogger(__name__)


@register_model("s2t_ctc")
class S2TCTCModel(FairseqEncoderModel):

    def __init__(self, encoder):
        super().__init__(encoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        S2TTransformerModel.add_args(parser)
        PDSS2TTransformerModel.add_specific_args(parser)
        S2TSATEModel.add_specific_args(parser)

        # encoder
        parser.add_argument(
            "--encoder-type",
            default="transformer",
            type=str,
            help="encoder type",
        )
        pass

    @classmethod
    def build_encoder(cls, args, task=None):
        # encoder = S2TCTCEncoder(args, task)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        embed_tokens = build_embedding(
            task.target_dictionary, args.encoder_embed_dim
        )

        encoder_type = getattr(args, "encoder_type", "transformer")
        if encoder_type == "transformer":
            encoder = S2TTransformerEncoder(args, task, embed_tokens)
        elif encoder_type == "pds":
            encoder = PDSS2TTransformerEncoder(args, task, embed_tokens)
        elif encoder_type == "sate":
            encoder = S2TSATEEncoder(args, task, embed_tokens)
        else:
            logger.error("Unsupported architecture: %s." % encoder_type)

        if getattr(args, "load_pretrained_encoder_from", None):
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from, strict=False
            )

        return encoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        encoder = cls.build_encoder(args, task)
        if getattr(args, "encoder_freeze_module", None):
            utils.freeze_parameters(encoder, args.encoder_freeze_module)
            logging.info("freeze the encoder module: {}".format(args.encoder_freeze_module))

        return cls(encoder)

    def get_normalized_probs(
            self,
            net_output,
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (T, B, D) tensor
        if isinstance(net_output, list):
            logits = net_output[0]
        else:
            logits = net_output["ctc_logit"][0]
        if log_probs:
            return utils.log_softmax(logits, dim=-1)
        else:
            return utils.softmax(logits, dim=-1)

    def forward(self, src_tokens, src_lengths, prev_output_tokens=None):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """

        if isinstance(src_tokens, list):
            src_lengths = src_tokens[1]
            src_tokens = src_tokens[0]
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)

        return encoder_out


class S2TCTCEncoder(FairseqEncoder):
    """Speech-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args, task=None):
        super().__init__(None)

        # setattr(args, "ctc_weight", 1.0)
        encoder_type = getattr(args, "encoder_type", "transformer")
        if encoder_type == "transformer":
            self.encoder = S2TTransformerEncoder(args, task)
        elif encoder_type == "pds":
            self.encoder = PDSS2TTransformerEncoder(args, task)
        elif encoder_type == "sate":
            self.encoder = S2TSATEEncoder(args, task)
        else:
            logger.error("Unsupported architecture: %s." % encoder_type)

        return

    def dump(self, fstream, info=""):
        self.encoder.dump(fstream, info)

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.update_num = num_updates
        self.encoder.set_num_updates(num_updates)

    def set_flag(self, **kwargs):
        self.encoder.set_flag(**kwargs)

    def set_ctc_infer(self, ctc_infer, post_process, src_dict=None, tgt_dict=None, path=None):
        self.encoder.set_ctc_infer(ctc_infer, post_process, src_dict=src_dict, tgt_dict=tgt_dict, path=path)

    def ctc_valid(self, lprobs, targets, input_lengths, dictionary, lang="source"):
        return self.encoder.ctc_valid(lprobs, targets, input_lengths, dictionary, lang)

    def forward(self, src_tokens, src_lengths, **kwargs):

        return self.encoder(src_tokens, src_lengths, **kwargs)

    def reorder_encoder_out(self, encoder_out, new_order):
        return self.encoder.reorder_encoder_out(encoder_out, new_order)


class CTCDecoder(object):

    def __init__(self, models, args, dictionary, blank_idx):
        self.dict = dictionary
        self.vocab_size = len(dictionary)

        self.blank = blank_idx
        self.pad = dictionary.pad()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()

        self.ctc_self_ensemble = getattr(args, "ctc_self_ensemble", False)
        self.ctc_inter_logit = getattr(args, "ctc_inter_logit", 0)
        assert not (self.ctc_self_ensemble is True and self.ctc_inter_logit is True), \
            "Self ensemble and inference by intermediate logit can not be True at the same time."

        if self.ctc_self_ensemble:
            logger.info("Using self ensemble for CTC inference")
        if self.ctc_inter_logit != 0:
            logger.info("Using intermediate logit %d for CTC inference" % self.ctc_inter_logit)

        self.vocab_size = len(dictionary)
        self.beam_size = args.beam
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(self.beam_size, self.vocab_size - 1)

        from fairseq.sequence_generator import EnsembleModel
        if isinstance(models, EnsembleModel):
            self.model = models
        else:
            self.model = EnsembleModel(models)
        self.model = models[0]
        self.model.eval()

        self.lm_model = getattr(args, "kenlm_model", None)
        self.lm_weight = getattr(args, "lm_weight", 0)
        # if self.lm_model is not None:
            # self.lm_model.eval()

        self.infer = "greedy"
        if self.beam_size > 0:
            try:
                from ctcdecode import CTCBeamDecoder
                self.infer = "beam"
                self.ctc_decoder = CTCBeamDecoder(
                    # dictionary.symbols,
                    [chr(idx + 100) for idx in range(len(dictionary.symbols))],
                    model_path=self.lm_model,
                    alpha=self.lm_weight,
                    beta=1 if self.lm_weight > 0 else 0,
                    cutoff_top_n=40,
                    cutoff_prob=1.0,
                    beam_width=self.beam_size,
                    num_processes=80,
                    blank_id=self.blank,
                    log_probs_input=False
                )
            except ImportError:
                logger.warning("Cannot import the CTCBeamDecoder library. We use the greedy search for CTC decoding.")
        
        self.cal_flops = getattr(args, "cal_flops", False)

    def generate(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs):

        net_input = sample["net_input"]

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
        src_tokens = net_input["src_tokens"]
        src_lengths = net_input["src_lengths"]
        bsz, src_len = src_tokens.size()[:2]

        if self.cal_flops:
            from thop import profile
            macs, encoder_outs = profile(self.model, inputs=(net_input.values()))
            gmacs = macs / 1e9
            logger.info("GMACs: %f. GFLOPs: %f" % (gmacs, gmacs * 2))
            print("GMACs: %f. GFLOPs: %f" % (gmacs, gmacs * 2))


            from torchprofile import profile_macs
            macs = profile_macs(self.model, [src_tokens, src_lengths])
            gmacs = macs / 1e9
            logger.info("GMACs: %f. GFLOPs: %f" % (gmacs, gmacs * 2))
            print("GMACs: %f. GFLOPs: %f" % (gmacs, gmacs * 2))

            exit()
        encoder_outs = self.model(src_tokens=src_tokens,
                                    src_lengths=src_lengths)

        ctc_logit = None
        inter_logits = []
        if "xctc_logit" in encoder_outs:
            if len(encoder_outs["xctc_logit"]) > 0:
                ctc_logit = encoder_outs["xctc_logit"][0].transpose(0, 1)
            inter_logits = encoder_outs.get("inter_xctc_logits", [])
        if ctc_logit is None:
            ctc_logit = encoder_outs["ctc_logit"][0].transpose(0, 1)
        if len(inter_logits) > 0:
            inter_logits = encoder_outs.get("inter_ctc_logits", [])
        inter_logits_num = len(inter_logits)

        encoder_padding_mask = encoder_outs["encoder_padding_mask"][0]

        if self.ctc_inter_logit != 0:
            if inter_logits_num != 0:
                assert self.ctc_inter_logit <= inter_logits_num
                ctc_logit_item = inter_logits[-self.ctc_inter_logit]
                if isinstance(ctc_logit_item, list):
                    ctc_logit = ctc_logit_item[0].transpose(0, 1)
                    if len(ctc_logit_item) >= 2:
                        encoder_padding_mask = ctc_logit_item[1]
                
        logit_length = (~encoder_padding_mask).long().sum(-1)
        finalized = []
        if self.infer == "beam":
            beam_results, beam_scores, time_steps, out_lens = self.ctc_decoder.decode(
                utils.softmax(ctc_logit, -1), logit_length
            )

            for idx in range(bsz):
                hypos = []
                #for beam_idx in range(beam_size):
                for beam_idx in range(1):
                    hypo = dict()
                    length = out_lens[idx][beam_idx]
                    scores = beam_scores[idx, beam_idx]

                    hypo["tokens"] = beam_results[idx, beam_idx, : length]
                    hypo["score"] = scores
                    hypo["attention"] = None
                    hypo["alignment"] = None
                    hypo["positional_scores"] = torch.Tensor([scores / length] * length)
                    hypos.append(hypo)
                finalized.append(hypos)

        # elif self.infer == "greedy":
        else:
            ctc_probs = utils.log_softmax(ctc_logit, -1)
            if self.ctc_self_ensemble:
                if inter_logits_num != 0:
                    for i in range(inter_logits_num):
                        if isinstance(inter_logits[i], list):
                            logit = inter_logits[i][0]
                        else:
                            logit = inter_logits[i]

                        inter_logits_prob = utils.log_softmax(logits.transpose(0, 1), -1)
                        ctc_probs += inter_logits_prob

            topk_prob, topk_index = ctc_probs.topk(1, dim=2)

            topk_prob = topk_prob.squeeze(-1)
            topk_index = topk_index.squeeze(-1)

            real_indexs = topk_index.masked_fill(encoder_padding_mask, self.blank).cpu()
            real_probs = topk_prob.masked_fill(topk_index == self.blank, self.blank)
            scores = -real_probs.sum(-1, keepdim=True).cpu()

            for idx in range(bsz):
                hypos = []
                hypo = dict()

                hyp = real_indexs[idx].unique_consecutive()
                hyp = hyp[hyp != self.blank]
                length = len(hyp)

                hypo["tokens"] = hyp
                hypo["score"] = scores[idx]
                hypo["attention"] = None
                hypo["alignment"] = None
                hypo["positional_scores"] = torch.Tensor([hypo["score"] / length] * length)
                hypos.append(hypo)
                finalized.append(hypos)

        return finalized


@register_model_architecture(model_name="s2t_ctc", arch_name="s2t_ctc")
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

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)

    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.encoder_no_scale_embedding = getattr(args, "encoder_no_scale_embedding", False)

    args.encoder_embed_linear = getattr(args, "encoder_embed_linear", False)
    args.encoder_embed_norm = getattr(args, "encoder_embed_norm", False)

    # CTC
    args.ctc_layer = getattr(args, "ctc_layer", 0)
    args.xctc_layer = getattr(args, "xctc_layer", 0)
    args.share_ctc_and_embed = getattr(args, "share_ctc_and_embed", False)
    args.share_xctc_and_embed = getattr(args, "share_xctc_and_embed", False)

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

    # Prediction-aware encoding (pae)
    args.ctc_pae = getattr(args, "ctc_pae", "none")
    args.share_pae_and_ctc = getattr(args, "share_pae_and_ctc", False)
    args.pae_embed_norm = getattr(args, "pae_embed_norm", False)
    args.pae_out_norm = getattr(args, "pae_out_norm", False)
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
    args.pds_dropout = getattr(args, "pds_dropout", args.dropout)

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


@register_model_architecture("s2t_ctc", "s2t_ctc_s")
def s2t_ctc_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)


@register_model_architecture("s2t_ctc", "s2t_ctc_s_relative")
def s2t_ctc_s_relative(args):
    args.max_encoder_relative_length = 100
    args.k_only = True
    s2t_ctc_s(args)


@register_model_architecture("s2t_ctc", "s2t_ctc_xs")
def s2t_ctc_xs(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 4)
    args.dropout = getattr(args, "dropout", 0.3)
    s2t_ctc_s(args)


@register_model_architecture("s2t_ctc", "s2t_ctc_sp")
def s2t_ctc_sp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_ctc_s(args)


@register_model_architecture("s2t_ctc", "s2t_ctc_m")
def s2t_ctc_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    base_architecture(args)


@register_model_architecture("s2t_ctc", "s2t_ctc_mp")
def s2t_ctc_mp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_ctc_m(args)


@register_model_architecture("s2t_ctc", "s2t_ctc_l")
def s2t_ctc_l(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.2)
    base_architecture(args)


@register_model_architecture("s2t_ctc", "s2t_ctc_lp")
def s2t_ctc_lp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_ctc_l(args)
