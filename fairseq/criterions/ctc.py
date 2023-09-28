# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II
from typing import Optional
import numpy as np
import logging

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask
from fairseq.logging.meters import safe_round

logger = logging.getLogger(__name__)

try:
    from fairseq.torch_imputer import best_alignment, imputer_loss
except:
    # logger.error("Imputer is not available.")
    pass


@dataclass
class CtcCriterionConfig(FairseqDataclass):
    zero_infinity: bool = field(
        default=True,
        metadata={"help": "zero inf loss when source length <= target length"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    post_process: str = field(
        default="sentencepiece",
        metadata={
            "help": "how to post process predictions into words. can be letter, "
            "wordpiece, BPE symbols, etc. "
            "See fairseq.data.data_utils.post_process() for full list of options"
        },
    )
    ctc_weight: float = field(
        default=0.0,
        metadata={"help": "weight of CTC loss"},
    )
    ctc_entropy_weight: float = field(
        default=0.0,
        metadata={"help": "weight of CTC entropy"},
    )
    ctc_entropy_cutoff: int = field(
        default=0,
        metadata={"help": "cutoff for CTC entropy computation"},
    )
    ctc_blank_entropy: bool = field(
        default=False,
        metadata={"help": "minimize the blank entropy in CTC"},
    )
    inter_ctc_weight: float = field(
        default=0.0,
        metadata={"help": "weight of intermediate CTC loss"},
    )
    inter_ctc_mlo: str = field(
        default="",
        metadata={"help": "the objective order to calculate intermediate CTC loss"},
    )

    xctc_weight: float = field(
        default=0.0,
        metadata={"help": "weight of CTC loss for target sentence"},
    )
    inter_xctc_weight: float = field(
        default=0.0,
        metadata={"help": "weight of intermediate CTC loss for target sentence"},
    )
    axctc_weight: float = field(
        default=0.0,
        metadata={"help": "weight of CTC loss for aligned target sentence"},
    )
    inter_axctc_weight: float = field(
        default=0.0,
        metadata={
            "help": "weight of intermediate CTC loss for aligned target sentence"
        },
    )
    ctc_masked_loss: bool = field(
        default=False,
        metadata={"help": "calculate masked ctc loss"},
    )

    cal_all_ctc: bool = field(
        default=False,
        metadata={"help": "calculate all ctc results"},
    )

    ctc_self_distill_weight: float = field(
        default=0.0,
        metadata={"help": "weight of the self distillation CTC loss"},
    )
    xctc_self_distill_weight: float = field(
        default=0.0,
        metadata={
            "help": "weight of the self distillation CTC loss for target sentence"
        },
    )
    ctc_self_distill_prob: float = field(
        default=0.1,
        metadata={"help": "probability to use distillation loss"},
    )
    ctc_self_distill_temperature: float = field(
        default=1,
        metadata={"help": "temperature for ctc self distillation"},
    )
    ctc_mixup_consistent_weight: float = field(
        default=0,
        metadata={"help": "consistent regularization for CTC loss in mixup"},
    )
    inter_ctc_mixup_consistent_weight: float = field(
        default=0,
        metadata={"help": "consistent regularization for inter CTC loss in mixup"},
    )

    wer_kenlm_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "if this is provided, use kenlm to compute wer (along with other wer_* args)"
        },
    )
    wer_lexicon: Optional[str] = field(
        default=None,
        metadata={"help": "lexicon to use with wer_kenlm_model"},
    )
    wer_lm_weight: float = field(
        default=2.0,
        metadata={"help": "lm weight to use with wer_kenlm_model"},
    )
    wer_word_score: float = field(
        default=-1.0,
        metadata={"help": "lm word score to use with wer_kenlm_model"},
    )

    wer_args: Optional[str] = field(
        default=None,
        metadata={
            "help": "DEPRECATED: tuple of (wer_kenlm_model, wer_lexicon, wer_lm_weight, wer_word_score)"
        },
    )


@register_criterion("ctc", dataclass=CtcCriterionConfig)
class CtcCriterion(FairseqCriterion):
    def __init__(
        self, cfg: CtcCriterionConfig, task: FairseqTask, ctc_weight=1.0, save_dir=None
    ):
        super().__init__(task)

        if cfg.wer_args is not None:
            (
                cfg.wer_kenlm_model,
                cfg.wer_lexicon,
                cfg.wer_lm_weight,
                cfg.wer_word_score,
            ) = eval(cfg.wer_args)

        if cfg.wer_kenlm_model is not None:
            from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

            dec_args = Namespace()
            dec_args.nbest = 1
            dec_args.criterion = "ctc"
            dec_args.kenlm_model = cfg.wer_kenlm_model
            dec_args.lexicon = cfg.wer_lexicon
            dec_args.beam = 50
            dec_args.beam_size_token = min(50, len(task.target_dictionary))
            dec_args.beam_threshold = min(50, len(task.target_dictionary))
            dec_args.lm_weight = cfg.wer_lm_weight
            dec_args.word_score = cfg.wer_word_score
            dec_args.unk_weight = -math.inf
            dec_args.sil_weight = 0

            self.w2l_decoder = W2lKenLMDecoder(dec_args, task.target_dictionary)
        else:
            self.w2l_decoder = None

        self.blank_idx = (
            task.target_dictionary.index(task.blank_symbol)
            if hasattr(task, "blank_symbol")
            else 0
        )
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.post_process = cfg.post_process
        self.sentence_avg = cfg.sentence_avg
        self.save_dir = save_dir

        self.cal_all_ctc = cfg.cal_all_ctc
        self.ctc_weight = ctc_weight
        self.inter_ctc_weight = cfg.inter_ctc_weight
        inter_ctc_mlo = cfg.inter_ctc_mlo
        if inter_ctc_mlo is not None and inter_ctc_mlo != "":
            self.inter_ctc_mlo = inter_ctc_mlo.split(":")
        else:
            self.inter_ctc_mlo = None
        self.xctc_weight = cfg.xctc_weight
        self.inter_xctc_weight = cfg.inter_xctc_weight
        self.axctc_weight = cfg.axctc_weight
        self.inter_axctc_weight = cfg.inter_axctc_weight
        self.ctc_masked_loss = cfg.ctc_masked_loss

        self.ctc_self_distill_weight = cfg.ctc_self_distill_weight
        self.xctc_self_distill_weight = float(cfg.xctc_self_distill_weight)
        self.ctc_self_distill_prob = float(cfg.ctc_self_distill_prob)
        self.ctc_self_distill_temperature = float(cfg.ctc_self_distill_temperature)

        self.ctc_entropy_weight = cfg.ctc_entropy_weight
        self.ctc_entropy_cutoff = cfg.ctc_entropy_cutoff
        self.ctc_blank_entropy = cfg.ctc_blank_entropy

        self.ctc_mixup_consistent_weight = cfg.ctc_mixup_consistent_weight
        self.inter_ctc_mixup_consistent_weight = cfg.inter_ctc_mixup_consistent_weight

        self.all_ctc_weight = (
            self.ctc_weight
            + self.inter_ctc_weight
            + self.xctc_weight
            + self.inter_xctc_weight
            + self.axctc_weight
            + self.inter_axctc_weight
            + self.ctc_self_distill_weight
            + self.xctc_self_distill_weight
            + self.ctc_entropy_weight
            + self.ctc_mixup_consistent_weight
            + self.inter_ctc_mixup_consistent_weight
        )

        if self.all_ctc_weight > 0:
            self.ctc_loss = torch.nn.CTCLoss(
                blank=self.blank_idx, reduction="none", zero_infinity=True
            )

        self.ctc_names = []
        self.use_ctc = self.ctc_weight + self.inter_ctc_weight > 0
        self.use_xctc = self.xctc_weight + self.inter_xctc_weight > 0
        self.use_axctc = self.axctc_weight + self.inter_axctc_weight > 0
        self.use_source_distill = self.use_target_distill = False
        if self.ctc_self_distill_prob > 0:
            if self.ctc_self_distill_weight:
                self.use_source_distill = True
            if self.xctc_self_distill_weight > 0:
                self.use_target_distill = True

    def forward(self, model, sample, reduce=True):
        # net_output = model(**sample["net_input"])
        src_tokens, src_lengths, prev_output_tokens = sample["net_input"].values()

        if self.training and getattr(model.encoder, "pae_ground_truth_ratio", 0) != 0:
            ctc_alignment_oracle = self.get_ground_truth_alignment(model, sample)
            net_output = model.encoder(src_tokens, src_lengths,
                                        ctc_alignment_oracle=ctc_alignment_oracle)
        else:
            net_output = model.encoder(src_tokens=src_tokens, src_lengths=src_lengths)

        ntokens = sample["ntokens"]
        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens

        logging_output = {
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        loss, logging_output = self.compute_ctc_loss(
            model, sample, net_output, logging_output
        )
        return loss, sample_size, logging_output

    def get_ground_truth_alignment(self, model, sample):
        ctc_alignment_oracle = dict()

        def get_ctc_align(
            logit, tokens, input_lengths, target_lengths, pad_idx, blank_idx
        ):
            logit = torch.log_softmax(logit, dim=-1, dtype=torch.float32)
            best_aligns = best_alignment(
                logit,
                tokens,
                input_lengths,
                target_lengths,
                blank_idx,
                zero_infinity=True,
            )
            best_aligns_pad = torch.tensor(
                [a + [0] * (logit.size(0) - len(a)) for a in best_aligns],
                device=logit.device,
                dtype=tokens.dtype,
            )
            oracle_pos = torch.div(best_aligns_pad, 2, rounding_mode="floor").clip(
                max=tokens.shape[1] - 1
            )
            oracle = tokens.gather(-1, oracle_pos)
            oracle.masked_fill_(best_aligns_pad % 2 == 0, blank_idx)
            # logger.info((logit.argmax(dim=-1).transpose(0, 1) != oracle))
            mistake_flag = logit.argmax(dim=-1).transpose(0, 1) != oracle
            mistake_num = mistake_flag.sum(-1)
            mistake_ratio = mistake_num / input_lengths 

            def get_symbol(state, targets_list):
                if state % 2 == 0:
                    symbol = 0

                else:
                    symbol = targets_list[state // 2]

                return symbol

            # align_probs = []
            # logit = logit.transpose(0, 1)

            # for l_p, best_a, toks in zip(logit.to("cpu"), best_aligns, tokens.tolist()):
            #     align_p = []

            #     for p, a in zip(l_p, best_a):
            #         align_p.append(p[get_symbol(a, toks)].item())

            #     align_probs.append(align_p)

            # for model_align, mel_l, b_align, b_p, toks in zip(
            #     logit, input_lengths, oracle.tolist(), align_probs, tokens.tolist()
            # ):
            #     model_p, model_align = model_align.max(1)
            #     model_p = model_p[:mel_l].sum().item()
            #     model_align = model_align[:mel_l].tolist()
            #     b_p = sum(b_p)

            #     print(
            #             f"model: {model_align} ({model_p:.3f})\nbest: {b_align} ({b_p:.3f})"
            #         )

            return oracle, best_aligns_pad, mistake_flag, mistake_ratio

        src_tokens, src_lengths, prev_output_tokens = sample["net_input"].values()
        with torch.no_grad():
            encoder_out = model.encoder(src_tokens, src_lengths)

            ctc_logit = None
            if "ctc_logit" in encoder_out and len(encoder_out["ctc_logit"]) != 0:
                ctc_logit = encoder_out["ctc_logit"][0]
            elif (
                "inter_ctc_logits" in encoder_out
                and len(encoder_out["inter_ctc_logits"]) != 0
            ):
                ctc_logit = encoder_out["inter_ctc_logits"][-1]

            if ctc_logit is not None:
                if "transcript" in sample:
                    tokens = sample["transcript"]["tokens"]
                else:
                    tokens = sample["target"]
                pad_mask = (tokens != self.pad_idx) & (tokens != self.eos_idx)
                target_lengths = pad_mask.sum(-1)

                if "ctc_padding_mask" in encoder_out:
                    non_padding_mask = ~encoder_out["ctc_padding_mask"][0]
                else:
                    non_padding_mask = ~encoder_out["encoder_padding_mask"][0]
                input_lengths = non_padding_mask.long().sum(-1)

                ctc_alignment_oracle["ctc"] = get_ctc_align(
                    ctc_logit,
                    tokens,
                    input_lengths,
                    target_lengths,
                    self.pad_idx,
                    self.blank_idx,
                )

            xctc_logit = None
            if "xctc_logit" in encoder_out and len(encoder_out["xctc_logit"]) != 0:
                xctc_logit = encoder_out["xctc_logit"][0]
            elif (
                "inter_xctc_logits" in encoder_out
                and len(encoder_out["inter_xctc_logits"]) != 0
            ):
                xctc_logit = encoder_out["inter_xctc_logits"][-1]

            if xctc_logit is not None:
                if "ctc_padding_mask" in encoder_out:
                    non_padding_mask = ~encoder_out["ctc_padding_mask"][0]
                else:
                    non_padding_mask = ~encoder_out["encoder_padding_mask"][0]
                input_lengths = non_padding_mask.long().sum(-1)

                tokens = self.get_ctc_target_text(sample)
                target_pad_mask = (tokens != self.pad_idx) & (tokens != self.eos_idx)
                target_lengths = target_pad_mask.sum(-1)

                ctc_alignment_oracle["xctc"] = get_ctc_align(
                    xctc_logit,
                    tokens,
                    input_lengths,
                    target_lengths,
                    self.pad_idx,
                    self.blank_idx,
                )

            axctc_logit = None
            if "axctc_logit" in encoder_out and len(encoder_out["axctc_logit"]) != 0:
                axctc_logit = encoder_out["axctc_logit"][0]
            elif (
                "inter_axctc_logits" in encoder_out
                and len(encoder_out["inter_axctc_logits"]) != 0
            ):
                axctc_logit = encoder_out["inter_axctc_logits"][-1]
            if axctc_logit is not None:
                tokens = self.get_aligned_target_text(sample)
                target_pad_mask = (tokens != self.pad_idx) & (tokens != self.eos_idx)
                target_lengths = target_pad_mask.sum(-1)
                ctc_alignment_oracle["axctc"] = get_ctc_align(
                    axctc_logit,
                    tokens,
                    input_lengths,
                    target_lengths,
                    self.pad_idx,
                    self.blank_idx,
                )

        return ctc_alignment_oracle

    def get_ctc_loss(
        self,
        model,
        ctc_logit,
        targets,
        input_lengths,
        target_lengths,
        loss_coef,
        force_emit=None,
    ):
        lprobs = model.get_normalized_probs(
            [ctc_logit], log_probs=True
        ).contiguous()  # (T, B, C) from the encoder
        lprobs.batch_first = False

        loss = 0
        with torch.backends.cudnn.flags(enabled=False):
            for item_targets, item_target_lengths, item_coef in zip(
                targets, target_lengths, loss_coef
            ):
                if force_emit is not None and self.ctc_masked_loss:
                    item_loss = imputer_loss(
                        lprobs,
                        item_targets,
                        force_emit,
                        input_lengths,
                        item_target_lengths,
                        blank=self.blank_idx,
                        reduction="none",
                        zero_infinity=True,
                    )
                else:
                    item_loss = self.ctc_loss(
                    lprobs,
                    item_targets,
                    input_lengths,
                    item_target_lengths,
                )
                loss += (item_loss * item_coef).sum()
        return loss, lprobs

    @staticmethod
    def get_ctc_self_distill_loss(
        distill_num, teacher_logit, student_logits, non_padding_mask, temperature=1.0
    ):

        ctc_self_distill_losses = []
        for i in range(distill_num):
            logit = student_logits[i]
            if type(logit) == list:
                student_logit = logit[0]
                non_padding_mask = ~logit[1]
            else:
                student_logit = logit

            if student_logit.size() != teacher_logit.size():
                continue

            loss = F.kl_div(
                F.log_softmax(student_logit / temperature, dim=-1, dtype=torch.float32),
                # F.log_softmax(teacher_logit / temperature, dim=-1, dtype=torch.float32),
                F.log_softmax(
                    teacher_logit.detach() / temperature, dim=-1, dtype=torch.float32
                ),
                log_target=True,
                reduction="none",
            )
            loss = loss.sum(-1).transpose(0, 1).masked_fill(~non_padding_mask, 0.0).sum()
            ctc_self_distill_losses.append(loss)
        return ctc_self_distill_losses

    def get_ctc_target_text(self, sample):
        if "ctc_target" in sample:
            return sample["ctc_target"]["tokens"]
        return sample["target"]

    def get_aligned_target_text(self, sample):
        if "aligned_target" in sample:
            return sample["aligned_target"]["tokens"]
        return sample["target"]

    def get_targets_for_ctc_loss(self, target_tokens, net_output):
        target_pad_mask = (target_tokens != self.pad_idx) & (
            target_tokens != self.eos_idx
        )

        if "mixup" in net_output and net_output["mixup"] is not None:
            mixup_coef = net_output["mixup"]["coef"]
            mixup_idx1 = net_output["mixup"]["index1"]
            mixup_idx2 = net_output["mixup"]["index2"]

            mask1 = target_pad_mask[mixup_idx1]
            mask2 = target_pad_mask[mixup_idx2]
            target_tokens1 = target_tokens[[mixup_idx1]].masked_select(mask1)
            target_tokens2 = target_tokens[[mixup_idx2]].masked_select(mask2)
            target_lengths1 = mask1.sum(-1)
            target_lengths2 = mask2.sum(-1)
            target_tokens = [target_tokens1, target_tokens2]
            target_lengths = [target_lengths1, target_lengths2]
            loss_coef = [mixup_coef, 1 - mixup_coef]
        else:
            target_tokens = [target_tokens.masked_select(target_pad_mask)]
            target_lengths = [target_pad_mask.sum(-1)]
            loss_coef = [1]

        return target_tokens, target_lengths, loss_coef

    def compute_ctc_loss(self, model, sample, net_output, logging_output):
        if "transcript" in sample:
            tokens = sample["transcript"]["tokens"]
        else:
            tokens = sample["target"]
        if "ctc_padding_mask" in net_output:
            non_padding_mask = ~net_output["ctc_padding_mask"][0]
        else:
            non_padding_mask = ~net_output["encoder_padding_mask"][0]

        mixup = False
        if "mixup" in net_output and net_output["mixup"] is not None:
            mixup = True

        input_lengths = non_padding_mask.long().sum(-1)
        nfeatures = input_lengths.sum().item()
        logging_output["nfeatures"] = nfeatures

        transcripts, transcript_lengths, loss_coef = self.get_targets_for_ctc_loss(tokens, net_output)

        all_ctc_logits = dict()
        self.ctc_names = []
        lprobs = None
        target_lprobs = None
        ctc_entropy = []

        inter_ctc_num = 0
        inter_ctc_loss = 0
        if "inter_ctc_logits" in net_output:
            inter_ctc_num = len(net_output["inter_ctc_logits"])

        # calculate the inter CTC loss
        if self.inter_ctc_weight > 0 and inter_ctc_num > 0:
            logits = net_output["inter_ctc_logits"]
            for i in range(inter_ctc_num):
                inter_transcripts, inter_transcript_lengths, inter_loss_coef = transcripts, transcript_lengths, loss_coef
                if self.inter_ctc_mlo is not None:
                    order = self.inter_ctc_mlo[i]
                    tokens_key = "transcript%s" % order
                    if sample.get(tokens_key, None):
                        inter_tokens = sample[tokens_key]["tokens"]
                        inter_transcripts, inter_transcript_lengths, inter_loss_coef = self.get_targets_for_ctc_loss(inter_tokens, net_output)
                
                logit = logits[i]
                force_emit = None
                if type(logit) == list:
                    inter_ctc_logit = logit[0]
                    inter_input_lengths = (
                        (~logit[1]).long().sum(-1)
                        if logit[1] is not None
                        else input_lengths
                    )
                    if len(logit) >= 3:
                        force_emit = logit[2]
                else:
                    inter_ctc_logit = logit
                    inter_input_lengths = input_lengths

                if self.ctc_entropy_weight > 0:
                    if self.ctc_entropy_cutoff != 0:
                        cut_ctc_logit = inter_ctc_logit.sort(dim=-1, descending=True)[
                            0
                        ][:, :, 0 : self.ctc_entropy_cutoff]
                        # cut_ctc_logit = cut_ctc_logit / cut_ctc_logit.sum(dim=-1, keepdim=True)
                        ctc_entropy.append(
                            Categorical(logits=cut_ctc_logit).entropy().sum()
                        )
                    if self.ctc_blank_entropy:
                        blank_prob = F.softmax(
                            inter_ctc_logit, dim=-1, dtype=torch.float32
                        )[:, :, self.blank_idx]
                        entropy = -blank_prob * blank_prob.log()
                        ctc_entropy.append(entropy.sum())
                    else:
                        ctc_entropy.append(
                            Categorical(logits=inter_ctc_logit).entropy().sum()
                        )

                inter_loss, inter_lprobs = self.get_ctc_loss(
                    model,
                    inter_ctc_logit,
                    inter_transcripts,
                    inter_input_lengths,
                    inter_transcript_lengths,
                    inter_loss_coef,
                    force_emit,
                )
                inter_ctc_loss += inter_loss
                lprobs = inter_lprobs

            inter_ctc_loss /= inter_ctc_num
            logging_output["inter_ctc_loss"] = utils.item(inter_ctc_loss.data)

        ctc_loss = 0
        use_ctc = False
        if (
            self.ctc_weight > 0
            and "ctc_logit" in net_output
            and len(net_output["ctc_logit"]) > 0
        ):
            use_ctc = True
            logit = net_output["ctc_logit"][0]
            # all_ctc_logits["ctc_logit"] = [ctc_logit, input_lengths]

            force_emit = None
            if type(logit) == list:
                ctc_logit = logit[0]
                input_lengths = (
                    (~logit[1]).long().sum(-1)
                    if logit[1] is not None
                    else input_lengths
                )
                if len(logit) >= 3:
                    force_emit = logit[2]
            else:
                ctc_logit = logit

            ctc_loss, lprobs = self.get_ctc_loss(
                model,
                ctc_logit,
                transcripts,
                input_lengths,
                transcript_lengths,
                loss_coef,
                force_emit,
            )

            if self.ctc_entropy_weight > 0:
                if self.ctc_entropy_cutoff != 0:
                    cut_ctc_logit = ctc_logit.sort(dim=-1, descending=True)[0][
                        :, :, 0 : self.ctc_entropy_cutoff
                    ]
                    # cut_ctc_logit = cut_ctc_logit / cut_ctc_logit.sum(dim=-1, keepdim=True)
                    ctc_entropy.append(
                        Categorical(logits=cut_ctc_logit).entropy().sum()
                    )
                else:
                    ctc_entropy.append(Categorical(logits=ctc_logit).entropy().sum())

            logging_output["ctc_loss"] = utils.item(ctc_loss.data)

        # calculate the target CTC loss
        axctc_loss = 0
        inter_axctc_loss = 0
        inter_axctc_num = 0

        if self.use_axctc:
            aligned_target_tokens = self.get_aligned_target_text(sample)
            target_tokens, target_lengths, loss_coef = self.get_targets_for_ctc_loss(
                aligned_target_tokens, net_output
            )

            if "inter_axctc_logits" in net_output:
                inter_axctc_num = len(net_output["inter_axctc_logits"])
            if inter_axctc_num != 0 and self.inter_axctc_weight > 0:
                logits = net_output["inter_axctc_logits"]
                for i in range(inter_axctc_num):
                    logit = logits[i]
                    force_emit = None
                    if type(logit) == list:
                        inter_axctc_logit = logit[0]
                        inter_input_lengths = (
                            (~logit[1]).long().sum(-1)
                            if logit[1] is not None
                            else input_lengths
                        )
                        if len(logit) >= 3:
                            force_emit = logit[2]
                    else:
                        inter_axctc_logit = logit
                        inter_input_lengths = input_lengths

                    # all_ctc_logits["inter_axctc_logit%d" % i] = [inter_axctc_logit, inter_input_lengths]
                    inter_loss, target_inter_lprobs = self.get_ctc_loss(
                        model,
                        inter_axctc_logit,
                        target_tokens,
                        inter_input_lengths,
                        target_lengths,
                        loss_coef,
                        force_emit,
                    )
                    inter_axctc_loss += inter_loss
                    target_lprobs = target_inter_lprobs

                inter_axctc_loss /= inter_axctc_num
                logging_output["inter_axctc_loss"] = utils.item(inter_axctc_loss.data)

            if self.axctc_weight > 0:
                assert "axctc_logit" in net_output
                logit = net_output["axctc_logit"][0]
                # all_ctc_logits["axctc_logit"] = [axctc_logit, input_lengths]

                force_emit = None
                if type(logit) == list:
                    axctc_logit = logit[0]
                    input_lengths = (
                        (~logit[1]).long().sum(-1)
                        if logit[1] is not None
                        else input_lengths
                    )
                    if len(logit) >= 3:
                        force_emit = logit[2]
                else:
                    axctc_logit = logit

                axctc_loss, target_lprobs = self.get_ctc_loss(
                    model,
                    axctc_logit,
                    target_tokens,
                    input_lengths,
                    target_lengths,
                    loss_coef,
                    force_emit,
                )
                logging_output["axctc_loss"] = utils.item(axctc_loss.data)

        xctc_loss = 0
        inter_xctc_loss = 0
        inter_xctc_num = 0

        if self.use_xctc:
            ctc_target_tokens = self.get_ctc_target_text(sample)
            target_tokens, target_lengths, loss_coef = self.get_targets_for_ctc_loss(
                ctc_target_tokens, net_output
            )

            if "inter_xctc_logits" in net_output:
                inter_xctc_num = len(net_output["inter_xctc_logits"])
            if inter_xctc_num != 0 and self.inter_xctc_weight > 0:
                logits = net_output["inter_xctc_logits"]
                for i in range(inter_xctc_num):
                    logit = logits[i]

                    force_emit = None
                    if type(logit) == list:
                        inter_xctc_logit = logit[0]
                        inter_input_lengths = (
                            (~logit[1]).long().sum(-1)
                            if logit[1] is not None
                            else input_lengths
                        )
                        if len(logit) >= 3:
                            force_emit = logit[2]
                    else:
                        inter_xctc_logit = logit
                        inter_input_lengths = input_lengths

                    # all_ctc_logits["inter_xctc_logit%d" % i] = [inter_xctc_logit, inter_input_lengths]
                    inter_loss, target_inter_lprobs = self.get_ctc_loss(
                        model,
                        inter_xctc_logit,
                        target_tokens,
                        inter_input_lengths,
                        target_lengths,
                        loss_coef,
                        force_emit,
                    )
                    inter_xctc_loss += inter_loss
                    target_lprobs = target_inter_lprobs

                inter_xctc_loss /= inter_xctc_num
                logging_output["inter_xctc_loss"] = utils.item(inter_xctc_loss.data)

            if self.xctc_weight > 0:
                assert "xctc_logit" in net_output
                logit = net_output["xctc_logit"][0]

                force_emit = None
                if type(logit) == list:
                    xctc_logit = logit[0]
                    input_lengths = (
                        (~logit[1]).long().sum(-1)
                        if logit[1] is not None
                        else input_lengths
                    )
                    if len(logit) >= 3:
                        force_emit = logit[2]
                else:
                    xctc_logit = logit
                # all_ctc_logits["xctc_logit"] = [xctc_logit, input_lengths]

                xctc_loss, target_lprobs = self.get_ctc_loss(
                    model,
                    xctc_logit,
                    target_tokens,
                    input_lengths,
                    target_lengths,
                    loss_coef,
                    force_emit,
                )
                logging_output["xctc_loss"] = utils.item(xctc_loss.data)

        # calculate the self distillation CTC loss
        ctc_self_distill_loss = 0
        if self.use_source_distill or self.use_target_distill:
            ctc_self_distill_choice = torch.rand(1).uniform_()

            cal_source_distill = cal_target_distill = False
            if not self.training:
                cal_source_distill = True if self.use_source_distill else False
                cal_target_distill = True if self.use_target_distill else False
            else:
                if ctc_self_distill_choice <= self.ctc_self_distill_prob:
                    if self.use_source_distill and self.use_target_distill:
                        cal_source_distill = (
                            True
                            if ctc_self_distill_choice > self.ctc_self_distill_prob / 2
                            else False
                        )
                        cal_target_distill = not cal_source_distill
                    else:
                        cal_source_distill = self.use_source_distill
                        cal_target_distill = self.use_target_distill

            # source self distillation
            if cal_source_distill:
                ctc_self_distill_num = 0
                non_padding = non_padding_mask

                # if self.ctc_weight > 0 and self.ctc_self_distill_weight > 0 and inter_ctc_num > 0:
                if self.ctc_self_distill_weight > 0 and inter_ctc_num > 0:
                    teacher_logit = ctc_logit
                    student_logits = net_output["inter_ctc_logits"]
                    ctc_self_distill_num = inter_ctc_num
                elif self.ctc_self_distill_weight > 0 and inter_ctc_num > 1:
                    teacher_logit = net_output["inter_ctc_logits"][-1]
                    student_logits = net_output["inter_ctc_logits"][:-1]
                    ctc_self_distill_num = inter_ctc_num - 1

                if ctc_self_distill_num != 0:
                    source_ctc_self_distill_losses = self.get_ctc_self_distill_loss(
                        ctc_self_distill_num,
                        teacher_logit,
                        student_logits,
                        non_padding,
                        self.ctc_self_distill_temperature,
                    )

                    ctc_self_distill_num = len(source_ctc_self_distill_losses)
                    source_ctc_self_distill_loss = (
                        sum(source_ctc_self_distill_losses) / ctc_self_distill_num
                    )
                    logging_output["ctc_self_distill_loss"] = utils.item(
                        source_ctc_self_distill_loss.data
                    )
                    ctc_self_distill_loss += (
                        source_ctc_self_distill_loss * self.ctc_self_distill_weight
                    )

            # target self distillation
            if cal_target_distill:
                ctc_self_distill_num = 0
                non_padding = non_padding_mask

                if (
                    self.xctc_weight > 0
                    and self.xctc_self_distill_weight > 0
                    and inter_xctc_num > 0
                ):
                    teacher_logit = xctc_logit
                    student_logits = net_output["inter_xctc_logits"]
                    ctc_self_distill_num = inter_xctc_num
                elif self.xctc_self_distill_weight > 0 and inter_xctc_num > 1:
                    teacher_logit = net_output["inter_xctc_logits"][-1]
                    student_logits = net_output["inter_xctc_logits"][:-1]
                    ctc_self_distill_num = inter_xctc_num - 1

                if ctc_self_distill_num != 0:
                    xctc_self_distill_losses = self.get_ctc_self_distill_loss(
                        ctc_self_distill_num,
                        teacher_logit,
                        student_logits,
                        non_padding,
                        self.ctc_self_distill_temperature,
                    )

                    ctc_self_distill_num = len(xctc_self_distill_losses)

                    xctc_self_distill_loss = (
                        sum(xctc_self_distill_losses) / ctc_self_distill_num
                    )
                    logging_output["xctc_self_distill_loss"] = utils.item(
                        xctc_self_distill_loss.data
                    )
                    ctc_self_distill_loss += (
                        xctc_self_distill_loss * self.xctc_self_distill_weight
                    )

        ctc_mixup_consistent_loss = 0
        inter_ctc_mixup_consistent_loss = 0
        if use_ctc and mixup is True:
            mixup_coef = net_output["mixup"]["coef"]
            mixup_idx1 = net_output["mixup"]["index1"]
            mixup_idx2 = net_output["mixup"]["index2"]

            mixup_pos = mixup_idx1 != mixup_idx2
            mixup_real_coef = mixup_coef[mixup_pos]
            loss_coef = [mixup_real_coef, 1 - mixup_real_coef]
            mixup_real_idx1 = mixup_idx1[mixup_pos]
            mixup_real_idx2 = mixup_idx2[mixup_pos]


            def get_ctc_mixup_consistent_loss(ctc_logit, non_padding_mask):
                mixup_consistent_loss = 0
                mixup_real_logit = ctc_logit[:, mixup_pos, :]
                no_mixup_logit = ctc_logit[:, ~mixup_pos, :]
                mixup_target_logit = [
                    no_mixup_logit[:, mixup_real_idx1, :],
                    no_mixup_logit[:, mixup_real_idx2, :],
                ]
                mixup_target_pad_mask = [
                    non_padding_mask[mixup_real_idx1],
                    non_padding_mask[mixup_real_idx2],
                ]

                for logit, pad, coef in zip(
                    mixup_target_logit, mixup_target_pad_mask, loss_coef
                ):
                    loss = F.kl_div(
                        F.log_softmax(mixup_real_logit, dim=-1, dtype=torch.float32),
                        # F.log_softmax(logit, dim=-1, dtype=torch.float32),
                        F.log_softmax(logit.detach(), dim=-1, dtype=torch.float32),
                        log_target=True,
                        reduction="none",
                    )
                    mixup_consistent_loss += (
                        loss.sum(-1).transpose(0, 1).masked_fill(~pad, 0.0).sum(-1) * coef
                    ).sum()
                return mixup_consistent_loss

            if self.ctc_mixup_consistent_weight > 0:
                ctc_logit = net_output["ctc_logit"][0]
                ctc_mixup_consistent_loss = get_ctc_mixup_consistent_loss(ctc_logit, non_padding_mask)
                logging_output["ctc_mixup_consistent_loss"] = utils.item(
                    ctc_mixup_consistent_loss.data
                )

            if self.inter_ctc_mixup_consistent_weight > 0:
                if inter_ctc_num > 0:
                    logits = net_output["inter_ctc_logits"]
                    for i in range(inter_ctc_num):
                        logit = logits[i]
                        if type(logit) == list:
                            inter_ctc_logit = logit[0]
                            inter_non_padding_mask = ~logit[1] if logit[1] is not None else non_padding_mask
                        else:
                            inter_ctc_logit = logit
                            inter_non_padding_mask = non_padding_mask

                        inter_ctc_mixup_consistent_loss += get_ctc_mixup_consistent_loss(inter_ctc_logit, inter_non_padding_mask)

                logging_output["inter_ctc_mixup_consistent_loss"] = utils.item(
                    inter_ctc_mixup_consistent_loss.data
                )

        if len(ctc_entropy) != 0:
            ctc_entropy = sum(ctc_entropy) / len(ctc_entropy)
            logging_output["ctc_entropy"] = utils.item(ctc_entropy.data)
        else:
            ctc_entropy = 0

        loss = (
            self.ctc_weight * ctc_loss
            + self.axctc_weight * axctc_loss
            + self.xctc_weight * xctc_loss
            + self.inter_ctc_weight * inter_ctc_loss
            + self.inter_axctc_weight * inter_axctc_loss
            + self.inter_xctc_weight * inter_xctc_loss
            + ctc_self_distill_loss
            + self.ctc_entropy_weight * ctc_entropy
            + self.ctc_mixup_consistent_weight * ctc_mixup_consistent_loss
            + self.inter_ctc_mixup_consistent_weight * inter_ctc_mixup_consistent_loss
        )

        if loss != 0:
            logging_output["all_ctc_loss"] = utils.item(loss.data)

            if torch.isnan(loss) or torch.isinf(loss) or utils.item(loss.data) < 0:
                logger.warning("Illegal loss %f!" % loss)
                if ctc_loss != 0 and (torch.isnan(ctc_loss) or torch.isinf(ctc_loss)):
                    logger.warning("CTC loss %f!" % ctc_loss)

        # CER is not completely accurate and is for reference only.
        if not model.training:
            encoder = (
                model.encoder.encoder
                if hasattr(model.encoder, "encoder")
                else model.encoder
            )
            if hasattr(encoder, "ctc_valid"):
                if lprobs is not None:
                    lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()
                    if mixup:
                        # idx = mixup_idx1 if mixup_coef > 0.5 else mixup_idx2
                        # tokens = tokens[idx]
                        no_mixup_idx = mixup_idx1 == mixup_idx2
                        idx = mixup_idx1[no_mixup_idx]
                        lprobs_t = lprobs_t[idx]
                        tokens = tokens[idx]

                    c_err, c_len, w_errs, w_len, wv_errs = encoder.ctc_valid(
                        lprobs_t,
                        tokens,
                        input_lengths,
                        self.task.source_dictionary,
                        lang="source",
                    )

                    logging_output["wv_errors"] = wv_errs
                    logging_output["w_errors"] = w_errs
                    logging_output["w_total"] = w_len
                    logging_output["c_errors"] = c_err
                    logging_output["c_total"] = c_len

                if target_lprobs is not None:
                    target_lprobs_t = (
                        target_lprobs.transpose(0, 1).float().contiguous().cpu()
                    )
                    target_tokens = self.get_ctc_target_text(sample)
                    if mixup:
                        idx = mixup_idx1 if mixup_coef > 0.5 else mixup_idx2
                        target_tokens = target_tokens[idx]

                    c_err, c_len, w_errs, w_len, wv_errs = model.encoder.ctc_valid(
                        target_lprobs_t,
                        target_tokens,
                        input_lengths,
                        self.task.target_dictionary,
                        lang="target",
                    )

                    logging_output["target_wv_errors"] = wv_errs
                    logging_output["target_w_errors"] = w_errs
                    logging_output["target_w_total"] = w_len
                    logging_output["target_c_errors"] = c_err
                    logging_output["target_c_total"] = c_len

                if self.cal_all_ctc:
                    logging_output["save_dir"] = self.save_dir
                    for name, items in all_ctc_logits.items():
                        logit, lengths = items
                        if "target" in name:
                            dictionary = self.task.target_dictionary
                            ctc_tokens = target_tokens
                            lang = "target"
                        else:
                            dictionary = self.task.source_dictionary
                            ctc_tokens = tokens
                            lang = "source"
                        c_err, c_len, w_errs, w_len, wv_errs = model.encoder.ctc_valid(
                            logit, ctc_tokens, lengths, dictionary, lang
                        )
                        cer = c_err * 100 / c_len
                        wer = w_errs * 100 / w_len

                        logging_output["dump_%s_cer" % name] = cer
                        logging_output["dump_%s_wer" % name] = wer

        return loss, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        ctc_loss_sum = utils.item(
            sum(log.get("ctc_loss", 0) for log in logging_outputs)
        )
        ctc_entropy_sum = utils.item(
            sum(log.get("ctc_entropy", 0) for log in logging_outputs)
        )
        inter_ctc_loss_sum = utils.item(
            sum(log.get("inter_ctc_loss", 0) for log in logging_outputs)
        )
        xctc_loss_sum = utils.item(
            sum(log.get("xctc_loss", 0) for log in logging_outputs)
        )
        inter_xctc_loss_sum = utils.item(
            sum(log.get("inter_xctc_loss", 0) for log in logging_outputs)
        )
        axctc_loss_sum = utils.item(
            sum(log.get("axctc_loss", 0) for log in logging_outputs)
        )
        inter_axctc_loss_sum = utils.item(
            sum(log.get("inter_axctc_loss", 0) for log in logging_outputs)
        )
        ctc_self_distill_loss_sum = utils.item(
            sum(log.get("ctc_self_distill_loss", 0) for log in logging_outputs)
        )
        xctc_self_distill_loss_sum = utils.item(
            sum(log.get("xctc_self_distill_loss", 0) for log in logging_outputs)
        )
        ctc_mixup_consistent_loss = utils.item(
            sum(log.get("ctc_mixup_consistent_loss", 0) for log in logging_outputs)
        )
        inter_ctc_mixup_consistent_loss = utils.item(
            sum(log.get("inter_ctc_mixup_consistent_loss", 0) for log in logging_outputs)
        )
        all_ctc_loss_sum = utils.item(
            sum(log.get("all_ctc_loss", 0) for log in logging_outputs)
        )
        # loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))

        nfeatures = utils.item(sum(log.get("nfeatures", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        if all_ctc_loss_sum > 0:
            if "loss" not in logging_outputs[0]:
                metrics.log_scalar(
                    "loss",
                    all_ctc_loss_sum / sample_size / math.log(2),
                    sample_size,
                    round=3,
                )
            else:
                if all_ctc_loss_sum != ctc_loss_sum:
                    metrics.log_scalar(
                        "all_ctc_loss",
                        all_ctc_loss_sum / sample_size / math.log(2),
                        sample_size,
                        round=3,
                    )
        if ctc_loss_sum > 0:
            metrics.log_scalar(
                "ctc_loss",
                ctc_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )
        if ctc_entropy_sum > 0:
            metrics.log_scalar(
                "ctc_entropy",
                ctc_entropy_sum / nsentences / math.log(2),
                sample_size,
                round=3,
            )
        if inter_ctc_loss_sum > 0:
            metrics.log_scalar(
                "inter_ctc_loss",
                inter_ctc_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )
        if xctc_loss_sum > 0:
            metrics.log_scalar(
                "xctc_loss",
                xctc_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )
        if inter_xctc_loss_sum > 0:
            metrics.log_scalar(
                "inter_xctc_loss",
                inter_xctc_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )
        if axctc_loss_sum > 0:
            metrics.log_scalar(
                "axctc_loss",
                axctc_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )
        if inter_axctc_loss_sum > 0:
            metrics.log_scalar(
                "inter_axctc_loss",
                inter_axctc_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )

        if ctc_self_distill_loss_sum > 0:
            metrics.log_scalar(
                "ctc_self_distill_loss",
                ctc_self_distill_loss_sum / nsentences / math.log(2),
                sample_size,
                round=3,
            )
        if xctc_self_distill_loss_sum > 0:
            metrics.log_scalar(
                "xctc_self_distill_loss_sum",
                xctc_self_distill_loss_sum / nsentences / math.log(2),
                sample_size,
                round=3,
            )
        if ctc_mixup_consistent_loss > 0:
            metrics.log_scalar(
                "ctc_mixup_consistent_loss",
                ctc_mixup_consistent_loss / sample_size / math.log(2),
                sample_size,
                round=3,
            )
        if inter_ctc_mixup_consistent_loss > 0:
            metrics.log_scalar(
                "inter_ctc_mixup_consistent_loss",
                inter_ctc_mixup_consistent_loss / sample_size / math.log(2),
                sample_size,
                round=3,
            )

        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", ctc_loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        # wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        # metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "cer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )

        target_c_errors = sum(log.get("target_c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_target_c_errors", target_c_errors)
        target_c_total = sum(log.get("target_c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_target_c_total", target_c_total)
        target_w_errors = sum(log.get("target_w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_target_w_errors", target_w_errors)
        target_w_total = sum(log.get("target_w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_target_w_total", target_w_total)

        if target_c_total > 0:
            metrics.log_derived(
                "target_cer",
                lambda meters: safe_round(
                    meters["_target_c_errors"].sum
                    * 100.0
                    / meters["_target_c_total"].sum,
                    3,
                )
                if meters["_target_c_total"].sum > 0
                else float("nan"),
            )
        if target_w_total > 0:
            metrics.log_derived(
                "target_wer",
                lambda meters: safe_round(
                    meters["_target_w_errors"].sum
                    * 100.0
                    / meters["_target_w_total"].sum,
                    3,
                )
                if meters["_target_w_total"].sum > 0
                else float("nan"),
            )

        # save_dir = logging_outputs.get("save_dir", None)
        # if save_dir is not None and os.path.exists(save_dir):
        #     out = open(os.path.join(save_dir, "ctc_results"), "a")
        # else:
        #     out = sys.stdout
        #
        # for key in logging_outputs:
        #     if key.startswith("dump"):
        #         print("%s: %.2f" % (key, logging_outputs[key]), end="\t", file=out)
        # print("", file=out)
        # out.close()
        #
        # out = sys.stdout

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
