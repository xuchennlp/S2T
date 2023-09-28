# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.data.data_utils import post_process
from fairseq.logging.meters import safe_round

from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from .ctc import CtcCriterion, CtcCriterionConfig


@register_criterion("join_speech_and_text_loss")
class JoinSpeechTextLoss(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self, task, label_smoothing,
                 sentence_avg,
                 cfg: CtcCriterionConfig,
                 ctc_weight=0.0):
        super().__init__(task, sentence_avg, label_smoothing)

        self.report_accuracy = True
        self.ctc_weight = ctc_weight
        self.ctc_criterion = CtcCriterion(cfg, task, ctc_weight)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        CtcCriterion.add_args(parser)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        speech_tokens, speech_lengths, prev_output_tokens = sample["net_input"].values()
        text_src_tokens = sample["transcript"]["tokens"]
        text_src_lengths = sample["transcript"]["lengths"]

        encoder_out = model.encoder(speech_tokens, speech_lengths,
                                    text_src_tokens, text_src_lengths)
        net_output = model.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )

        use_mixup = False
        if "mixup" in encoder_out and encoder_out["mixup"] is not None:
            use_mixup = True

        loss, nll_loss, other_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        n_tokens = sample["ntokens"]
        n_sentences = sample["target"].size(0)
        if use_mixup:
            sample_size //= 2
            n_tokens //= 2
            n_sentences //= 2

        logging_output = {
            "trans_loss": utils.item(loss.data) if reduce else loss.data,
            "nll_loss": utils.item(nll_loss.data) if reduce else nll_loss.data,
            "ntokens": n_tokens,
            "nsentences": n_sentences,
            "sample_size": sample_size,
        }

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        if self.ctc_criterion.all_ctc_weight > 0:
            ctc_loss, logging_output = self.ctc_criterion.compute_ctc_loss(model, sample, encoder_out, logging_output)
            loss = (1 - self.ctc_weight) * loss + ctc_loss
        logging_output["loss"] = utils.item(loss.data) if reduce else loss.data

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        trans_loss_sum = utils.item(
            sum(log.get("trans_loss", 0) for log in logging_outputs)
        )
        nll_loss_sum = utils.item(
            sum(log.get("nll_loss", 0) for log in logging_outputs)
        )
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if trans_loss_sum != loss_sum:
            metrics.log_scalar(
                "trans_loss", trans_loss_sum / ntokens / math.log(2), ntokens, round=3
            )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )
        if "ctc_loss" in logging_outputs[0] or "all_ctc_loss" in logging_outputs[0]:
            CtcCriterion.reduce_metrics(logging_outputs)

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
