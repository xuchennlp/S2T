# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import logging
import random

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.data.data_utils import post_process
from fairseq.logging.meters import safe_round

from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from .ctc import CtcCriterion, CtcCriterionConfig

logger = logging.getLogger(__name__)


@register_criterion("label_smoothed_cross_entropy_with_ctc")
class LabelSmoothedCrossEntropyCriterionWithCTC(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self, task, label_smoothing,
                sentence_avg,
                cfg: CtcCriterionConfig,
                ctc_weight=0.0,
                save_dir=None,
                cal_mixup_loss=True,
                mixup_consistent_weight=0.0,
                only_train_enc_prob=0.0,
                get_oracle_when_only_train_enc=False
                ):
        super().__init__(task, sentence_avg, label_smoothing,
                        report_accuracy=True,
                        cal_mixup_loss=cal_mixup_loss,
                        mixup_consistent_weight=mixup_consistent_weight)

        self.report_accuracy = True
        self.ctc_weight = ctc_weight
        self.ctc_criterion = CtcCriterion(cfg, task, ctc_weight, save_dir)
        self.save_dir = save_dir

        self.only_train_enc_prob = only_train_enc_prob
        if self.only_train_enc_prob > 0:
            logger.info("Only train the encoder with a probability of %.2f" % self.only_train_enc_prob)

        self.get_oracle_when_only_train_enc = get_oracle_when_only_train_enc
        if self.get_oracle_when_only_train_enc> 0:
            logger.info("Get oracle when only the encoder is trained")

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        CtcCriterion.add_args(parser)

        parser.add_argument(
            "--only-train-enc-prob",
            type=float,
            default=0,
            help="the probability to train the encoder only",
        )
        parser.add_argument(
            "--get-oracle-when-only-train-enc",
            action="store_true",
            help="get the oracle for ctc when train the encoder only",
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        src_tokens, src_lengths, prev_output_tokens = sample["net_input"].values()

        train_enc_only = False
        if self.training and self.only_train_enc_prob != 0 and self.ctc_criterion.all_ctc_weight > 0:
            p = torch.rand(1).uniform_()
            if p < self.only_train_enc_prob:
                train_enc_only = True

        if hasattr(model.encoder, "use_raw_text") and model.encoder.use_raw_text:
            assert "transcript" in sample
            text_src_tokens = sample["transcript"]["tokens"]
            text_src_lengths = sample["transcript"]["lengths"]

            encoder_out = model.encoder(src_tokens, src_lengths,
                                        text_src_tokens, text_src_lengths)
        else:
            if self.training and not (self.get_oracle_when_only_train_enc and not train_enc_only) and getattr(model.encoder, "pae_ground_truth_ratio", 0) != 0:

                seed = random.randint(0, 100000)
                with utils.set_torch_seed(seed):
                    ctc_alignment_oracle = self.ctc_criterion.get_ground_truth_alignment(model, sample)
                    encoder_out = model.encoder(src_tokens, src_lengths,
                                                ctc_alignment_oracle=ctc_alignment_oracle)
            else:
                encoder_out = model.encoder(src_tokens=src_tokens, src_lengths=src_lengths)

        net_output = model.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )

        loss, nll_loss, other_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        n_tokens = sample["ntokens"]
        n_sentences = sample["target"].size(0)

        if "mixup" in encoder_out and encoder_out["mixup"] is not None:
            mixup = encoder_out["mixup"]
            ratio = mixup["ratio"]

            if mixup["keep_org"]:
                n_tokens = int(n_tokens * (1 + ratio))
                sample_size = int(sample_size * (1 + ratio)) if self.sentence_avg else n_tokens
                n_sentences = int(n_sentences * (1 + ratio))
            else:
                if ratio > 1:
                    n_tokens = int(n_tokens * ratio)
                    sample_size = int(sample_size * ratio) if self.sentence_avg else n_tokens
                    n_sentences = int(n_sentences * ratio)

        logging_output = {
            "trans_loss": utils.item(loss.data) if reduce else loss.data,
            "nll_loss": utils.item(nll_loss.data) if reduce else nll_loss.data,
            "ntokens": n_tokens,
            "nsentences": n_sentences,
            "sample_size": sample_size,
        }
        if len(other_loss) != 0:
            for key, value in other_loss.items():
                loss += value
                logging_output[key] = utils.item(value.data)

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        ctc_loss = 0
        if self.ctc_criterion.all_ctc_weight > 0:
            ctc_loss, logging_output = self.ctc_criterion.compute_ctc_loss(model, sample, encoder_out, logging_output)
            loss = loss + ctc_loss

        # if hasattr(model.encoder, "get_loss"):
        #     encoder_loss = model.encoder.get_loss()
        #     if encoder_loss != 0:
        #         loss += encoder_loss * sample_size
        #         logging_output["encoder_loss"] = utils.item(encoder_loss.data)
        logging_output["loss"] = utils.item(loss.data) if reduce else loss.data

        if train_enc_only:
            loss = ctc_loss

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
        mixup_consistent_loss_sum = utils.item(
            sum(log.get("mixup_consistent_loss", 0) for log in logging_outputs)
        )
        enc_loss_sum = utils.item(
            sum(log.get("encoder_loss", 0) for log in logging_outputs)
        )

        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        n_sentences = utils.item(sum(log.get("nsentences", 0) for log in logging_outputs))
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
        if mixup_consistent_loss_sum != 0:
            metrics.log_scalar(
                "mixup_consistent_loss", mixup_consistent_loss_sum / ntokens / math.log(2), ntokens, round=3
            )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )
        if enc_loss_sum != 0:
            metrics.log_scalar("enc_loss", enc_loss_sum, sample_size, round=3)

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
