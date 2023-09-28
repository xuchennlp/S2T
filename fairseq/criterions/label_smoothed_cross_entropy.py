# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    mixup_consistent_weight: float = field(
        default=0.0,
        metadata={"help": "the weight for consistency regularization of mixup"},
    )
    cal_mixup_loss: bool = field(
        default=False,
        metadata={"help": "calculate the loss for the mixed samples"},
    ) 
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
            cal_mixup_loss=True,
            mixup_consistent_weight=0.0,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = float(label_smoothing)
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.cal_mixup_loss = cal_mixup_loss
        self.mixup_consistent_weight = mixup_consistent_weight

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss, other_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
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
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True, sample=sample)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()

        return lprobs.view(-1, lprobs.size(-1)), target

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss = nll_loss = 0
        other_loss = dict()

        if "mixup" in net_output[1] and net_output[1]["mixup"] is not None:
            mixup = net_output[1]["mixup"]
            idx1 = mixup["index1"]
            idx2 = mixup["index2"]
            mixup_flag = mixup["mixup_flag"]
            mixup_idx1 = idx1[mixup_flag]
            mixup_idx2 = idx2[mixup_flag]
            org_idx = idx1[~mixup_flag]

            seq_len = target.size(1)
            lprobs = lprobs.view(-1, seq_len, lprobs.size(-1))

            if mixup["mixup_decoder_emb"]:
                org_lprobs = lprobs[~mixup_flag, :, :]
            else:
                decoder_org_flag = mixup["decoder_org_flag"] 
                org_lprobs = lprobs[decoder_org_flag, :, :]

            if len(org_idx) > 0:
                org_target = target[org_idx]
                org_loss, org_nll_loss = label_smoothed_nll_loss(
                    org_lprobs.view(-1, org_lprobs.size(-1)),
                    org_target.view(-1),
                    self.eps,
                    ignore_index=self.padding_idx,
                    reduce=False,
                ) 
                loss += org_loss.sum()
                nll_loss += org_nll_loss.sum()

            if any(mixup_flag):
                if mixup["mixup_decoder_emb"]:
                    mixup_lprobs = [lprobs[mixup_flag, :, :], lprobs[mixup_flag, :, :]]
                else:
                    decoder_mixup_flag1 = mixup["decoder_mixup_flag1"]
                    decoder_mixup_flag2 = mixup["decoder_mixup_flag2"]
                    mixup_lprobs = [lprobs[decoder_mixup_flag1, :, :], lprobs[decoder_mixup_flag2, :, :]] 

                mixup_targets = [target[mixup_idx1], target[mixup_idx2]]
                mixup_coef = net_output[1]["mixup"]["coef"][mixup_flag]
                loss_coef = [mixup_coef, 1 - mixup_coef]

                if self.cal_mixup_loss:
                    for item_lprobs, item_target, item_coef in zip(mixup_lprobs, mixup_targets, loss_coef):
                        batch_size = item_target.size(0)
                        item_loss, item_nll_loss = label_smoothed_nll_loss(
                            item_lprobs.view(-1, item_lprobs.size(-1)),
                            item_target.view(-1),
                            self.eps,
                            ignore_index=self.padding_idx,
                            reduce=False,
                        )
                        loss += (item_loss.sum(-1).view(batch_size, -1).sum(-1) * item_coef).sum()
                        nll_loss += (item_nll_loss.sum(-1).view(batch_size, -1).sum(-1) * item_coef).sum()
                
                mixup_consistent_loss = 0
                if self.mixup_consistent_weight > 0:
                    non_padding_mask = ~org_target.eq(self.padding_idx)
    
                    teacher_lprobs = [org_lprobs[mixup_idx1, :, :], org_lprobs[mixup_idx2, :, :]]
                    target_pad_mask = [non_padding_mask[mixup_idx1], non_padding_mask[mixup_idx2]]
    
                    for item_mixup_lprobs, tgt_lprobs, pad, coef in zip(mixup_lprobs, teacher_lprobs, target_pad_mask, loss_coef):
                        item_loss = F.kl_div(
                            F.log_softmax(item_mixup_lprobs, dim=-1, dtype=torch.float32),
                            F.log_softmax(tgt_lprobs.detach(), dim=-1, dtype=torch.float32),
                            log_target=True,
                            reduction="none",
                        )
                        mixup_consistent_loss += (item_loss.sum(-1).masked_fill_(~pad, 0.0).sum(-1) * coef).sum()
                    other_loss["mixup_consistent_loss"] = mixup_consistent_loss * self.mixup_consistent_weight
        else:
            target = target.view(-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs,
                target,
                self.eps,
                ignore_index=self.padding_idx,
                reduce=reduce,
            )
        return loss, nll_loss, other_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)

        if "mixup" in net_output[1] and net_output[1]["mixup"] is not None:
            mixup = net_output[1]["mixup"]
            mixup_idx1 = mixup["index1"]
            mixup_idx2 = mixup["index2"]
            mixup_flag = mixup["mixup_flag"]

            if all(mixup_flag):
                return torch.Tensor([0]), torch.Tensor([0])

            idx = mixup_idx1[~mixup_flag]
            lprobs = lprobs.view(-1, target.size(1), lprobs.size(-1))[idx, :, :].view(-1, lprobs.size(-1))
            target = target[idx].view(-1)
        else:
            target = target.view(-1)

        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        mixup_consistent_loss_sum = sum(log.get("mixup_consistent_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )
        if mixup_consistent_loss_sum != 0:
            metrics.log_scalar(
                "mixup_consistent_loss", mixup_consistent_loss_sum / sample_size / math.log(2), sample_size, round=3
            )

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
