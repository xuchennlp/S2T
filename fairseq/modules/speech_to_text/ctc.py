import logging

import editdistance
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
)
from fairseq.data.data_utils import post_process

logger = logging.getLogger(__name__)


class CTC(nn.Module):

    def __init__(self, embed_dim, dictionary_size, dropout,
                 need_layernorm=False, dictionary=None):
        super(CTC, self).__init__()

        self.embed_dim = embed_dim
        self.ctc_projection = nn.Linear(embed_dim, dictionary_size)

        nn.init.normal_(self.ctc_projection.weight, mean=0, std=embed_dim ** -0.5)
        nn.init.constant_(self.ctc_projection.bias, 0.0)

        self.ctc_dropout_module = FairseqDropout(
            p=dropout, module_name=self.__class__.__name__
        )

        self.need_layernorm = need_layernorm
        if self.need_layernorm:
            self.LayerNorm = LayerNorm(embed_dim)

        self.dictionary = dictionary
        self.infer_decoding = False
        self.post_process = "sentencepiece"
        self.blank_idx = 0

        self.path = None
        self.save_stream = None

    def set_infer(self, is_infer, text_post_process, dictionary, path):
        self.infer_decoding = is_infer
        self.post_process = text_post_process
        self.dictionary = dictionary
        self.path = path
        if self.path is not None:
            self.save_stream = open(self.path, "a")
        else:
            self.save_stream = None

    def forward(self, x, padding=None, tag=None, is_top=False):
        if self.need_layernorm:
            x = self.LayerNorm(x)

        x = self.ctc_projection(self.ctc_dropout_module(x))

        if not self.training and self.infer_decoding and is_top:
            assert self.dictionary is not None
            input_lengths = (~padding).sum(-1)
            self.infer(x.transpose(0, 1).float().contiguous().cpu(), input_lengths, tag)

        return x

    def softmax(self, x, temperature=1.0):
        return F.softmax(self.ctc_projection(x) / temperature, dim=-1, dtype=torch.float32)

    def log_softmax(self, x, temperature=1.0):
        return F.log_softmax(self.ctc_projection(x) / temperature, dim=-1, dtype=torch.float32)

    def argmax(self, x):
        return torch.argmax(self.ctc_projection(x), dim=-1)

    def infer(self, logits_or_probs, lengths, tag=None):
        for lp, inp_l in zip(
                logits_or_probs,
                lengths,
        ):
            lp = lp[:inp_l].unsqueeze(0)

            toks = lp.argmax(dim=-1).unique_consecutive()
            pred_units_arr = toks[toks != self.dictionary.bos()].tolist()

            pred_units = self.dictionary.string(pred_units_arr)
            pred_words_raw = post_process(pred_units, self.post_process).split()

            if self.save_stream is not None:
                self.save_stream.write(" ".join(pred_words_raw) + "\n")

            if tag is not None:
                logger.info("%s CTC prediction: %s" % (tag, " ".join(pred_words_raw)))
            else:
                logger.info("CTC prediction: %s" % (" ".join(pred_words_raw)))

    def valid(self, logits_or_probs, targets, input_lengths, dictionary):

        c_err = 0
        c_len = 0
        w_errs = 0
        w_len = 0
        wv_errs = 0

        with torch.no_grad():
            for lp, t, inp_l in zip(
                    logits_or_probs,
                    targets,
                    input_lengths,
            ):
                lp = lp[:inp_l].unsqueeze(0)

                p = (t != dictionary.pad()) & (t != dictionary.eos())
                targ = t[p]
                targ_units = dictionary.string(targ)
                targ_units_arr = targ.tolist()

                toks = lp.argmax(dim=-1).unique_consecutive()
                pred_units_arr = toks[toks != self.blank_idx].tolist()

                c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                c_len += len(targ_units_arr)

                targ_words = post_process(targ_units, self.post_process).split()

                pred_units = dictionary.string(pred_units_arr)
                pred_words_raw = post_process(pred_units, self.post_process).split()

                dist = editdistance.eval(pred_words_raw, targ_words)
                w_errs += dist
                wv_errs += dist

                w_len += len(targ_words)

        return c_err, c_len, w_errs, w_len, wv_errs
