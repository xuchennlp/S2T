#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-Head Attention layer definition."""

import math
import torch
from torch import nn
import torch.nn.functional as F
import logging
from fairseq.modules.rotary_positional_embedding import (
    RotaryPositionalEmbedding,
    apply_rotary_pos_emb,
)
from .layer_norm import LayerNorm
from torch.distributions import Categorical



class ESPNETMultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.
    Args:
        n_head: The number of heads.
        n_feat: The number of features.
        dropout: Dropout rate.
    """

    def __init__(self, n_feat, n_head, dropout):
        """Construct an MultiHeadedAttention object."""
        super(ESPNETMultiHeadedAttention, self).__init__()
        self.encoder_decoder_attention = False
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        self.cal_localness = False
        self.localness = None
        self.localness_window = 0.1
        self.localness_num = 0

        self.cal_entropy = False
        self.entropy = None
        self.entropy_num = 0
    
        self.cal_topk = False
        self.topk_weights = None
        self.topk_num = 0

        self.cal_monotonic = False
        self.monotonic_weights = None
        self.monotonic_num = 0

    def set_flag(self, **kwargs):
        if kwargs.get("cal_localness", False) and not self.encoder_decoder_attention: 
            self.cal_localness = True
            self.localness_window = kwargs.get("localness_window", 0.1)
        if kwargs.get("cal_entropy", False): # and self.encoder_decoder_attention: 
            self.cal_entropy = True
        if kwargs.get("cal_topk_cross_attn_weights", False):
            self.cal_topk = True
            self.weights_topk = kwargs.get("topk_cross_attn_weights", 1)
        if kwargs.get("cal_monotonic_cross_attn_weights", False) and self.encoder_decoder_attention: 
            self.cal_monotonic = True

    def dump(self, fstream, info):
        if self.cal_localness:
            print("%s window size: %.2f localness: %.4f" % (info, self.localness_window, self.localness), file=fstream)
        
        if self.cal_entropy:
            print("%s Entropy: %.2f" % (info, self.entropy), file=fstream)

        if self.cal_topk:
            print("%s top%d attn weights: %s" % (info, self.weights_topk, " ".join([str(round(w, 2)) for w in self.topk_weights.tolist()])), file=fstream)
        
        if self.cal_monotonic:
            print("%s monotonic cross attn weights: %f" % (info, self.monotonic_weights), file=fstream)

    def forward_qkv(self, query, key, value, **kwargs):
        """Transform query, key and value.
        Args:
            query: Query tensor  B X T1 X C
            key: Key tensor B X T2 X C
            value: Value tensor  B X T2 X C
        Returns:
            torch.Tensor: Transformed query tensor  B X n_head X T1 X d_k
            torch.Tensor: Transformed key tensor B X n_head X T2 X d_k
            torch.Tensor: Transformed value tensor  B X n_head X T2 X d_k
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.
        Args:
            value: Transformed value B X n_head X T2 X d_k.
            scores: Attention score  B X n_head X T1 X T2
            mask: Mask  T2 X B
        Returns:
            torch.Tensor: Transformed value  B X T1 X d_model
                weighted by the attention score  B X T1 X T2
        """
        n_batch = value.size(0)
        if mask is not None:
            scores = scores.masked_fill(
                mask.unsqueeze(1).unsqueeze(2).to(bool),
                # -1e8 if scores.dtype == torch.float32 else -1e4
                float("-inf"),  # (batch, head, time1, time2)
            )
        # self.attn = torch.softmax(scores, dim=-1)   # (batch, head, time1, time2)

        scores = scores.clamp(min=-1e8 if scores.dtype == torch.float32 else -1e4,
                              max=1e8 if scores.dtype == torch.float32 else 1e4)
        self.attn = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(scores)  # (batch, head, time1, time2)

        if torch.isnan(self.attn).any():
            logging.warning("Tensor attention scores has nan.")
            # torch.save(scores, "scores.pt")
            # torch.save(self.attn, "attn.pt")
            # exit()

        attn_weights_float = self.attn
        self.num_heads = self.h
        bsz = n_batch
        src_len = scores.size(3)
        tgt_len = scores.size(2)
        # self.cal_localness_func(self.attn, n_batch, scores.size(3), scores.size(2))
        self.cal_localness_func(attn_weights_float, bsz, src_len, tgt_len)
        self.cal_entropy_func(attn_weights_float, bsz, src_len, tgt_len)
        self.cal_topk_func(attn_weights_float, bsz, src_len, tgt_len)
        self.cal_monotonic_func(attn_weights_float, bsz, src_len, tgt_len)


        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, key_padding_mask=None, **kwargs):
        """Compute scaled dot product attention.
        Args:
            query (torch.Tensor): Query tensor T X B X C
            key (torch.Tensor): Key tensor T X B X C
            value (torch.Tensor): Value tensor T X B X C
            key_padding_mask (torch.Tensor): Mask tensor T X B
        Returns:
            torch.Tensor: Output tensor T X B X D.
        """
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = self.forward_attention(v, scores, key_padding_mask)
        scores = scores.transpose(0, 1)
        return scores, None

    def cal_localness_func(self, attn_weights_float, bsz, src_len, tgt_len):
        if not self.training and self.cal_localness:
            weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0).mean(dim=0)

            localness = 0
            item_localness = 0
            window = int(src_len * self.localness_window)
            for i in range(window, src_len - window):
                item_localness = 0
                for j in range(-window, window + 1):
                    # if j == 0:
                        # continue
                    item_localness += weights[:, i, i + j]
                localness += item_localness
            localness = localness / (src_len - 2 * window)
            localness *= 100

            if self.localness_num == 0:
                self.localness = localness.mean()
            else:
                self.localness = (self.localness * self.localness_num + localness.mean()) / (self.localness_num + 1)
            self.localness_num += 1
    
    def cal_entropy_func(self, attn_weights_float, bsz, src_len, tgt_len):
        if not self.training and self.cal_entropy:
            weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)

            entropy = Categorical(weights).entropy()
            # mean_entropy = entropy.mean([1, 2])
            mean_entropy = entropy.mean()

            if self.entropy_num == 0:
                self.entropy = mean_entropy
            else:
                self.entropy = (self.entropy * self.entropy_num + mean_entropy) / (self.entropy_num + 1)
            self.entropy_num += 1

    def cal_topk_func(self, attn_weights_float, bsz, src_len, tgt_len):
        if not self.training and self.cal_topk:
            weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)

            weights_topk = min(src_len, self.weights_topk)
            topk_weights = torch.topk(weights, k=weights_topk, dim=-1, sorted=True)[0]
            mean_topk_weights = topk_weights.mean([0, 1, 2])

            if weights_topk < self.weights_topk:
                zero = torch.zeros(self.weights_topk - len(mean_topk_weights)).type_as(mean_topk_weights)
                mean_topk_weights = torch.cat((mean_topk_weights, zero))
            
            sum_topk_weights = torch.cumsum(mean_topk_weights, dim=0)

            if self.topk_num == 0:
                self.topk_weights = sum_topk_weights
            else:
                self.topk_weights = (self.topk_weights * self.topk_num + sum_topk_weights) / (self.topk_num + 1)
            self.topk_num += 1

    def cal_monotonic_func(self, attn_weights_float, bsz, src_len, tgt_len):
        if not self.training and self.cal_monotonic:
            weights = attn_weights_float.view(
                bsz * self.num_heads, tgt_len, src_len
            )

            topk_idx = torch.topk(weights, k=1, dim=-1)[1]

            topk_idx_prev = topk_idx[:, :-1, :]
            topk_idx_last = topk_idx[:, 1:, :]
            is_monotonic = topk_idx_last > topk_idx_prev
            
            monotonic_weight = (is_monotonic == True).sum() / is_monotonic.numel()

            # monotonic_weight = 0
            # for i in range(0, bsz * self.num_heads):
            #     sent_weight = weights[i, :, :]
            #     for j in range(1, tgt_len):
            #         monotonic_weight += sum(sent_weight[j, :topk_idx[i, j - 1]])
            
            # monotonic_weight /= (bsz * self.num_heads * (tgt_len - 1))

            if self.monotonic_num == 0:
                self.monotonic_weights = monotonic_weight
            else:
                self.monotonic_weights = (self.monotonic_weights * self.monotonic_num + monotonic_weight) / (self.monotonic_num + 1)
            self.monotonic_num += 1


class RelPositionMultiHeadedAttention(ESPNETMultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head: The number of heads.
        n_feat: The number of features.
        dropout: Dropout rate.
        zero_triu: Whether to zero the upper triangular part of attention matrix.
    """

    def __init__(self, n_feat, n_head, dropout, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_feat, n_head, dropout)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.
        Args:
            x: Input tensor B X n_head X T X 2T-1
        Returns:
            torch.Tensor: Output tensor.
        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
        ]  # only keep the positions from 0 to time2

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, key_padding_mask=None, **kwargs):
        """Compute scaled dot product attention.
        Args:
            query: Query tensor T X B X C
            key: Key tensor T X B X C
            value: Value tensor T X B X C
            pos_emb: Positional embedding tensor 2T-1 X B(1) X C
            key_padding_mask: Mask tensor T X B
        Returns:
            torch.Tensor: Output tensor T X B X C.
        """
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        pos_emb = pos_emb.transpose(0, 1)
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)
        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, 2*time1-1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        scores = self.forward_attention(v, scores, key_padding_mask)
        scores = scores.transpose(0, 1)
        return scores, None


class ReducedRelPositionMultiHeadedAttention(RelPositionMultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head: The number of heads.
        n_feat: The number of features.
        dropout: Dropout rate.
        zero_triu: Whether to zero the upper triangular part of attention matrix.
    """

    def __init__(self, n_feat, n_head, dropout, zero_triu=False,
                 sample_ratio=1,
                 reduced_method="conv",
                 reduced_q=False,
                 ):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_feat, n_head, dropout)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

        super().__init__(n_feat, n_head, dropout, zero_triu)

        self.sample_ratio = sample_ratio
        self.reduced_method = reduced_method
        self.reduced_q = reduced_q
        if reduced_q:
            assert self.reduced_method == 'group', "only support grouped method for query reduction"

        if self.sample_ratio > 1:
            if reduced_method == "conv":
                self.sr = nn.Conv1d(n_feat, n_feat,
                                    kernel_size=sample_ratio,
                                    stride=sample_ratio,
                                    )
                self.norm = LayerNorm(n_feat)
            elif reduced_method == "pool":
                self.linear = nn.Linear(n_feat, n_feat)
                self.norm = LayerNorm(n_feat)
                self.act = nn.GELU()
            elif reduced_method == "group":
                pass

    def forward(self, query, key, value, pos_emb, key_padding_mask=None, **kwargs):
        """Compute scaled dot product attention.
        Args:
            query: Query tensor T X B X C
            key: Key tensor T X B X C
            value: Value tensor T X B X C
            pos_emb: Positional embedding tensor 2T-1 X B(1) X C
            key_padding_mask: Mask tensor T X B
        Returns:
            torch.Tensor: Output tensor T X B X C.
        """
        # (bsz, seq_len, dim)
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        pos_emb = pos_emb.transpose(0, 1)

        tgt_len = query.size(1)

        query_ = query
        if self.sample_ratio > 1:
            assert tgt_len % self.sample_ratio == 0, \
                ("sample ratio %d is mismatched with length %d" % (self.sample_ratio, tgt_len))
            if self.reduced_method == "conv":
                query_ = query.transpose(1, 2)  # bsz, dim, seq_len
                query_ = self.sr(query_).transpose(1, 2)  # bsz, seq_len, dim
                query_ = self.norm(query_)
            elif self.reduced_method == "pool":
                query_ = query.transpose(1, 2)  # bsz, dim, seq_len
                pool_length = int(tgt_len / self.sample_ratio)
                query_ = nn.functional.adaptive_max_pool1d(query_, pool_length).transpose(1, 2)
                query_ = self.act(self.norm(query_))

            key = value = query_
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask[:, ::self.sample_ratio]

        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        # q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)
        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, 2*time1-1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        scores = self.forward_attention(v, scores, key_padding_mask)
        scores = scores.transpose(0, 1)
        return scores, None


class LegacyRelPositionMultiHeadedAttention(RelPositionMultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (old version).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.
    """
    def __init__(self, n_feat, n_head, dropout, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_feat, n_head, dropout, zero_triu)

    def rel_shift(self, x):
        """Compute relative positional encoding.
        Args:
            x: Input tensor B X n_head X T X 2T-1
        Returns:
            torch.Tensor: Output tensor.
        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x


class RotaryPositionMultiHeadedAttention(ESPNETMultiHeadedAttention):
    def __init__(
        self,
        n_feat,
        n_head,
        dropout,
        precision,
        rotary_emd_base=10000,
    ):
        """Construct an RotaryPositionMultiHeadedAttention object."""
        super().__init__(n_feat, n_head, dropout)
        precision = torch.float
        self.rotary_ndims = self.d_k  # also try self.d_k//2
        if precision == "fp16":
            precision = torch.half

        self.rotary_emb = RotaryPositionalEmbedding(
            self.rotary_ndims, base=rotary_emd_base, precision=precision
        )

    def forward(self, query, key, value, key_padding_mask=None, **kwargs):
        """Compute rotary position attention.
        Args:
            query: Query tensor T X B X C
            key: Key tensor T X B X C
            value: Value tensor T X B X C
            key_padding_mask: Mask tensor T X B
        Returns:
            torch.Tensor: Output tensor T X B X D.
        Notes:
            Assumes self attn
        """

        T, B, C = value.size()
        query = query.view(T, B, self.h, self.d_k)
        key = key.view(T, B, self.h, self.d_k)
        value = value.view(T, B, self.h, self.d_k)
        cos, sin = self.rotary_emb(value, seq_len=T)
        query, key = apply_rotary_pos_emb(
            query, key, cos, sin, offset=0
        )  # offset is based on layer_past

        query = query.view(T, B, self.h * self.d_k)
        key = key.view(T, B, self.h * self.d_k)
        value = value.view(T, B, self.h * self.d_k)

        # TBD to BTD
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = self.forward_attention(v, scores, key_padding_mask)
        scores = scores.transpose(0, 1)
        return scores, None
