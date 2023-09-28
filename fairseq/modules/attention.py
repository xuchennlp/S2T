# Copyright 2021, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# Multi-Head Attention Layers
###############################################################################


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layer

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads

    References:
        Attention Is All You Need, Vaswani et al.
        https://arxiv.org/abs/1706.03762

    """

    def __init__(self, dim_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        # Attention Params
        self.num_heads = num_heads  # H
        self.dim_model = dim_model  # D
        self.dim_head = dim_model // num_heads  # d

        # Linear Layers
        self.query_layer = nn.Linear(self.dim_model, self.dim_model)
        self.key_layer = nn.Linear(self.dim_model, self.dim_model)
        self.value_layer = nn.Linear(self.dim_model, self.dim_model)
        self.output_layer = nn.Linear(self.dim_model, self.dim_model)

    def forward(self, query, key, value, mask=None):

        """Scaled Dot-Product Multi-Head Attention

        Args:
            query: Query of shape (B, T, D)
            key: Key of shape (B, T, D)
            value: Value of shape (B, T, D)
            mask: Optional position mask of shape (1 or B, 1 or H, 1 or T, 1 or T)

        Return:
            out: Attention output of shape (B, T, D)
            att_w: Attention weights of shape (B, H, T, T)

        """

        # Batch size B
        batch_size = query.size(0)

        # Linear Layers
        query = self.query_layer(query)
        key = self.key_layer(key)
        value = self.value_layer(value)

        # Reshape and Transpose (B, T, D) -> (B, H, T, d)
        query = query.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # Att scores (B, H, T, T)
        att_scores = query.matmul(key.transpose(2, 3)) / key.shape[-1] ** 0.5

        # Apply mask
        if mask is not None:
            att_scores += (mask * -1e9)

        # Att weights (B, H, T, T)
        att_w = att_scores.softmax(dim=-1)

        # Att output (B, H, T, d)
        out = att_w.matmul(value)

        # Transpose and Reshape (B, H, T, d) -> (B, T, D)
        out = out.transpose(1, 2).reshape(batch_size, -1, self.dim_model)

        # Output linear layer
        out = self.output_layer(out)

        return out, att_w.detach()

    def pad(self, query, key, value, mask, chunk_size):

        # Compute Overflows
        overflow_Q = query.size(1) % chunk_size
        overflow_KV = key.size(1) % chunk_size

        padding_Q = chunk_size - overflow_Q if overflow_Q else 0
        padding_KV = chunk_size - overflow_KV if overflow_KV else 0

        batch_size, seq_len_KV, _ = key.size()

        # Input Padding (B, T, D) -> (B, T + P, D)
        query = F.pad(query, (0, 0, 0, padding_Q), value=0)
        key = F.pad(key, (0, 0, 0, padding_KV), value=0)
        value = F.pad(value, (0, 0, 0, padding_KV), value=0)

        # Update Padding Mask
        if mask is not None:

            # (B, 1, 1, T) -> (B, 1, 1, T + P)
            if mask.size(2) == 1:
                mask = F.pad(mask, pad=(0, padding_KV), value=1)
            # (B, 1, T, T) -> (B, 1, T + P, T + P)
            else:
                mask = F.pad(mask, pad=(0, padding_Q, 0, padding_KV), value=1)

        elif padding_KV:

            # None -> (B, 1, 1, T + P)
            mask = F.pad(query.new_zeros(batch_size, 1, 1, seq_len_KV), pad=(0, padding_KV), value=1)

        return query, key, value, mask, padding_Q


class GroupedMultiHeadAttention(MultiHeadAttention):
    """Grouped Multi-Head Attention Layer

    Grouped multi-head attention reduces attention complexity from out(T2·D) to out(T2·D/G)
    by grouping neighbouring time elements along the feature dimension before applying
    scaled dot-product attention.

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        group_size: attention group size

    """

    def __init__(self, dim_model, num_heads, group_size):
        super(GroupedMultiHeadAttention, self).__init__(dim_model, num_heads)

        # Attention Params
        self.group_size = group_size  # G
        self.dim_head = (self.group_size * dim_model) // self.num_heads  # d

    def forward(self, query, key, value, mask=None):
        # Batch size B
        batch_size = query.size(0)

        # Linear Layers
        query = self.query_layer(query)
        key = self.key_layer(key)
        value = self.value_layer(value)

        # Chunk Padding
        query, key, value, mask, padding = self.pad(query, key, value, mask, chunk_size=self.group_size)

        # Reshape and Transpose (B, T, D) -> (B, H, T//G, d)
        query = query.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # Att scores (B, H, T//G, T//G)
        att_scores = query.matmul(key.transpose(2, 3)) / key.shape[-1] ** 0.5

        # Apply mask
        if mask is not None:
            # Slice Mask (B, 1, T, T) -> (B, 1, T//G, T//G)
            mask = mask[:, :, ::self.group_size, ::self.group_size]

            # Apply mask
            att_scores += (mask * -1e9)

        # Att weights (B, H, T//G, T//G)
        att_w = att_scores.softmax(dim=-1)

        # Att output (B, H, T//G, d)
        out = att_w.matmul(value)

        # Transpose and Reshape (B, H, T//G, d) -> (B, T, D)
        out = out.transpose(1, 2).reshape(batch_size, -1, self.dim_model)

        # Slice Padding
        out = out[:, :out.size(1) - padding]

        # Output linear layer
        out = self.output_layer(out)

        return out, att_w.detach()


class LocalMultiHeadAttention(MultiHeadAttention):
    """Local Multi-Head Attention Layer

    Local multi-head attention restricts the attended positions to a local neighborhood
    around the query position. This is achieved by segmenting the hidden sequence into
    non overlapping blocks of size key and performing scaled dot-product attention in
    parallel for each of these blocks.

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        kernel_size: attention kernel size / window

    References:
        Image Transformer, Parmar et al.
        https://arxiv.org/abs/1802.05751

    """

    def __init__(self, dim_model, num_heads, kernel_size):
        super(LocalMultiHeadAttention, self).__init__(dim_model, num_heads)

        # Attention Params
        self.kernel_size = kernel_size  # key

    def forward(self, query, key, value, mask=None):

        # Batch size B
        batch_size = query.size(0)

        # Linear Layers
        query = self.query_layer(query)
        key = self.key_layer(key)
        value = self.value_layer(value)

        # Chunk Padding
        query, key, value, mask, padding = self.pad(query, key, value, mask, chunk_size=self.kernel_size)

        # Reshape and Transpose (B, T, D) -> (B, T//key, H, key, d)
        query = query.reshape(batch_size, -1, self.kernel_size, self.num_heads, self.dim_head).transpose(2, 3)
        key = key.reshape(batch_size, -1, self.kernel_size, self.num_heads, self.dim_head).transpose(2, 3)
        value = value.reshape(batch_size, -1, self.kernel_size, self.num_heads, self.dim_head).transpose(2, 3)

        # Att scores (B, T//key, H, key, key)
        att_scores = query.matmul(key.transpose(3, 4)) / key.shape[-1] ** 0.5

        # Apply mask
        if mask is not None:

            # Slice mask (B, 1, T, T) -> (B, T//key, 1, key, key)
            masks = []
            for m in range(mask.size(-1) // self.kernel_size):
                masks.append(mask[:, :, m * self.kernel_size: (m + 1) * self.kernel_size,
                             m * self.kernel_size: (m + 1) * self.kernel_size])
            mask = torch.stack(masks, dim=1)

            # Apply mask
            att_scores = att_scores.float() - mask.float() * 1e9

        # Att weights (B, T//key, H, key, key)
        att_w = att_scores.softmax(dim=-1)

        # Att output (B, T//key, H, key, d)
        out = att_w.matmul(value)

        # Transpose and Reshape (B, T//key, H, key, d) -> (B, T, D)
        out = out.transpose(2, 3).reshape(batch_size, -1, self.dim_model)

        # Slice Padding
        out = out[:, :out.size(1) - padding]

        # Output linear layer
        out = self.output_layer(out)

        return out, att_w.detach()


class StridedMultiHeadAttention(MultiHeadAttention):
    """Strided Multi-Head Attention Layer

    Strided multi-head attention performs global sequence downsampling by striding
    the attention query before applying scaled dot-product attention. This results in
    strided attention maps where query positions can attend to the entire sequence
    context to perform downsampling.

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        stride: query stride

    """

    def __init__(self, dim_model, num_heads, stride):
        super(StridedMultiHeadAttention, self).__init__(dim_model, num_heads)

        # Attention Params
        self.stride = stride  # S

    def forward(self, query, key, value, mask=None):
        # Query Subsampling (B, T, D) -> (B, T//S, D)
        query = query[:, ::self.stride]

        # Mask Subsampling (B, 1, T, T) -> (B, 1, T//S, T)
        if mask is not None:
            mask = mask[:, :, ::self.stride]

        # Multi-Head Attention
        return super(StridedMultiHeadAttention, self).forward(query, key, value, mask)


class StridedLocalMultiHeadAttention(MultiHeadAttention):
    """Strided Local Multi-Head Attention Layer

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        kernel_size: attention kernel size / window
        stride: query stride

    """

    def __init__(self, dim_model, num_heads, kernel_size, stride):
        super(StridedLocalMultiHeadAttention, self).__init__(dim_model, num_heads)

        # Assert
        assert kernel_size % stride == 0, "Attention kernel size has to be a multiple of attention stride"

        # Attention Params
        self.kernel_size = kernel_size  # key
        self.stride = stride  # S

    def forward(self, query, key, value, mask=None):

        # Batch size B
        batch_size = query.size(0)

        # Query Subsampling (B, T, D) -> (B, T//S, D)
        query = query[:, ::self.stride]

        # Linear Layers
        query = self.query_layer(query)
        key = self.key_layer(key)
        value = self.value_layer(value)

        # Chunk Padding
        query, key, value, mask, padding = self.pad(query, key, value, mask, chunk_size=self.kernel_size)

        # Reshape and Transpose (B, T//S, D) -> (B, T//key, H, key//S, d)
        query = query.reshape(batch_size, -1, self.kernel_size // self.stride, self.num_heads, self.dim_head).transpose(2, 3)
        # Reshape and Transpose (B, T, D) -> (B, T//key, H, key, d)
        key = key.reshape(batch_size, -1, self.kernel_size, self.num_heads, self.dim_head).transpose(2, 3)
        value = value.reshape(batch_size, -1, self.kernel_size, self.num_heads, self.dim_head).transpose(2, 3)

        # Att scores (B, T//key, H, key//S, key)
        att_scores = query.matmul(key.transpose(3, 4)) / key.shape[-1] ** 0.5

        # Apply mask
        if mask is not None:

            # Slice mask (B, 1, T, T) -> (B, T//key, 1, key, key)
            masks = []
            for m in range(mask.size(-1) // self.kernel_size):
                masks.append(mask[:, :, m * self.kernel_size: (m + 1) * self.kernel_size,
                             m * self.kernel_size: (m + 1) * self.kernel_size])
            mask = torch.stack(masks, dim=1)

            # Subsample mask (B, T//key, 1, key, key) -> (B, T//key, 1, key//S, key)
            mask = mask[:, :, :, ::self.stride]

            # Apply mask
            att_scores = att_scores.float() - mask.float() * 1e9

        # Att weights (B, T//key, H, key//S, key)
        att_w = att_scores.softmax(dim=-1)

        # Att output (B, T//key, H, key//S, d)
        out = att_w.matmul(value)

        # Transpose and Reshape (B, T//key, H, key//S, d) -> (B, T//S, D)
        out = out.transpose(2, 3).reshape(batch_size, -1, self.dim_model)

        # Slice Padding
        out = out[:, :(out.size(1) - padding - 1) // self.stride + 1]

        # Output linear layer
        out = self.output_layer(out)

        return out, att_w.detach()


class MultiHeadLinearAttention(MultiHeadAttention):
    """Multi-Head Linear Attention

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads

    References:
        Efficient Attention: Attention with Linear Complexities, Shen et al.
        https://arxiv.org/abs/1812.01243

        Efficient conformer-based speech recognition with linear attention, Li et al.
        https://arxiv.org/abs/2104.06865

    """

    def __init__(self, dim_model, num_heads):
        super(MultiHeadLinearAttention, self).__init__(dim_model, num_heads)

    def forward(self, query, key, value):
        # Batch size B
        batch_size = query.size(0)

        # Linear Layers
        query = self.query_layer(query)
        key = self.key_layer(key)
        value = self.value_layer(value)

        # Reshape and Transpose (B, T, D) -> (B, N, T, d)
        query = query.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # Global Context Vector (B, N, d, d)
        KV = (key / key.shape[-1] ** (1.0 / 4.0)).softmax(dim=-2).transpose(2, 3).matmul(value)

        # Attention Output (B, N, T, d)
        out = (query / query.shape[-1] ** (1.0 / 4.0)).softmax(dim=-1).matmul(KV)

        # Transpose and Reshape (B, N, T, d) -> (B, T, D)
        out = out.transpose(1, 2).reshape(batch_size, -1, self.dim_model)

        # Output linear layer
        out = self.output_layer(out)

        return out, KV.detach()


###############################################################################
# Multi-Head Self-Attention Layers with Relative Sinusoidal Poditional Encodings
###############################################################################

class RelPosMultiHeadSelfAttention(MultiHeadAttention):
    """Multi-Head Self-Attention Layer with Relative Sinusoidal Positional Encodings

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        causal: whether the attention is causal or unmasked
        max_pos_encoding: maximum relative distance between elements

    References:
        Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context, Dai et al.
        https://arxiv.org/abs/1901.02860

    """

    def __init__(self, dim_model, num_heads, causal, max_pos_encoding):
        super(RelPosMultiHeadSelfAttention, self).__init__(dim_model, num_heads)

        # Position Embedding Layer
        self.pos_layer = nn.Linear(self.dim_model, self.dim_model)
        self.causal = causal

        # Global content and positional bias
        self.u = nn.Parameter(torch.Tensor(self.dim_model))  # Content bias
        self.v = nn.Parameter(torch.Tensor(self.dim_model))  # Pos bias
        torch.nn.init.xavier_uniform_(self.u.reshape(self.num_heads, self.dim_head))  # glorot uniform
        torch.nn.init.xavier_uniform_(self.v.reshape(self.num_heads, self.dim_head))  # glorot uniform

        # Relative Sinusoidal Positional Encodings
        self.rel_pos_enc = RelativeSinusoidalPositionalEncoding(max_pos_encoding, self.dim_model, self.causal)

    def rel_to_abs(self, att_scores):

        """Relative to absolute position indexing

        Args:
            att_scores: absolute-by-relative indexed attention scores of shape
            (B, H, T, Th + 2*T-1) for full context and (B, H, T, Th + T) for causal context

        Return:
            att_scores: absolute-by-absolute indexed attention scores of shape (B, H, T, Th + T)

        References:
            causal context:
            Music Transformer, Huang et al.
            https://arxiv.org/abs/1809.04281

            full context:
            Attention Augmented Convolutional Networks, Bello et al.
            https://arxiv.org/abs/1904.09925

        """

        # Causal Context
        if self.causal:

            # Att Scores (B, H, T, Th + T)
            batch_size, num_heads, seq_length1, seq_length2 = att_scores.size()

            # Column Padding (B, H, T, 1 + Th + T)
            att_scores = F.pad(att_scores, pad=(1, 0), value=0)

            # Flatten (B, H, T + TTh + TT)
            att_scores = att_scores.reshape(batch_size, num_heads, -1)

            # Start Padding (B, H, Th + T + TTh + TT)
            att_scores = F.pad(att_scores, pad=(seq_length2 - seq_length1, 0), value=0)

            # Reshape (B, H, 1 + T, Th + T)
            att_scores = att_scores.reshape(batch_size, num_heads, 1 + seq_length1, seq_length2)

            # Slice (B, H, T, Th + T)
            att_scores = att_scores[:, :, 1:]

        # Full Context
        else:

            # Att Scores (B, H, T, Th + 2*T-1)
            batch_size, num_heads, seq_length1, seq_length2 = att_scores.size()

            # Column Padding (B, H, T, Th + 2*T)
            att_scores = F.pad(att_scores, pad=(0, 1), value=0)

            # Flatten (B, H, TTh + 2*TT)
            att_scores = att_scores.reshape(batch_size, num_heads, -1)

            # End Padding (B, H, TTh + 2*TT + Th + T - 1)
            att_scores = F.pad(att_scores, pad=(0, seq_length2 - seq_length1), value=0)

            # Reshape (B, H, T + 1, Th + 2*T-1)
            att_scores = att_scores.reshape(batch_size, num_heads, 1 + seq_length1, seq_length2)

            # Slice (B, H, T, Th + T)
            att_scores = att_scores[:, :, :seq_length1, seq_length1 - 1:]

        return att_scores

    def forward(self, query, key, value, mask=None, hidden=None):

        """Scaled Dot-Product Self-Attention with relative sinusoidal position encodings

        Args:
            query: Query of shape (B, T, D)
            key: Key of shape (B, T, D)
            value: Value of shape (B, T, D)
            mask: Optional position mask of shape (1 or B, 1 or H, 1 or T, 1 or T)
            hidden: Optional Key and Value hidden states for decoding

        Return:
            out: Attention output of shape (B, T, D)
            att_w: Attention weights of shape (B, H, T, Th + T)
            hidden: Key and value hidden states

        """

        # Batch size B
        batch_size = query.size(0)

        # Linear Layers
        query = self.query_layer(query)
        key = self.key_layer(key)
        value = self.value_layer(value)

        # Hidden State Provided
        if hidden:
            key = torch.cat([hidden["key"], key], dim=1)
            value = torch.cat([hidden["value"], value], dim=1)

        # Update Hidden State
        hidden = {"key": key.detach(), "value": value.detach()}

        # Add Bias
        Qu = query + self.u
        Qv = query + self.v

        # Relative Positional Embeddings (B, Th + 2*T-1, D) / (B, Th + T, D)
        E = self.pos_layer(self.rel_pos_enc(batch_size, query.size(1), key.size(1) - query.size(1)))

        # Reshape and Transpose (B, T, D) -> (B, H, T, d)
        Qu = Qu.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        Qv = Qv.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        # Reshape and Transpose (B, Th + T, D) -> (B, H, Th + T, d)
        key = key.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        # Reshape and Transpose (B, Th + 2*T-1, D) -> (B, H, Th + 2*T-1, d) / (B, Th + T, D) -> (B, H, Th + T, d)
        E = E.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # att_scores (B, H, T, Th + T)
        att_scores_K = Qu.matmul(key.transpose(2, 3))
        att_scores_E = self.rel_to_abs(Qv.matmul(E.transpose(2, 3)))
        att_scores = (att_scores_K + att_scores_E) / key.shape[-1] ** 0.5

        # Apply mask
        if mask is not None:
            att_scores += (mask * -1e9)

        # Att weights (B, H, T, Th + T)
        att_w = att_scores.softmax(dim=-1)

        # Att output (B, H, T, d)
        out = att_w.matmul(value)

        # Transpose and Reshape (B, H, T, d) -> (B, T, D)
        out = out.transpose(1, 2).reshape(batch_size, -1, self.dim_model)

        # Output linear layer
        out = self.output_layer(out)

        return out, att_w.detach(), hidden


class GroupedRelPosMultiHeadSelfAttention(RelPosMultiHeadSelfAttention):
    """Grouped Multi-Head Self-Attention Layer with Relative Sinusoidal Positional Encodings

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        causal: whether the attention is causal or unmasked
        max_pos_encoding: maximum relative distance between elements
        group_size: attention group size

    """

    def __init__(self, dim_model, num_heads, causal, max_pos_encoding, group_size):
        super(GroupedRelPosMultiHeadSelfAttention, self).__init__(dim_model, num_heads, causal, max_pos_encoding)

        # Attention Params
        self.group_size = group_size  # G
        self.dim_head = (self.group_size * dim_model) // self.num_heads  # d

        # Grouped Relative Sinusoidal Positional Encodings
        self.rel_pos_enc = GroupedRelativeSinusoidalPositionalEncoding(max_pos_encoding, self.dim_model,
                                                                       self.group_size, self.causal)

    def forward(self, query, key, value, mask=None, hidden=None):

        # Batch size B
        batch_size = query.size(0)

        # Linear Layers
        query = self.query_layer(query)
        key = self.key_layer(key)
        value = self.value_layer(value)

        # Hidden State Provided
        if hidden:
            Kh = torch.cat([hidden["key"], key], dim=1)
            Vh = torch.cat([hidden["value"], value], dim=1)
            key = torch.cat([hidden["key"][:, hidden["key"].size(1) % self.group_size:], key], dim=1)
            value = torch.cat([hidden["value"][:, hidden["value"].size(1) % self.group_size:], value], dim=1)

            # Update Hidden State
            hidden = {"key": Kh.detach(), "value": Vh.detach()}

        else:

            # Update Hidden State
            hidden = {"key": key.detach(), "value": value.detach()}

        # Chunk Padding
        query, key, value, mask, padding = self.pad(query, key, value, mask, chunk_size=self.group_size)

        # Add Bias
        Qu = query + self.u
        Qv = query + self.v

        # Relative Positional Embeddings (B, Th + 2*T-G, D) / (B, Th + T, D)
        E = self.pos_layer(self.rel_pos_enc(batch_size, query.size(1), key.size(1) - query.size(1)))

        # Reshape and Transpose (B, T, D) -> (B, H, T//G, d)
        Qu = Qu.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        Qv = Qv.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        # Reshape and Transpose (B, Th + T, D) -> (B, H, Th//G + T//G, d)
        key = key.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        # Reshape and Transpose (B, Th + 2*T-G, D) -> (B, H, Th//G + 2*T//G-1, d) / (B, Th + T, D) -> (B, H, Th//G + T//G, d)
        E = E.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # att_scores (B, H, T//G, Th//G + T//G)
        att_scores_K = Qu.matmul(key.transpose(2, 3))
        att_scores_E = self.rel_to_abs(Qv.matmul(E.transpose(2, 3)))
        att_scores = (att_scores_K + att_scores_E) / key.shape[-1] ** 0.5

        # Apply mask
        if mask is not None:
            # Slice Mask (B, 1, T, T) -> (B, 1, T//G, T//G)
            mask = mask[:, :, ::self.group_size, ::self.group_size]

            # Apply mask
            att_scores += (mask * -1e9)

        # Att weights (B, H, T//G, Th//G + T//G)
        att_w = att_scores.softmax(dim=-1)

        # Att output (B, H, T//G, d)
        out = att_w.matmul(value)

        # Transpose and Reshape (B, H, T//G, d) -> (B, T, D)
        out = out.transpose(1, 2).reshape(batch_size, -1, self.dim_model)

        # Slice Padding
        out = out[:, :out.size(1) - padding]

        # Output linear layer
        out = self.output_layer(out)

        return out, att_w.detach(), hidden


class LocalRelPosMultiHeadSelfAttention(RelPosMultiHeadSelfAttention):
    """Local Multi-Head Self-Attention with Relative Sinusoidal Positional Encodings

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        causal: whether the attention is causal or unmasked
        kernel_size: attention kernel size / window

    References:
        Music Transformer, Huang et al.
        https://arxiv.org/abs/1809.04281

    """

    def __init__(self, dim_model, num_heads, causal, kernel_size):
        super(LocalRelPosMultiHeadSelfAttention, self).__init__(dim_model, num_heads, causal, kernel_size)

        # Attention Params
        self.kernel_size = kernel_size  # key

    def rel_to_abs(self, att_scores):

        """Relative to absolute position indexing

        Args:
            att_scores: absolute-by-relative indexed attention scores of shape
            (B, N, T, 2 * key - 1) for full context and (B, H, T, key) for causal context

        Return:
            att_scores: absolute-by-absolute indexed attention scores of shape (B, T//key, H, key, key)

        References:
            Causal context:
            Music Transformer, Huang et al.
            https://arxiv.org/abs/1809.04281
        """

        # Causal Context
        if self.causal:

            # Att Scores (B, H, T, key)
            batch_size, num_heads, seq_length1, seq_length2 = att_scores.size()

            # Reshape (B, T//key, H, key, key)
            att_scores = att_scores.reshape(batch_size, -1, self.num_heads, self.kernel_size, self.kernel_size)

            # Column Padding (B, T//key, H, key, 1 + key)
            att_scores = F.pad(att_scores, pad=(1, 0), value=0)

            # Reshape (B, T//key, H, 1 + key, key)
            att_scores = att_scores.reshape(batch_size, -1, self.num_heads, self.kernel_size + 1, self.kernel_size)

            # Slice (B, T//key, H, key, key)
            att_scores = att_scores[:, :, :, 1:]

        # Full Context
        else:

            # Att Scores (B, H, T, 2 * key - 1)
            batch_size, num_heads, seq_length1, seq_length2 = att_scores.size()

            # Reshape (B, T//key, H, key, 2 * key - 1)
            att_scores = att_scores.reshape(batch_size, -1, self.num_heads, self.kernel_size, seq_length2)

            # Column Padding (B, T//key, H, key, 2 * key)
            att_scores = F.pad(att_scores, pad=(0, 1), value=0)

            # Flatten (B, T//key, H, key * 2 * key)
            att_scores = att_scores.reshape(batch_size, -1, self.num_heads, 2 * self.kernel_size ** 2)

            # End Padding (B, T//key, H, key * 2 * key + key - 1)
            att_scores = F.pad(att_scores, pad=(0, self.kernel_size - 1), value=0)

            # Reshape (B, T//key, H, key + 1, 2 * key - 1)
            att_scores = att_scores.reshape(batch_size, -1, self.num_heads, self.kernel_size + 1, seq_length2)

            # Slice (B, T//key, H, key, key)
            att_scores = att_scores[:, :, :, :self.kernel_size, self.kernel_size - 1:]

        return att_scores

    def forward(self, query, key, value, mask=None, hidden=None):

        # Batch size B
        batch_size = query.size(0)

        # Linear Layers
        query = self.query_layer(query)
        key = self.key_layer(key)
        value = self.value_layer(value)

        # Chunk Padding
        query, key, value, mask, padding = self.pad(query, key, value, mask, chunk_size=self.kernel_size)

        # Add Bias
        Qu = query + self.u
        Qv = query + self.v

        # Relative Positional Embeddings (B, 2*key-1, D) / (B, key, D)
        E = self.pos_layer(self.rel_pos_enc(batch_size))

        # Reshape and Transpose (B, T, D) -> (B, H, T, d)
        Qv = Qv.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        # Reshape and Transpose (B, T, D) -> (B, T//key, H, key, d)
        Qu = Qu.reshape(batch_size, -1, self.kernel_size, self.num_heads, self.dim_head).transpose(2, 3)
        key = key.reshape(batch_size, -1, self.kernel_size, self.num_heads, self.dim_head).transpose(2, 3)
        value = value.reshape(batch_size, -1, self.kernel_size, self.num_heads, self.dim_head).transpose(2, 3)
        # Reshape and Transpose (B, 2*key-1, D) -> (B, H, 2*key-1, d) / (B, key, D) -> (B, H, key, d)
        E = E.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # att_scores (B, T//key, H, key, key)
        att_scores_K = Qu.matmul(key.transpose(3, 4))
        att_scores_E = self.rel_to_abs(Qv.matmul(E.transpose(2, 3)))
        att_scores = (att_scores_K + att_scores_E) / key.shape[-1] ** 0.5

        # Mask scores
        if mask is not None:

            # Diagonal Mask (B, 1, T, T) -> (B, T//key, 1, key, key)
            masks = []
            for m in range(mask.size(-1) // self.kernel_size):
                masks.append(mask[:, :, m * self.kernel_size: (m + 1) * self.kernel_size,
                             m * self.kernel_size: (m + 1) * self.kernel_size])
            mask = torch.stack(masks, dim=1)

            # Apply Mask
            att_scores = att_scores.float() - mask.float() * 1e9

        # Attention weights (B, T//key, H, key, key)
        att_w = att_scores.softmax(dim=-1)

        # Attention output (B, T//key, H, key, d)
        out = att_w.matmul(value)

        # Transpose and Reshape (B, T//key, H, key, d) -> (B, T, D)
        out = out.transpose(2, 3).reshape(batch_size, -1, self.dim_model)

        # Slice Padding
        out = out[:, :out.size(1) - padding]

        # Output linear layer
        out = self.output_layer(out)

        return out, att_w.detach(), hidden


class StridedRelPosMultiHeadSelfAttention(RelPosMultiHeadSelfAttention):
    """Strided Multi-Head Self-Attention with Relative Sinusoidal Positional Encodings

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        causal: whether the attention is causal or unmasked
        max_pos_encoding: maximum relative distance between elements
        stride: query stride
    """

    def __init__(self, dim_model, num_heads, causal, max_pos_encoding, stride):
        super(StridedRelPosMultiHeadSelfAttention, self).__init__(dim_model, num_heads, causal, max_pos_encoding)

        # Attention Params
        self.stride = stride  # S

    def rel_to_abs(self, att_scores):

        """Relative to absolute position indexing

        Args:
            att_scores: absolute-by-relative indexed attention scores of shape
            (B, H, T//S, Th + 2 * T - 1) for full context and (B, H, T//S, Th + T) for causal context

        Return:
            att_scores: absolute-by-absolute indexed attention scores of shape (B, H, T//S,Th + T)

        """

        # Causal Context
        if self.causal:

            # Att Scores (B, H, T // S, Th + T)
            batch_size, num_heads, seq_length1, seq_length2 = att_scores.size()

            # Column Padding (B, H, T // S, Th + T + S)
            att_scores = F.pad(att_scores, pad=(1, self.stride - 1), value=0)

            # Flatten (B, H, TTh//S + TT//S + T)
            att_scores = att_scores.reshape(batch_size, num_heads, -1)

            # Start Padding (B, H, TTh//S + TT//S + T + Th)
            att_scores = F.pad(att_scores, pad=(seq_length2 - self.stride * seq_length1, 0), value=0)

            # Reshape (B, H, 1 + T // S, Th + T)
            att_scores = att_scores.reshape(batch_size, num_heads, seq_length1 + 1, seq_length2)

            # Slice (B, H, T // S, Th + T)
            att_scores = att_scores[:, :, 1:]

        # Full Context
        else:

            # Att Scores (B, H, T // S, Th + 2*T-1)
            batch_size, num_heads, seq_length1, seq_length2 = att_scores.size()

            # Column Padding (B, H, T // S, Th + 2*T-1 + S)
            att_scores = F.pad(att_scores, pad=(0, self.stride), value=0)

            # Flatten (B, H, TTh//S + 2*TT//S - T//S + T)
            att_scores = att_scores.reshape(batch_size, num_heads, -1)

            # End Padding (B, H, TTh//S + 2*TT//S - T//S + Th + 2T-1)
            att_scores = F.pad(att_scores, pad=(0, seq_length2 - seq_length1 * self.stride), value=0)

            # Reshape (B, H, T//S + 1, Th + 2*T-1)
            att_scores = att_scores.reshape(batch_size, num_heads, seq_length1 + 1, seq_length2)

            # Slice (B, H, T // S, Th + T)
            att_scores = att_scores[:, :, :seq_length1, seq_length1 * self.stride - 1:]

        return att_scores

    def forward(self, query, key, value, mask=None, hidden=None):

        # Batch size B
        batch_size = query.size(0)

        # Linear Layers
        query = self.query_layer(query)
        key = self.key_layer(key)
        value = self.value_layer(value)

        # Hidden State Provided
        if hidden:
            key = torch.cat([hidden["key"], key], dim=1)
            value = torch.cat([hidden["value"], value], dim=1)

        # Update Hidden State
        hidden = {"key": key.detach(), "value": value.detach()}

        # Chunk Padding
        query, key, value, mask, _ = self.pad(query, key, value, mask, chunk_size=self.stride)

        # Query Subsampling (B, T, D) -> (B, T//S, D)
        query = query[:, ::self.stride]

        # Add Bias
        Qu = query + self.u
        Qv = query + self.v

        # Relative Positional Embeddings (B, Th + 2*T-1, D) / (B, Th + T, D)
        E = self.pos_layer(self.rel_pos_enc(batch_size, self.stride * query.size(1), key.size(1) - self.stride * query.size(1)))

        # Reshape and Transpose (B, T//S, D) -> (B, H, T//S, d)
        Qu = Qu.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        Qv = Qv.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        # Reshape and Transpose (B, Th + T, D) -> (B, H, Th + T, d)
        key = key.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        # Reshape and Transpose (B, Th + 2*T-1, D) -> (B, H, Th + 2*T-1, d) / (B, Th + T, D) -> (B, H, Th + T, d)
        E = E.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # att_scores (B, H, T//S, Th + T)
        att_scores_K = Qu.matmul(key.transpose(2, 3))
        att_scores_E = self.rel_to_abs(Qv.matmul(E.transpose(2, 3)))
        att_scores = (att_scores_K + att_scores_E) / key.shape[-1] ** 0.5

        # Apply mask
        if mask is not None:

            # Mask Subsampling (B, 1, T, T) -> (B, 1, T//S, T)
            if mask is not None:
                mask = mask[:, :, ::self.stride]

            # Apply mask
            att_scores += (mask * -1e9)

        # Att weights (B, H, T//S, Th + T)
        att_w = att_scores.softmax(dim=-1)

        # Att output (B, H, T//S, d)
        out = att_w.matmul(value)

        # Transpose and Reshape (B, H, T//S, d) -> (B, T//S, D)
        out = out.transpose(1, 2).reshape(batch_size, -1, self.dim_model)

        # Output linear layer
        out = self.output_layer(out)

        return out, att_w.detach(), hidden


class StridedLocalRelPosMultiHeadSelfAttention(RelPosMultiHeadSelfAttention):
    """Strided Local Multi-Head Self-Attention with Relative Sinusoidal Positional Encodings

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        causal: whether the attention is causal or unmasked
        kernel_size: attention kernel size / window
        stride: query stride
    """

    def __init__(self, dim_model, num_heads, causal, kernel_size, stride):
        super(StridedLocalRelPosMultiHeadSelfAttention, self).__init__(dim_model, num_heads, causal, kernel_size)

        # Assert
        assert kernel_size % stride == 0, "Attention kernel size has to be a multiple of attention stride"

        # Attention Params
        self.kernel_size = kernel_size  # key
        self.stride = stride  # S

    def rel_to_abs(self, att_scores):

        """Relative to absolute position indexing

        Args:
            att_scores: absolute-by-relative indexed attention scores of shape
            (B, H, T//S, 2 * key - 1) for full context and (B, H, T//S, key) for causal context

        Return:
            att_scores: absolute-by-absolute indexed attention scores of shape (B, T//key, H, key//S, key)
        """

        # Causal Context
        if self.causal:

            # Att Scores (B, H, T//S, key)
            batch_size, num_heads, seq_length1, seq_length2 = att_scores.size()

            # Reshape (B, T//key, H, key//S, key)
            att_scores = att_scores.reshape(batch_size, -1, self.num_heads, self.kernel_size // self.stride,
                                            self.kernel_size)

            # Column Padding (B, T//key, H, key//S, key + S)
            att_scores = F.pad(att_scores, pad=(1, self.stride - 1), value=0)

            # Reshape (B, T//key, H, 1 + key//S, key)
            att_scores = att_scores.reshape(batch_size, -1, self.num_heads, self.kernel_size // self.stride + 1,
                                            self.kernel_size)

            # Slice (B, T//key, H, key//S, key)
            att_scores = att_scores[:, :, :, 1:]

        # Full Context
        else:

            # Att Scores (B, H, T//S, 2*key-1)
            batch_size, num_heads, seq_length1, seq_length2 = att_scores.size()

            # Reshape (B, T//key, H, key//S, 2*key-1)
            att_scores = att_scores.reshape(batch_size, -1, self.num_heads, self.kernel_size // self.stride,
                                            seq_length2)

            # Column Padding (B, T//key, H, key//S, 2*key-1 + S)
            att_scores = F.pad(att_scores, pad=(0, self.stride), value=0)

            # Flatten (B, T//key, H, 2KK//S - key//S + key)
            att_scores = att_scores.reshape(batch_size, -1, self.num_heads,
                                            self.kernel_size // self.stride * (2 * self.kernel_size - 1 + self.stride))

            # End Padding (B, T//key, H, 2KK//S - key//S + 2K-1)
            att_scores = F.pad(att_scores, pad=(0, self.kernel_size - 1), value=0)

            # Reshape (B, T//key, H, key//S + 1, 2*key-1)
            att_scores = att_scores.reshape(batch_size, -1, self.num_heads, self.kernel_size // self.stride + 1,
                                            seq_length2)

            # Slice (B, T//key, H, key//S, key)
            att_scores = att_scores[:, :, :, :self.kernel_size // self.stride, self.kernel_size - 1:]

        return att_scores

    def forward(self, query, key, value, mask=None, hidden=None):

        # Batch size B
        batch_size = query.size(0)

        # Chunk Padding
        query, key, value, mask, padding = self.pad(query, key, value, mask, chunk_size=self.kernel_size)

        # Query Subsampling (B, T, D) -> (B, T//S, D)
        query = query[:, ::self.stride]

        # Linear Layers
        query = self.query_layer(query)
        key = self.key_layer(key)
        value = self.value_layer(value)

        # Add Bias
        Qu = query + self.u
        Qv = query + self.v

        # Relative Positional Embeddings (B, 2*key-1, D) / (B, key, D)
        E = self.pos_layer(self.rel_pos_enc(batch_size))

        # Reshape and Transpose (B, T//S, D) -> (B, H, T//S, d)
        Qv = Qu.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        # Reshape and Transpose (B, T//S, D) -> (B, T//key, H, key//S, d)
        Qu = Qv.reshape(batch_size, -1, self.kernel_size // self.stride, self.num_heads, self.dim_head).transpose(2, 3)
        # Reshape and Transpose (B, T, D) -> (B, T//key, H, key, d)
        key = key.reshape(batch_size, -1, self.kernel_size, self.num_heads, self.dim_head).transpose(2, 3)
        value = value.reshape(batch_size, -1, self.kernel_size, self.num_heads, self.dim_head).transpose(2, 3)
        # Reshape and Transpose (B, 2*key-1, D) -> (B, H, 2*key-1, d) / (B, key, D) -> (B, H, key, d)
        E = E.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # att_scores (B, T//key, H, key//S, key)
        att_scores_K = Qu.matmul(key.transpose(3, 4))
        att_scores_E = self.rel_to_abs(Qv.matmul(E.transpose(2, 3)))
        att_scores = (att_scores_K + att_scores_E) / key.shape[-1] ** 0.5

        # Mask scores
        if mask is not None:

            # Diagonal Mask (B, 1, T, T) -> (B, T//key, 1, key, key)
            masks = []
            for m in range(mask.size(-1) // self.kernel_size):
                masks.append(mask[:, :, m * self.kernel_size: (m + 1) * self.kernel_size,
                             m * self.kernel_size: (m + 1) * self.kernel_size])
            mask = torch.stack(masks, dim=1)

            # Stride Mask (B, T//key, 1, key, key) -> (B, T//key, 1, key//S, key)
            mask = mask[:, :, :, ::self.stride]

            # Apply Mask
            att_scores = att_scores.float() - mask.float() * 1e9

        # Attention weights (B, T//key, H, key//S, key)
        att_w = att_scores.softmax(dim=-1)

        # Attention output (B, T//key, H, key//S, d)
        out = att_w.matmul(value)

        # Transpose and Reshape (B, T//key, H, key//S, d) -> (B, T//S, D)
        out = out.transpose(2, 3).reshape(batch_size, -1, self.dim_model)

        # Slice Padding
        out = out[:, :(self.stride * out.size(1) - padding - 1) // self.stride + 1]

        # Output linear layer
        out = self.output_layer(out)

        return out, att_w.detach(), hidden


###############################################################################
# Positional Encodings
###############################################################################

class SinusoidalPositionalEncoding(nn.Module):
    """

    Sinusoidal Positional Encoding

    Reference: "Attention Is All You Need" by Vaswani et al.
    https://arxiv.org/abs/1706.03762

    """

    def __init__(self, max_len, dim_model):
        super(SinusoidalPositionalEncoding, self).__init__()

        pos_encoding = torch.zeros(max_len, dim_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        i = torch.arange(0, dim_model // 2, dtype=torch.float).unsqueeze(0)
        angles = pos / 10000 ** (2 * i / dim_model)

        pos_encoding[:, 0::2] = angles.sin()
        pos_encoding[:, 1::2] = angles.cos()
        pos_encoding = pos_encoding.unsqueeze(0)

        self.register_buffer('pos_encoding', pos_encoding, persistent=False)

    def forward(self, batch_size=1, seq_len=None):

        # (B, T, D)
        if seq_len is not None:
            P = self.pos_encoding[:, :seq_len]

        # (B, Tmax, D)
        else:
            P = self.pos_encoding

        return P.repeat(batch_size, 1, 1)


class RelativeSinusoidalPositionalEncoding(nn.Module):
    """
        Relative Sinusoidal Positional Encoding

        Positional encoding for left context (sin) and right context (cos)
        Total context = 2 * max_len - 1
    """

    def __init__(self, max_len, dim_model, causal=False):
        super(RelativeSinusoidalPositionalEncoding, self).__init__()

        # PE
        pos_encoding = torch.zeros(2 * max_len - 1, dim_model)

        # Positions (max_len - 1, ..., max_len - 1)
        pos_left = torch.arange(start=max_len - 1, end=0, step=-1, dtype=torch.float)
        pos_right = torch.arange(start=0, end=-max_len, step=-1, dtype=torch.float)
        pos = torch.cat([pos_left, pos_right], dim=0).unsqueeze(1)

        # Angles
        angles = pos / 10000 ** (2 * torch.arange(0, dim_model // 2, dtype=torch.float).unsqueeze(0) / dim_model)

        # Rel Sinusoidal PE
        pos_encoding[:, 0::2] = angles.sin()
        pos_encoding[:, 1::2] = angles.cos()

        pos_encoding = pos_encoding.unsqueeze(0)

        self.register_buffer('pos_encoding', pos_encoding, persistent=False)
        self.max_len = max_len
        self.causal = causal

    def forward(self, batch_size=1, seq_len=None, hidden_len=0):

        # Causal Context
        if self.causal:

            # (B, Th + T, D)
            if seq_len is not None:
                R = self.pos_encoding[:, self.max_len - seq_len - hidden_len: self.max_len]

            # (B, Tmax, D)
            else:
                R = self.pos_encoding[:, :self.max_len]

        # Full Context
        else:

            # (B, Th + 2*T-1, D)
            if seq_len is not None:
                R = self.pos_encoding[:, self.max_len - seq_len - hidden_len: self.max_len - 1 + seq_len]

            # (B, 2*Tmax-1, D)
            else:
                R = self.pos_encoding

        return R.repeat(batch_size, 1, 1)


class GroupedRelativeSinusoidalPositionalEncoding(nn.Module):
    """
        Relative Sinusoidal Positional Encoding for grouped multi-head attention

        Positional encoding for left context (sin) and right context (cos)
        Total context = 2 * max_len - group_size
    """

    def __init__(self, max_len, dim_model, group_size=1, causal=False):
        super(GroupedRelativeSinusoidalPositionalEncoding, self).__init__()

        # PE
        pos_encoding = torch.zeros(2 * max_len - group_size % 2, dim_model)

        # Positions (max_len - 1, ..., max_len - 1)
        pos_left = torch.arange(start=max_len - 1, end=group_size % 2 - 1, step=-1, dtype=torch.float)
        pos_right = torch.arange(start=0, end=-max_len, step=-1, dtype=torch.float)
        pos = torch.cat([pos_left, pos_right], dim=0).unsqueeze(1)

        # Angles
        angles = pos / 10000 ** (2 * torch.arange(0, dim_model // 2, dtype=torch.float).unsqueeze(0) / dim_model)

        # Rel Sinusoidal PE
        pos_encoding[:, 0::2] = angles.sin()
        pos_encoding[:, 1::2] = angles.cos()

        pos_encoding = pos_encoding.unsqueeze(0)

        self.register_buffer('pos_encoding', pos_encoding, persistent=False)
        self.max_len = max_len
        self.causal = causal
        self.group_size = group_size

    def forward(self, batch_size=1, seq_len=None, hidden_len=0):

        # Causal Context
        if self.causal:

            # (B, Th + T, D)
            if seq_len is not None:
                R = self.pos_encoding[:, self.max_len - seq_len - hidden_len: self.max_len]

            # (B, Tmax, D)
            else:
                R = self.pos_encoding[:, :self.max_len]
        else:

            # (B, Th + 2*T-G, D)
            if seq_len is not None:
                R = self.pos_encoding[:,
                    self.max_len - seq_len + self.group_size // 2 - hidden_len: self.max_len - self.group_size % 2 + seq_len - self.group_size // 2]

            # (B, 2*Tmax-G, D)
            else:
                R = self.pos_encoding

        return R.repeat(batch_size, 1, 1)


class MultiHeadSelfAttentionModule(nn.Module):
    """Multi-Head Self-Attention Module

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        dropout: residual dropout probability
        max_pos_encoding: maximum position
        relative_pos_enc: whether to use relative postion embedding
        causal: True for causal attention with masked future context
        group_size: Attention group size
        kernel_size: Attention kernel size
        stride: Query stride
        linear_att: whether to use multi-head linear self-attention

    """

    def __init__(self,
                 dim_model,
                 num_heads,
                 dropout,
                 max_pos_encoding,
                 relative_pos_enc=False,
                 causal=False,
                 group_size=1,
                 kernel_size=None,
                 stride=1,
                 linear_att=False):
        super(MultiHeadSelfAttentionModule, self).__init__()

        # Assert
        assert not (group_size > 1 and kernel_size is not None), "Local grouped attention not implemented"
        assert not (group_size > 1 and stride > 1), "Strided grouped attention not implemented"
        assert not (linear_att and relative_pos_enc), "Linear attention requires absolute positional encodings"

        # Pre Norm
        # self.norm = nn.LayerNorm(dim_model, eps=1e-6)

        # Multi-Head Linear Attention
        if linear_att:
            self.mhsa = MultiHeadLinearAttention(dim_model, num_heads)

        # Grouped Multi-Head Self-Attention
        elif group_size > 1:
            if relative_pos_enc:
                self.mhsa = GroupedRelPosMultiHeadSelfAttention(dim_model, num_heads, causal, max_pos_encoding,
                                                                group_size)
            else:
                self.mhsa = GroupedMultiHeadAttention(dim_model, num_heads, group_size)

        # Local Multi-Head Self-Attention
        elif kernel_size is not None and stride == 1:
            if relative_pos_enc:
                self.mhsa = LocalRelPosMultiHeadSelfAttention(dim_model, num_heads, causal, kernel_size)
            else:
                self.mhsa = LocalMultiHeadAttention(dim_model, num_heads, kernel_size)

        # Strided Multi-Head Self-Attention
        elif kernel_size is None and stride > 1:
            if relative_pos_enc:
                self.mhsa = StridedRelPosMultiHeadSelfAttention(dim_model, num_heads, causal, max_pos_encoding, stride)
            else:
                self.mhsa = StridedMultiHeadAttention(dim_model, num_heads, stride)

        # Strided Local Multi-Head Self-Attention
        elif stride > 1 and kernel_size is not None:
            if relative_pos_enc:
                self.mhsa = StridedLocalRelPosMultiHeadSelfAttention(dim_model, num_heads, causal, kernel_size, stride)
            else:
                self.mhsa = StridedLocalMultiHeadAttention(dim_model, num_heads, kernel_size, stride)

        # Multi-Head Self-Attention
        else:
            if relative_pos_enc:
                self.mhsa = RelPosMultiHeadSelfAttention(dim_model, num_heads, causal, max_pos_encoding)
            else:
                self.mhsa = MultiHeadAttention(dim_model, num_heads)

        # Dropout
        # self.dropout = nn.Dropout(Pdrop)

        # Module Params
        self.rel_pos_enc = relative_pos_enc
        self.linear_att = linear_att

    def forward(self, x, mask=None, hidden=None):

        x = x.transpose(0, 1)
        if mask is not None:
            mask = mask.view(mask.size(0), 1, 1, mask.size(-1))

        # Multi-Head Self-Attention
        if self.linear_att:
            x, attention = self.mhsa(x, x, x)
        elif self.rel_pos_enc:
            x, attention, hidden = self.mhsa(x, x, x, mask, hidden)
        else:
            x, attention = self.mhsa(x, x, x, mask)

        # Dropout
        # x = self.dropout(x)

        x = x.transpose(0, 1)
        return x, attention
