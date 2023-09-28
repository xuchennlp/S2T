# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

from .squeeze_excitation import SEAttention
from .activations import swish, Swish
from .adaptive_input import AdaptiveInput
from .adaptive_softmax import AdaptiveSoftmax
from .attention import MultiHeadSelfAttentionModule
from .beamable_mm import BeamableMM
from .character_token_embedder import CharacterTokenEmbedder
from .downsample_convolution import DownSampleConvolutionModule
from .conv_tbc import ConvTBC
from .cross_entropy import cross_entropy
from .downsampled_multihead_attention import DownsampledMultiHeadAttention
from .dlcl import DynamicLinearCombination
from .dynamic_convolution import DynamicConv, DynamicConv1dTBC
from .dynamic_crf_layer import DynamicCRF
from .fairseq_dropout import FairseqDropout
from .fp32_group_norm import Fp32GroupNorm
from .gelu import gelu, gelu_accurate
from .grad_multiply import GradMultiply
from .gumbel_vector_quantizer import GumbelVectorQuantizer
from .kmeans_vector_quantizer import KmeansVectorQuantizer
from .layer_drop import LayerDropModuleList
from .layer_norm import Fp32LayerNorm, LayerNorm
from .learned_positional_embedding import LearnedPositionalEmbedding
from .lightweight_convolution import LightweightConv, LightweightConv1dTBC
from .linearized_convolution import LinearizedConvolution
from .local_multihead_attention import LocalMultiheadAttention
from .location_attention import LocationAttention
from .multihead_attention import MultiheadAttention
from .positional_embedding import PositionalEmbedding
from .reduced_multihead_attention import ReducedMultiheadAttention
from .rel_position_multihead_attention import RelPositionMultiheadAttention
from .relative_multihead_attention import RelativeMultiheadAttention
from .same_pad import SamePad
from .scalar_bias import ScalarBias
from .sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from .transformer_sentence_encoder_layer import TransformerSentenceEncoderLayer
from .transformer_sentence_encoder import TransformerSentenceEncoder
from .transpose_last import TransposeLast
from .unfold import unfold1d
from .transformer_layer import TransformerDecoderLayer, TransformerEncoderLayer
from .transformer_s2_layer import TransformerS2EncoderLayer, TransformerS2DecoderLayer
from .vggblock import VGGBlock
from .rotary_positional_embedding import RotaryPositionalEmbedding
from .positional_encoding import (
    PositionalEncoding,
    LegacyRelPositionalEncoding,
    RelPositionalEncoding,
)
from .espnet_multihead_attention import (
    ESPNETMultiHeadedAttention,
    RelPositionMultiHeadedAttention,
    ReducedRelPositionMultiHeadedAttention,
    LegacyRelPositionMultiHeadedAttention,
    RotaryPositionMultiHeadedAttention,
)
from .convolution import ConvolutionModule
from .s2t_transformer_layer import S2TTransformerEncoderLayer
from .s2t_transformer_s2_layer import S2TTransformerS2EncoderLayer
from .pds_layer import PDSTransformerEncoderLayer

__all__ = [
    "DynamicLinearCombination",
    "AdaptiveInput",
    "AdaptiveSoftmax",
    "BeamableMM",
    "CharacterTokenEmbedder",
    "S2TTransformerEncoderLayer",
    "S2TTransformerS2EncoderLayer",
    "ConvolutionModule",
    "ConvTBC",
    "cross_entropy",
    "DownSampleConvolutionModule",
    "DownsampledMultiHeadAttention",
    "DynamicConv1dTBC",
    "DynamicConv",
    "DynamicCRF",
    "FairseqDropout",
    "Fp32GroupNorm",
    "Fp32LayerNorm",
    "gelu",
    "gelu_accurate",
    "GradMultiply",
    "GumbelVectorQuantizer",
    "KmeansVectorQuantizer",
    "LayerDropModuleList",
    "LayerNorm",
    "LearnedPositionalEmbedding",
    "LightweightConv1dTBC",
    "LightweightConv",
    "LinearizedConvolution",
    "LocalMultiheadAttention",
    "MultiheadAttention",
    "MultiHeadSelfAttentionModule",
    "PositionalEmbedding",
    "PDSTransformerEncoderLayer",
    "ReducedMultiheadAttention",
    "RelPositionMultiheadAttention",
    "RelativeMultiheadAttention",
    "SamePad",
    "ScalarBias",
    "SEAttention",
    "SinusoidalPositionalEmbedding",
    "swish",
    "Swish",
    "TransformerSentenceEncoderLayer",
    "TransformerSentenceEncoder",
    "TransformerDecoderLayer",
    "TransformerEncoderLayer",
    "TransformerS2DecoderLayer",
    "TransformerS2EncoderLayer",
    "TransposeLast",
    "VGGBlock",
    "unfold1d",
    "ESPNETMultiHeadedAttention",
    "PositionalEmbedding",
    "PositionalEncoding",
    "LegacyRelPositionalEncoding",
    "RelPositionalEncoding",
    "RelPositionMultiHeadedAttention",
    "ReducedRelPositionMultiHeadedAttention",
    "LegacyRelPositionMultiHeadedAttention",
    "RotaryPositionalEmbedding",
    "RotaryPositionMultiHeadedAttention",
]
