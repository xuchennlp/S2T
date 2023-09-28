import numpy as np
from fairseq.data.audio.feature_transforms import (
    AudioFeatureTransform,
    register_audio_feature_transform,
)


@register_audio_feature_transform("utterance_cmvn")
class UtteranceCMVN(AudioFeatureTransform):
    """Utterance-level CMVN (cepstral mean and variance normalization)"""

    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        return UtteranceCMVN(
            _config.get("norm_means", True),
            _config.get("norm_vars", True),
            _config.get("cmvn_no_axis", False),
        )

    def __init__(self, norm_means=True, norm_vars=True, no_axis=False):
        self.norm_means, self.norm_vars, self.no_axis = norm_means, norm_vars, no_axis

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(norm_means={self.norm_means}, norm_vars={self.norm_vars}, no_axis={self.no_axis})"
        )

    def __call__(self, x):
        if self.no_axis:
            axis = None
        else:
            axis = 0
        mean = x.mean(axis=axis)
        square_sums = (x ** 2).sum(axis=axis)

        if self.norm_means:
            x = np.subtract(x, mean)
        if self.norm_vars:
            var = square_sums / x.shape[0] - mean ** 2
            std = np.sqrt(np.maximum(var, 1e-10))
            x = np.divide(x, std)

        return x
