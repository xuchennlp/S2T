import os.path as op
from typing import BinaryIO, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio

def get_waveform(
        path_or_fp: Union[str, BinaryIO],
        normalization=True,
        offset=None,
        size=None
) -> Tuple[np.ndarray, int]:
    """Get the waveform and sample rate of a 16-bit mono-channel WAV or FLAC.

    Args:
        path_or_fp (str or BinaryIO): the path or file-like object
        normalization (bool): Normalize values to [-1, 1] (Default: True)
    """
    if isinstance(path_or_fp, str):
        ext = op.splitext(op.basename(path_or_fp))[1]
        if ext not in {".flac", ".wav"}:
            raise ValueError(f"Unsupported audio format: {ext}")

    if offset is not None and size is not None:
        waveform, sample_rate = torchaudio.load(path_or_fp, frame_offset=offset, num_frames=size)
    else:
        waveform, sample_rate = torchaudio.load(path_or_fp)
    waveform = waveform.squeeze().numpy()

    if not normalization:
        waveform *= 2 ** 15  # denormalized to 16-bit signed integers

    return waveform, sample_rate


def _get_kaldi_fbank(waveform, sample_rate, n_bins=80) -> Optional[np.ndarray]:
    """Get mel-filter bank features via PyKaldi."""
    try:
        from kaldi.feat.mel import MelBanksOptions
        from kaldi.feat.fbank import FbankOptions, Fbank
        from kaldi.feat.window import FrameExtractionOptions
        from kaldi.matrix import Vector

        mel_opts = MelBanksOptions()
        mel_opts.num_bins = n_bins
        frame_opts = FrameExtractionOptions()
        frame_opts.samp_freq = sample_rate
        opts = FbankOptions()
        opts.mel_opts = mel_opts
        opts.frame_opts = frame_opts
        fbank = Fbank(opts=opts)
        features = fbank.compute(Vector(waveform), 1.0).numpy()
        return features
    except ImportError:
        return None


def _get_torchaudio_fbank(waveform, sample_rate, n_bins=80) -> Optional[np.ndarray]:
    """Get mel-filter bank features via TorchAudio."""
    try:
        import torchaudio.compliance.kaldi as ta_kaldi
        import torchaudio.sox_effects as ta_sox

        if not isinstance(waveform, torch.Tensor):
            waveform = torch.from_numpy(waveform)
        if len(waveform.shape) == 1:
            # Mono channel: D -> 1 x D
            waveform = waveform.unsqueeze(0)
        else:
            # Merge multiple channels to one: D x C -> 1 x D
            waveform, _ = ta_sox.apply_effects_tensor(waveform.T, sample_rate, [['channels', '1']])

        features = ta_kaldi.fbank(
            waveform, num_mel_bins=n_bins, sample_frequency=sample_rate
        )
        return features.numpy()
    except ImportError:
        return None


def get_fbank(
        path_or_fp: Union[str, BinaryIO],
        n_bins=80,
        offset=None,
        size=None,
) -> np.ndarray:
    """Get mel-filter bank features via PyKaldi or TorchAudio. Prefer PyKaldi
    (faster CPP implementation) to TorchAudio (Python implementation). Note that
    Kaldi/TorchAudio requires 16-bit signed integers as inputs and hence the
    waveform should not be normalized."""
    sound, sample_rate = get_waveform(path_or_fp, normalization=False, offset=offset, size=size)

    features = _get_kaldi_fbank(sound, sample_rate, n_bins)
    if features is None:
        features = _get_torchaudio_fbank(sound, sample_rate, n_bins)
    if features is None:
        raise ImportError(
            "Please install pyKaldi or torchaudio to enable "
            "online filterbank feature extraction"
        )

    return features


def get_fbank_with_perturb(waveform, sample_rate=16000, n_bins=80):
    import random

    speed = random.choice([0.9, 1.0, 1.1])
    if speed != 1.0:
        waveform = torch.from_numpy(waveform).float().unsqueeze(0)
        waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, sample_rate,
            [['speed', str(speed)], ['rate', str(sample_rate)]])
        waveform = waveform.squeeze()

    features = _get_kaldi_fbank(waveform, sample_rate)
    if features is None:
        features = _get_torchaudio_fbank(waveform, sample_rate, n_bins)

    return features
