"""Normalizes audio and caches its waveform + spectrogram tensors, keyed by
a hash of the source file's path so re-running preprocessing skips work
that's already cached.

No silence trimming: LJSpeech clips are already tightly trimmed by the
dataset's own creators (measured empirically -- VAD found ~0.000s of
leading silence and only chunk-rounding noise at the trailing edge across
a random sample), so a VAD pass here would cost a per-utterance ONNX
inference loop for no actual effect. If a future dataset isn't pre-trimmed,
silence trimming would need to be reintroduced.
"""
import hashlib
from pathlib import Path
from typing import Tuple, Union

import librosa
import numpy as np
import pyworld
import torch

from ..mel_processing import spectrogram_torch

_FILTER_LENGTH = 1024
_WINDOW_LENGTH = 1024
_HOP_LENGTH = 256


def _cache_paths(audio_path: Path, cache_dir: Path) -> Tuple[Path, Path]:
    cache_id = hashlib.sha256(str(audio_path.absolute()).encode("utf-8")).hexdigest()
    return cache_dir / f"{cache_id}.pt", cache_dir / f"{cache_id}.spec.pt"


def cache_norm_audio(
    audio_path: Union[str, Path],
    cache_dir: Union[str, Path],
    sample_rate: int,
    ignore_cache: bool = False,
) -> Tuple[Path, Path]:
    audio_path = Path(audio_path)
    cache_dir = Path(cache_dir)
    wav_path, spec_path = _cache_paths(audio_path, cache_dir)

    wav_tensor = None
    if ignore_cache or not wav_path.exists():
        # librosa already normalizes to float32 in [-1, 1]. Uses the
        # FloatTensor constructor rather than torch.from_numpy(): the latter
        # crashes ("Numpy is not available") under a torch/numpy ABI
        # mismatch that FloatTensor's copy-based path tolerates.
        samples, _sr = librosa.load(audio_path, sr=sample_rate)
        wav_tensor = torch.FloatTensor(samples).unsqueeze(0)
        torch.save(wav_tensor, wav_path)

    if ignore_cache or not spec_path.exists():
        if wav_tensor is None:
            wav_tensor = torch.load(wav_path)

        spec_tensor = spectrogram_torch(
            wav_tensor, _FILTER_LENGTH, _HOP_LENGTH, _WINDOW_LENGTH
        ).squeeze(0)
        torch.save(spec_tensor, spec_path)

    return wav_path, spec_path


def _interpolate_unvoiced(f0: "np.ndarray") -> "np.ndarray":
    """Fill F0=0 (unvoiced/silent) frames via linear interpolation in log-F0
    space, so the predictor isn't trained to regress towards zero in gaps."""
    voiced = f0 > 0
    if voiced.sum() in (0, len(f0)):
        return f0
    idx = np.arange(len(f0))
    log_f0 = np.log(f0 + 1e-8)
    log_f0_interp = np.interp(idx, idx[voiced], log_f0[voiced])
    f0_interp = np.exp(log_f0_interp)
    f0_interp[voiced] = f0[voiced]
    return f0_interp


def cache_f0(
    audio_norm_path: Union[str, Path],
    cache_dir: Union[str, Path],
    sample_rate: int,
    num_mel_frames: int,
    ignore_cache: bool = False,
) -> Path:
    """Extract a per-frame F0 contour (pyworld DIO + StoneMask) aligned to the
    same number of frames as the mel-spectrogram/duration ground truth.

    pyworld's frame count is consistently 1 frame longer than the
    spectrogram's for this hop_length/sample_rate combination (verified
    empirically, matching EdgeTTS's own cache_f0), so the contour is
    truncated/padded to num_mel_frames to align.
    """
    audio_norm_path = Path(audio_norm_path)
    cache_dir = Path(cache_dir)
    cache_id = audio_norm_path.stem
    f0_path = cache_dir / f"{cache_id}.f0.pt"

    if not ignore_cache and f0_path.exists():
        return f0_path

    audio_norm_tensor = torch.load(audio_norm_path)
    audio64 = audio_norm_tensor.squeeze(0).numpy().astype(np.float64)

    frame_period_ms = _HOP_LENGTH / sample_rate * 1000.0
    f0, t = pyworld.dio(audio64, sample_rate, frame_period=frame_period_ms)
    f0 = pyworld.stonemask(audio64, f0, t, sample_rate)
    f0 = _interpolate_unvoiced(f0)

    if len(f0) > num_mel_frames:
        f0 = f0[:num_mel_frames]
    elif len(f0) < num_mel_frames:
        f0 = np.pad(f0, (0, num_mel_frames - len(f0)), mode="edge")

    f0_tensor = torch.FloatTensor(f0)
    torch.save(f0_tensor, f0_path)

    return f0_path
