"""Linear-magnitude spectrogram via STFT (VITS's posterior encoder input),
plus the mel-spectrogram conversion used for the training-time mel loss
(comparing generated vs. real audio in mel space, not raw waveform space).
"""
import logging

import torch
from librosa.filters import mel as librosa_mel_fn

_LOGGER = logging.getLogger("banhmi_train.mel_processing")

_hann_windows: dict = {}
_mel_filterbanks: dict = {}


def _get_hann_window(win_size: int, y: torch.Tensor) -> torch.Tensor:
    key = (win_size, y.dtype, y.device)
    window = _hann_windows.get(key)
    if window is None:
        window = torch.hann_window(win_size, dtype=y.dtype, device=y.device)
        _hann_windows[key] = window
    return window


def spectrogram_torch(
    y: torch.Tensor, n_fft: int, hop_size: int, win_size: int
) -> torch.Tensor:
    if torch.max(torch.abs(y)) > 1.0:
        _LOGGER.warning("Audio exceeds the expected [-1, 1] range")

    # cuFFT (what torch.stft dispatches to on CUDA) rejects bf16/fp16 tensors
    # outright ("cuFFT doesn't support tensor of type: BFloat16"), so under
    # bf16 mixed precision (this project's training mode, --precision bf16)
    # this crashes unless the STFT itself runs in fp32. Autocast disabled + explicit .float() forces
    # that regardless of the ambient precision context; the magnitude output
    # then upcasts whatever downstream op consumes it back to bf16 as normal.
    with torch.autocast(device_type=y.device.type, enabled=False):
        y = y.float()
        window = _get_hann_window(win_size, y)

        pad = (n_fft - hop_size) // 2
        y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)

        spec = torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=window,
            center=False,
            pad_mode="reflect",
            onesided=True,
            return_complex=True,
        )

        return torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)


def _get_mel_filterbank(
    n_fft: int, num_mels: int, sample_rate: int, fmin: float, fmax, spec: torch.Tensor
) -> torch.Tensor:
    key = (n_fft, num_mels, sample_rate, fmin, fmax, spec.dtype, spec.device)
    fb = _mel_filterbanks.get(key)
    if fb is None:
        fb = torch.from_numpy(
            librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        ).to(dtype=spec.dtype, device=spec.device)
        _mel_filterbanks[key] = fb
    return fb


def spec_to_mel_torch(
    spec: torch.Tensor, n_fft: int, num_mels: int, sample_rate: int, fmin: float, fmax
) -> torch.Tensor:
    """Linear spectrogram -> mel, with log-magnitude compression."""
    mel_fb = _get_mel_filterbank(n_fft, num_mels, sample_rate, fmin, fmax, spec)
    mel = torch.matmul(mel_fb, spec)
    return torch.log(torch.clamp(mel, min=1e-5))


def mel_spectrogram_torch(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sample_rate: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax,
) -> torch.Tensor:
    """Waveform -> mel spectrogram (spectrogram_torch + spec_to_mel_torch,
    for the generator's own audio which has no cached linear spec)."""
    spec = spectrogram_torch(y, n_fft, hop_size, win_size)
    return spec_to_mel_torch(spec, n_fft, num_mels, sample_rate, fmin, fmax)
