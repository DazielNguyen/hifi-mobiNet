"""Converts floating-point model output into 16-bit PCM samples."""
import numpy as np

_INT16_PEAK = 32767.0


def audio_float_to_int16(audio: np.ndarray) -> np.ndarray:
    peak = max(float(np.max(np.abs(audio))), 0.01)
    scaled = audio * (_INT16_PEAK / peak)
    return np.clip(scaled, -_INT16_PEAK, _INT16_PEAK).astype(np.int16)
