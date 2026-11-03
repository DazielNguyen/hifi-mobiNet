"""A native 16 kHz WAV must never be relabeled as the Medium model's 22.05 kHz."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("harvard_score", ROOT / "scripts/evaluation/harvard/score.py")
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


@pytest.mark.parametrize("sr", [16000, 22050])
def test_native_rate_and_waveform_are_preserved(tmp_path, sr):
    path = tmp_path / "wave.wav"
    samples = np.linspace(-0.5, 0.5, sr, dtype=np.float32)
    sf.write(path, samples, sr, subtype="FLOAT")
    actual, rate = score.read_wave(path, sr)
    assert rate == sr
    assert np.array_equal(actual, samples)
    assert len(actual) / rate == 1.0


def test_wrong_rate_fails_before_scoring(tmp_path):
    path = tmp_path / "wave.wav"
    sf.write(path, np.ones(16000, dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="sample rate"):
        score.read_wave(path, 22050)


@pytest.mark.parametrize("samples", [np.zeros((100, 2)), np.array([0.0, np.nan]), np.array([])])
def test_invalid_audio_fails(tmp_path, samples):
    path = tmp_path / "wave.wav"
    sf.write(path, samples, 16000, subtype="FLOAT")
    with pytest.raises(ValueError, match="nonempty finite mono"):
        score.read_wave(path, 16000)
