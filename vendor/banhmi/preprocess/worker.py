"""Per-utterance phonemize + audio-cache work, run in ProcessPoolExecutor
worker processes.

Only plain espeak phonemization is used here -- mixed_lang (Vietnamese
code-switch detection) is an inference-time concern only, since the
training corpus (LJSpeech) is pure English. See banhmi_train/say.py for the
inference-side use of mixed_lang.

Each worker process calls init_worker() once (via ProcessPoolExecutor's
`initializer=`) to set up its own eSpeak voice/casing, so those don't get
re-created for every single utterance.
"""
import logging
from pathlib import Path
from typing import Optional

from banhmi_phonemize import phonemize_espeak, phoneme_ids_espeak

import torch

from .dataset import Utterance
from .norm_audio import cache_f0, cache_norm_audio

_LOGGER = logging.getLogger("banhmi_train.preprocess.worker")

_language: str = "en-us"
_casing = None
_cache_dir: Optional[Path] = None
_sample_rate: int = 22050
_skip_audio: bool = False


def get_text_casing(casing: str):
    if casing == "lower":
        return str.lower

    if casing == "upper":
        return str.upper

    if casing == "casefold":
        return str.casefold

    return lambda s: s


def init_worker(
    language: str,
    text_casing: str,
    cache_dir: Path,
    sample_rate: int,
    skip_audio: bool,
) -> None:
    global _language, _casing, _cache_dir, _sample_rate, _skip_audio
    _language = language
    _casing = get_text_casing(text_casing)
    _cache_dir = cache_dir
    _sample_rate = sample_rate
    _skip_audio = skip_audio


def process_utterance(utt: Utterance) -> Optional[Utterance]:
    try:
        sentences_phonemes = phonemize_espeak(_casing(utt.text), _language)
        utt.phonemes = [p for sentence in sentences_phonemes for p in sentence]
        utt.phoneme_ids = phoneme_ids_espeak(
            utt.phonemes, missing_phonemes=utt.missing_phonemes
        )

        if not _skip_audio:
            utt.audio_norm_path, utt.audio_spec_path = cache_norm_audio(
                utt.audio_path, _cache_dir, _sample_rate
            )
            num_mel_frames = torch.load(utt.audio_spec_path).shape[-1]
            utt.audio_f0_path = cache_f0(
                utt.audio_norm_path, _cache_dir, _sample_rate, num_mel_frames
            )

        return utt
    except Exception:
        _LOGGER.exception("Failed to process utterance: %s", utt.text)
        return None
