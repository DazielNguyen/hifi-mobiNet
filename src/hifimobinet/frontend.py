"""Use the original Banhmi frontend, with no fallback text normalization."""
from __future__ import annotations
import threading

MAX_TEXT_CHARS = 300
MAX_PHONEME_IDS = 800
_LOCK = threading.Lock()


def require_frontend():
    try:
        import banhmi_phonemize
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "Banhmi phonemizer unavailable. Build the preserved native frontend; "
            "audio comparison works without it. See docs/setup.md."
        ) from error
    return banhmi_phonemize


def encode(text: str) -> list[int]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter non-empty English text")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"Text exceeds the {MAX_TEXT_CHARS}-character limit")
    if any(ord(c) < 32 and c not in "\n\t" for c in text):
        raise ValueError("Control characters are not supported")
    frontend = require_frontend()
    # Match say.py and preprocess/worker.py: flatten sentences before encoding.
    # NFD, terminator punctuation, unsupported-symbol behavior and blank insertion
    # remain inside the unmodified frontend implementation.
    with _LOCK:
        sentences = frontend.phonemize_espeak(text, "en-us")
        ids = frontend.phoneme_ids_espeak([p for sentence in sentences for p in sentence])
    if not 3 < len(ids) <= MAX_PHONEME_IDS:
        raise ValueError("Encoded text is empty or exceeds the phoneme-ID limit")
    if any(not isinstance(i, int) or not 0 <= i < 256 for i in ids):
        raise ValueError("Frontend returned an ID outside the 256-symbol vocabulary")
    return ids
