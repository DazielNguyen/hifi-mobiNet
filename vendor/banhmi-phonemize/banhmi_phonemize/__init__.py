"""banhmi_phonemize: espeak-ng phonemization for BanhmiTTS.

Only the espeak-ng call itself (banhmi_phonemize_cpp.phonemize_espeak_raw)
is implemented in C++. Everything else -- Unicode normalization, phoneme/id
mapping, codepoint-mode "phonemization" -- is plain Python using the
standard library, matching the output of piper_phonemize's C++
implementation of the same steps.
"""
import os
import unicodedata
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

_DIR = Path(__file__).parent

if os.name == "nt":
    _DLLPATH = os.add_dll_directory(_DIR / "lib")

from banhmi_phonemize_cpp import phonemize_espeak_raw as _phonemize_espeak_raw

if os.name == "nt":
    _DLLPATH.close()
    del _DLLPATH


PAD = "_"
BOS = "^"
EOS = "$"

# Same symbol -> id assignment as piper_phonemize's DEFAULT_PHONEME_ID_MAP,
# reproduced here (not imported from piper) so ids stay numerically
# compatible with existing Piper-trained models. MAX_PHONEMES leaves room
# above the highest id actually assigned, for the embedding table size.
MAX_PHONEMES = 256

DEFAULT_PHONEME_ID_MAP: Dict[str, List[int]] = {
    symbol: [i]
    for i, symbol in enumerate(
        [
            PAD, BOS, EOS, " ", "!", "'", "(", ")", ",", "-", ".", ":", ";", "?",
            "a", "b", "c", "d", "e", "f", "h", "i", "j", "k", "l", "m", "n", "o",
            "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
            "æ", "ç", "ð", "ø", "ħ", "ŋ", "œ", "ǀ", "ǁ", "ǂ", "ǃ",
            "ɐ", "ɑ", "ɒ", "ɓ", "ɔ", "ɕ", "ɖ", "ɗ", "ɘ", "ə", "ɚ", "ɛ", "ɜ", "ɞ",
            "ɟ", "ɠ", "ɡ", "ɢ", "ɣ", "ɤ", "ɥ", "ɦ", "ɧ", "ɨ", "ɪ", "ɫ", "ɬ", "ɭ",
            "ɮ", "ɯ", "ɰ", "ɱ", "ɲ", "ɳ", "ɴ", "ɵ", "ɶ", "ɸ", "ɹ", "ɺ", "ɻ", "ɽ",
            "ɾ", "ʀ", "ʁ", "ʂ", "ʃ", "ʄ", "ʈ", "ʉ", "ʊ", "ʋ", "ʌ", "ʍ", "ʎ", "ʏ",
            "ʐ", "ʑ", "ʒ", "ʔ", "ʕ", "ʘ", "ʙ", "ʛ", "ʜ", "ʝ", "ʟ", "ʡ", "ʢ", "ʲ",
            "ˈ", "ˌ", "ː", "ˑ", "˞", "β", "θ", "χ", "ᵻ", "ⱱ",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "̧",  # combining cedilla
            "̃",  # combining tilde
            "̪",  # combining bridge below
            "̯",  # combining inverted breve below
            "̩",  # combining vertical line below
            "ʰ", "ˤ", "ε", "↓", "#", '"', "↑",
            "̺", "̻",  # Basque
            "g", "ʦ", "X",  # Luxembourgish
            "̝", "̊",  # Czech
        ]
    )
}


def get_espeak_map() -> Dict[str, List[int]]:
    return DEFAULT_PHONEME_ID_MAP


def get_max_phonemes() -> int:
    return MAX_PHONEMES


def get_codepoints_map() -> Dict[str, Dict[str, List[int]]]:
    # Codepoint/id maps are per-language and this fork only ships
    # espeak-ng-based phonemization (English + Vietnamese); text-codepoint
    # mode isn't populated for any language here.
    return {}


class TextCasing(str, Enum):
    IGNORE = "ignore"
    LOWER = "lower"
    UPPER = "upper"
    FOLD = "fold"


def phonemize_espeak(
    text: str,
    voice: str,
    data_path: Optional[Union[str, Path]] = None,
) -> List[List[str]]:
    if data_path is None:
        data_path = _DIR / "espeak-ng-data"

    raw_sentences = _phonemize_espeak_raw(text, voice, str(data_path))
    return [list(unicodedata.normalize("NFD", s)) for s in raw_sentences]


def phonemize_codepoints(
    text: str,
    casing: Union[str, TextCasing] = TextCasing.FOLD,
) -> List[List[str]]:
    casing = TextCasing(casing)
    if casing == TextCasing.LOWER:
        text = text.lower()
    elif casing == TextCasing.UPPER:
        text = text.upper()
    elif casing == TextCasing.FOLD:
        text = text.casefold()

    return [list(unicodedata.normalize("NFD", text))]


def _phonemes_to_ids(
    phonemes: List[str],
    phoneme_id_map: Dict[str, List[int]],
    missing_phonemes: Optional["Counter[str]"],
) -> List[int]:
    ids: List[int] = list(phoneme_id_map[BOS])
    ids.extend(phoneme_id_map[PAD])

    pad_ids = phoneme_id_map[PAD]
    for phoneme in phonemes:
        mapped = phoneme_id_map.get(phoneme)
        if mapped is None:
            if missing_phonemes is not None:
                missing_phonemes[phoneme] += 1
            continue
        ids.extend(mapped)
        ids.extend(pad_ids)

    ids.extend(phoneme_id_map[EOS])
    return ids


def phoneme_ids_espeak(
    phonemes: List[str],
    missing_phonemes: "Optional[Counter[str]]" = None,
) -> List[int]:
    return _phonemes_to_ids(phonemes, DEFAULT_PHONEME_ID_MAP, missing_phonemes)


def phoneme_ids_codepoints(
    language: str,
    phonemes: List[str],
    missing_phonemes: "Optional[Counter[str]]" = None,
) -> List[int]:
    codepoints_map = get_codepoints_map()
    if language not in codepoints_map:
        raise ValueError(f"No phoneme/id map for language: {language}")

    return _phonemes_to_ids(phonemes, codepoints_map[language], missing_phonemes)
