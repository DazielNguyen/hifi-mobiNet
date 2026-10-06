"""LJSpeech (single-speaker) dataset loading and the on-disk Utterance record.

Metadata is read as `id|text` (or LJSpeech's own `id|text|normalized_text` --
the last column is always used, since for a genuine single-speaker LJSpeech
corpus that's the normalized text, not a speaker name).
"""
import csv
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

_LOGGER = logging.getLogger("banhmi_train.preprocess.dataset")


@dataclass
class Utterance:
    text: str
    audio_path: Path
    phonemes: Optional[List[str]] = None
    phoneme_ids: Optional[List[int]] = None
    audio_norm_path: Optional[Path] = None
    audio_spec_path: Optional[Path] = None
    audio_f0_path: Optional[Path] = None
    missing_phonemes: "Counter[str]" = field(default_factory=Counter)


class PathEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def ljspeech_dataset(input_dir: Path, skip_audio: bool = False) -> Iterable[Utterance]:
    metadata_path = input_dir / "metadata.csv"
    assert metadata_path.exists(), f"Missing {metadata_path}"

    wav_dir = input_dir / "wav"
    if not wav_dir.is_dir():
        wav_dir = input_dir / "wavs"

    with open(metadata_path, "r", encoding="utf-8") as csv_file:
        # QUOTE_NONE: LJSpeech's text column routinely contains literal `"`
        # characters (quoted dialogue/titles) with no CSV-style escaping --
        # with quoting left on, Python's csv module treats an unescaped `"`
        # as opening a quoted field and silently swallows every following
        # `|` and newline until it finds a closing `"`, merging multiple
        # real rows into one and corrupting their text.
        reader = csv.reader(csv_file, delimiter="|", quoting=csv.QUOTE_NONE)
        for row in reader:
            assert len(row) >= 2, "Not enough columns"

            # id|text or id|text|normalized_text -- last column is the text
            # to speak either way.
            filename, text = row[0], row[-1]

            # Try file name relative to metadata
            wav_path = metadata_path.parent / filename

            if not wav_path.exists():
                # Try with .wav
                wav_path = metadata_path.parent / f"{filename}.wav"

            if not wav_path.exists():
                # Try wav/ or wavs/
                wav_path = wav_dir / filename

            if not wav_path.exists():
                # Try with .wav
                wav_path = wav_dir / f"{filename}.wav"

            if not skip_audio:
                if not wav_path.exists():
                    _LOGGER.warning("Missing %s", filename)
                    continue

                if wav_path.stat().st_size == 0:
                    _LOGGER.warning("Empty file: %s", wav_path)
                    continue

            yield Utterance(text=text, audio_path=wav_path)
