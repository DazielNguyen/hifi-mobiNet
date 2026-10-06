"""Reads dataset.jsonl (written by banhmi_train.preprocess) into padded
training batches. Single-speaker only, matching preprocess's schema -- no
speaker_id field exists in our dataset.jsonl at all (unlike upstream
Piper's Dataset/Collate, which carries an optional speaker_id throughout
for its multi-speaker case).
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from torch.utils.data import Dataset

_LOGGER = logging.getLogger("banhmi_train.vits.dataset")


@dataclass
class Utterance:
    phoneme_ids: List[int]
    audio_norm_path: Path
    audio_spec_path: Path
    text: Optional[str] = None
    audio_f0_path: Optional[Path] = None


@dataclass
class UtteranceTensors:
    phoneme_ids: torch.Tensor
    spectrogram: torch.Tensor
    audio_norm: torch.Tensor
    text: Optional[str] = None
    f0: Optional[torch.Tensor] = None


@dataclass
class Batch:
    phoneme_ids: torch.Tensor
    phoneme_lengths: torch.Tensor
    spectrograms: torch.Tensor
    spectrogram_lengths: torch.Tensor
    audios: torch.Tensor
    audio_lengths: torch.Tensor
    f0s: Optional[torch.Tensor] = None


class VitsDataset(Dataset):
    def __init__(self, dataset_path: Path, max_phoneme_ids: Optional[int] = None):
        self.utterances: List[Utterance] = list(
            self._load(Path(dataset_path), max_phoneme_ids)
        )

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> UtteranceTensors:
        utt = self.utterances[idx]
        return UtteranceTensors(
            phoneme_ids=torch.LongTensor(utt.phoneme_ids),
            audio_norm=torch.load(utt.audio_norm_path),
            spectrogram=torch.load(utt.audio_spec_path),
            text=utt.text,
            f0=torch.load(utt.audio_f0_path) if utt.audio_f0_path is not None else None,
        )

    @staticmethod
    def _load(dataset_path: Path, max_phoneme_ids: Optional[int]):
        num_skipped = 0
        with open(dataset_path, "r", encoding="utf-8") as dataset_file:
            for line_idx, line in enumerate(dataset_file):
                line = line.strip()
                if not line:
                    continue
                try:
                    utt_dict = json.loads(line)
                    phoneme_ids = utt_dict["phoneme_ids"]
                    if max_phoneme_ids is not None and len(phoneme_ids) > max_phoneme_ids:
                        num_skipped += 1
                        continue
                    f0_path = utt_dict.get("audio_f0_path")
                    yield Utterance(
                        phoneme_ids=phoneme_ids,
                        audio_norm_path=Path(utt_dict["audio_norm_path"]),
                        audio_spec_path=Path(utt_dict["audio_spec_path"]),
                        text=utt_dict.get("text"),
                        audio_f0_path=Path(f0_path) if f0_path else None,
                    )
                except Exception:
                    _LOGGER.exception("Error on line %s of %s", line_idx + 1, dataset_path)

        if num_skipped:
            _LOGGER.warning("Skipped %s utterance(s) longer than max_phoneme_ids", num_skipped)


class UtteranceCollate:
    def __init__(self, segment_size: int):
        self.segment_size = segment_size

    def __call__(self, utterances: Sequence[UtteranceTensors]) -> Batch:
        num_utterances = len(utterances)
        assert num_utterances > 0, "No utterances"

        max_phonemes_length = max(u.phoneme_ids.size(0) for u in utterances)
        max_spec_length = max(u.spectrogram.size(1) for u in utterances)
        # Audio segments are sliced at training time, so the padded batch
        # must be at least one segment long even if every clip is shorter.
        max_audio_length = max(max(u.audio_norm.size(1) for u in utterances), self.segment_size)
        num_mels = utterances[0].spectrogram.size(0)

        phonemes_padded = torch.zeros(num_utterances, max_phonemes_length, dtype=torch.long)
        spec_padded = torch.zeros(num_utterances, num_mels, max_spec_length)
        audio_padded = torch.zeros(num_utterances, 1, max_audio_length)
        # None (not all-zero) when any utterance in the batch has no cached F0
        # tensor -- keeps datasets preprocessed before F0 support existed
        # usable, just with --use-f0 unavailable for those batches.
        has_f0 = all(u.f0 is not None for u in utterances)
        f0_padded = torch.zeros(num_utterances, max_spec_length) if has_f0 else None

        phoneme_lengths = torch.zeros(num_utterances, dtype=torch.long)
        spec_lengths = torch.zeros(num_utterances, dtype=torch.long)
        audio_lengths = torch.zeros(num_utterances, dtype=torch.long)

        # Sorted by decreasing spectrogram length, as pytorch's RNN-style
        # padded-sequence utilities expect (not used directly here, but the
        # convention is kept since it's cheap and upstream relies on it).
        for i, utt in enumerate(sorted(utterances, key=lambda u: u.spectrogram.size(1), reverse=True)):
            n_phonemes = utt.phoneme_ids.size(0)
            n_spec = utt.spectrogram.size(1)
            n_audio = utt.audio_norm.size(1)

            phonemes_padded[i, :n_phonemes] = utt.phoneme_ids
            phoneme_lengths[i] = n_phonemes

            spec_padded[i, :, :n_spec] = utt.spectrogram
            spec_lengths[i] = n_spec

            audio_padded[i, :, :n_audio] = utt.audio_norm
            audio_lengths[i] = n_audio

            if f0_padded is not None:
                f0_padded[i, :n_spec] = utt.f0[:n_spec]

        return Batch(
            phoneme_ids=phonemes_padded,
            phoneme_lengths=phoneme_lengths,
            spectrograms=spec_padded,
            spectrogram_lengths=spec_lengths,
            audios=audio_padded,
            audio_lengths=audio_lengths,
            f0s=f0_padded,
        )
