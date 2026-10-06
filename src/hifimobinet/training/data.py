"""Dataset, split and sampler wiring shared by every training model.

Both models read the same preprocessed ``dataset.jsonl`` (fields
``phoneme_ids``, ``audio_norm_path``, ``audio_spec_path``, ``text``). Relative
tensor paths are resolved against an explicit ``data_root``. The split is
always an explicit file (train/val/test lists of ``audio_norm_path`` strings);
there is no random-split fallback. The test partition is never handed to a
training or validation dataloader.

Each model keeps its own source's Dataset/Collate classes: the internal
baseline uses BanhmiTTS's ``VitsDataset``/``UtteranceCollate`` and the Piper
model uses upstream Piper's ``PiperDataset``/``UtteranceCollate``. Both are
loaded unmodified from ``vendor/``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset, Subset

from .vendor import vendor_module

_LOGGER = logging.getLogger(__name__)
PARTITIONS = ("train", "val", "test")


@dataclass
class Split:
    partitions: Dict[str, List[str]]
    path: str
    sha256: str

    def check(self) -> None:
        sets = {name: set(self.partitions[name]) for name in PARTITIONS}
        for name in PARTITIONS:
            if len(sets[name]) != len(self.partitions[name]):
                raise ValueError(f"split {self.path}: duplicate entries in {name!r}")
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = sets[a] & sets[b]
            if overlap:
                raise ValueError(f"split {self.path}: {len(overlap)} items in both {a!r} and {b!r}")


def load_split(path: Path) -> Split:
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    missing = [name for name in PARTITIONS if not isinstance(data.get(name), list)]
    if missing:
        raise ValueError(f"split {path}: missing list(s) {missing}")
    split = Split({name: [str(x) for x in data[name]] for name in PARTITIONS}, str(path), hashlib.sha256(raw).hexdigest())
    split.check()
    return split


def _resolve(path: Path, data_root: Path) -> Path:
    return path if path.is_absolute() else data_root / path


@dataclass
class TrainingData:
    family: str
    full: Dataset
    train: Subset
    val: Subset
    keys: List[str]  # audio_norm_path string per full-dataset index (as written in dataset.jsonl)
    split: Split
    dataset_jsonl: str
    dataset_sha256: str

    def collate(self, segment_size: int):
        if self.family == "banhmi":
            return vendor_module("banhmi", "vits.dataset").UtteranceCollate(segment_size=segment_size)
        return vendor_module("piper_vits", "dataset").UtteranceCollate(is_multispeaker=False, segment_size=segment_size)


def load_training_data(family: str, dataset_jsonl: Path, data_root: Path, split: Split,
                       max_phoneme_ids: Optional[int] = None) -> TrainingData:
    """Load train/val partitions for ``family`` ('banhmi' or 'piper')."""
    dataset_jsonl = Path(dataset_jsonl)
    if family == "banhmi":
        full = vendor_module("banhmi", "vits.dataset").VitsDataset(dataset_jsonl, max_phoneme_ids=max_phoneme_ids)
    elif family == "piper":
        full = vendor_module("piper_vits", "dataset").PiperDataset([dataset_jsonl], max_phoneme_ids=max_phoneme_ids)
    else:
        raise ValueError(f"unknown dataset family {family!r}")

    keys = [str(u.audio_norm_path) for u in full.utterances]
    for utt in full.utterances:
        utt.audio_norm_path = _resolve(Path(utt.audio_norm_path), data_root)
        utt.audio_spec_path = _resolve(Path(utt.audio_spec_path), data_root)
        if getattr(utt, "audio_f0_path", None) is not None:
            # F0 is outside the selected recipes; never load it.
            utt.audio_f0_path = None

    index = {key: i for i, key in enumerate(keys)}
    if len(index) != len(keys):
        raise ValueError(f"{dataset_jsonl}: duplicate audio_norm_path entries")
    absent = {name: [k for k in split.partitions[name] if k not in index] for name in ("train", "val")}
    if any(absent.values()):
        raise ValueError(
            f"split entries missing from {dataset_jsonl} (max_phoneme_ids filtering or wrong dataset): "
            + ", ".join(f"{name}={len(v)}" for name, v in absent.items())
        )
    train_idx = [index[k] for k in split.partitions["train"]]
    val_idx = [index[k] for k in split.partitions["val"]]
    test_keys = set(split.partitions["test"])
    if test_keys & {keys[i] for i in train_idx + val_idx}:
        raise ValueError("test items reached the train/val subsets")
    return TrainingData(family, full, Subset(full, train_idx), Subset(full, val_idx), keys, split,
                        str(dataset_jsonl), hashlib.sha256(dataset_jsonl.read_bytes()).hexdigest())


def spectrogram_lengths(data: TrainingData, subset: Subset, cache_dir: Path,
                        read_only_caches: Sequence[Path] = ()) -> List[int]:
    """Frame count per subset item, for length bucketing.

    Same quantity BanhmiTTS's VitsModel._spectrogram_lengths computes, but the
    cache is written only under ``cache_dir`` (the run's own storage), never
    next to the source dataset. Existing caches keyed by the dataset.jsonl path
    strings may be read as a starting point.
    """
    cache: Dict[str, int] = {}
    for path in read_only_caches:
        if Path(path).is_file():
            try:
                cache.update({str(k): int(v) for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items()})
            except (OSError, ValueError):
                _LOGGER.warning("Ignoring unreadable length cache %s", path)
    own = Path(cache_dir) / "spectrogram_lengths.json"
    if own.is_file():
        cache.update(json.loads(own.read_text(encoding="utf-8")))
    lengths, dirty = [], False
    utterances = data.full.utterances
    raw_spec = _raw_spec_keys(data)
    for i in subset.indices:
        key = raw_spec[i]
        if key not in cache:
            cache[key] = int(torch.load(utterances[i].audio_spec_path, map_location="cpu").shape[-1])
            dirty = True
        lengths.append(cache[key])
    if dirty:
        own.parent.mkdir(parents=True, exist_ok=True)
        own.write_text(json.dumps(cache), encoding="utf-8")
    return lengths


def _raw_spec_keys(data: TrainingData) -> List[str]:
    out = []
    with open(data.dataset_jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(str(json.loads(line)["audio_spec_path"]))
    if len(out) != len(data.keys):
        # max_phoneme_ids filtering changes the row/utterance correspondence.
        rows = {}
        with open(data.dataset_jsonl, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    rows[str(row["audio_norm_path"])] = str(row["audio_spec_path"])
        out = [rows[k] for k in data.keys]
    return out


def make_subset_dataset(dataset_dir: Path, data_root: Path, split: Split, out_dir: Path,
                        n_train: int, n_val: int) -> Path:
    """Copy the first ``n_train``/``n_val`` canonical train/val utterances into a
    self-contained small dataset (dataset.jsonl, config.json, split.json, tensors).

    Test items are never selected. Returns ``out_dir``.
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty; choose a new directory")
    rows = {}
    with open(Path(dataset_dir) / "dataset.jsonl", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["audio_norm_path"])] = row
    chosen = {"train": split.partitions["train"][:n_train], "val": split.partitions["val"][:n_val], "test": []}
    tensors = out_dir / "tensors"
    tensors.mkdir(parents=True)
    new_split = {"train": [], "val": [], "test": []}
    lines = []
    for name in ("train", "val"):
        for key in chosen[name]:
            row = dict(rows[key])
            for field in ("audio_norm_path", "audio_spec_path"):
                src = _resolve(Path(row[field]), Path(data_root))
                dst = tensors / Path(row[field]).name
                shutil.copy2(src, dst)
                row[field] = str(dst.relative_to(out_dir))
            row.pop("audio_f0_path", None)
            new_split[name].append(row["audio_norm_path"])
            lines.append(json.dumps(row, ensure_ascii=False))
    (out_dir / "dataset.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy2(Path(dataset_dir) / "config.json", out_dir / "config.json")
    (out_dir / "split.json").write_text(json.dumps(new_split, indent=1), encoding="utf-8")
    (out_dir / "SUBSET_SOURCE.json").write_text(json.dumps({
        "source_split_sha256": split.sha256,
        "selected_from": {"train": "first n of canonical train", "val": "first n of canonical val"},
        "source_keys": chosen,
    }, indent=1), encoding="utf-8")
    return out_dir
