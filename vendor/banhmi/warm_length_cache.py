#!/usr/bin/env python3
"""Pre-warms VitsModel's spectrogram-length cache (see vits/training.py's
_spectrogram_lengths) in a single lightweight process, before launching the
real multi-GPU DDP job.

Why this needs to exist as a separate step: VitsModel.train_dataloader()
runs independently on every DDP rank, and on a full 13,100-utterance dataset
computing lengths from scratch takes minutes (each rank torch.load()s every
cached .spec.pt once). Cold-starting that inside the real DDP job let the
two ranks drift far enough apart that NCCL's watchdog aborted the whole run
with SIGABRT before either rank reached training. Running this first means
both ranks hit an already-warm cache (a JSON file read, not thousands of
individual tensor loads) and stay in lockstep.

Deliberately does not build VitsModel (which would construct the full
generator/discriminator stack for no reason) or touch CUDA/DDP -- just the
dataset + cache file, matching _spectrogram_lengths's own cache format
(keyed by audio_spec_path, so it's valid for any train/val/test split of
the same dataset, not just one particular random_split outcome).
"""
import argparse
import json
import logging
import time
from pathlib import Path

import torch

from .vits.dataset import VitsDataset

_LOGGER = logging.getLogger("banhmi_train.warm_length_cache")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, help="Path to preprocessed dataset directory")
    parser.add_argument("--max-phoneme-ids", type=int, help="Must match the value train.py will be run with")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    dataset_path = dataset_dir / "dataset.jsonl"
    cache_path = dataset_dir / ".spectrogram_lengths_cache.json"

    cache: dict = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cache = {}
    _LOGGER.info("Existing cache entries: %d", len(cache))

    dataset = VitsDataset(dataset_path, max_phoneme_ids=args.max_phoneme_ids)
    _LOGGER.info("Dataset utterances: %d", len(dataset))

    started = time.perf_counter()
    computed = 0
    for index, utterance in enumerate(dataset.utterances):
        spec_path = str(utterance.audio_spec_path)
        if spec_path in cache:
            continue
        spec = torch.load(spec_path, map_location="cpu")
        cache[spec_path] = int(spec.shape[-1])
        computed += 1
        if computed % 1000 == 0:
            elapsed = time.perf_counter() - started
            _LOGGER.info("Computed %d new lengths (%.1fs elapsed)", computed, elapsed)

    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    _LOGGER.info(
        "Done: %d new lengths computed, %d total cached, %.1fs elapsed. Cache: %s",
        computed, len(cache), time.perf_counter() - started, cache_path,
    )


if __name__ == "__main__":
    main()
