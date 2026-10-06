#!/usr/bin/env python3
"""Preprocess a single-speaker LJSpeech-format dataset for training.

Reads `metadata.csv` (`id|text` or LJSpeech's own `id|text|normalized_text`),
phonemizes each utterance with banhmi_phonemize (plain espeak -- no
Vietnamese code-switch detection; that's inference-only, see say.py), caches
normalized audio + spectrograms, and writes `dataset.jsonl` + `config.json`.
"""
import argparse
import json
import logging
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

from .config import write_config
from .dataset import PathEncoder, ljspeech_dataset
from .worker import init_worker, process_utterance

_DIR = Path(__file__).parent.parent
_VERSION = (_DIR / "VERSION").read_text(encoding="utf-8").strip()
_LOGGER = logging.getLogger("preprocess")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", required=True, help="Directory with LJSpeech-format dataset"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write output files for training",
    )
    parser.add_argument("--language", required=True, help="eSpeak-ng voice")
    parser.add_argument(
        "--sample-rate",
        type=int,
        required=True,
        help="Target sample rate for voice (hertz)",
    )
    parser.add_argument("--cache-dir", help="Directory to cache processed audio files")
    parser.add_argument("--max-workers", type=int)
    parser.add_argument(
        "--text-casing",
        choices=("ignore", "lower", "upper", "casefold"),
        default="ignore",
        help="Casing applied to utterance text",
    )
    parser.add_argument(
        "--dataset-name",
        help="Name of dataset to put in config (default: name of <output_dir>/../)",
    )
    parser.add_argument(
        "--audio-quality",
        help="Audio quality to put in config (default: name of <output_dir>)",
    )
    parser.add_argument(
        "--skip-audio", action="store_true", help="Don't preprocess audio"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level)
    logging.getLogger().setLevel(level)

    # Prevent log spam
    logging.getLogger("numba").setLevel(logging.WARNING)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir
        else output_dir / "cache" / str(args.sample_rate)
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    _LOGGER.info("Loading dataset from %s", input_dir)
    utterances = list(ljspeech_dataset(input_dir, args.skip_audio))
    num_utterances = len(utterances)
    assert num_utterances > 0, "No utterances found"
    _LOGGER.info("Single speaker dataset (%s utterances)", num_utterances)

    # Write config
    audio_quality = args.audio_quality or output_dir.name
    dataset_name = args.dataset_name or output_dir.parent.name
    write_config(
        output_dir, dataset_name, audio_quality, args.sample_rate, args.language, _VERSION
    )
    _LOGGER.info("Wrote dataset config")

    max_workers = args.max_workers or os.cpu_count() or 1
    # Bigger chunks = less inter-process overhead, but coarser progress
    # updates -- same tradeoff the old manual batch_size made explicit.
    chunksize = max(1, num_utterances // (max_workers * 4))

    _LOGGER.info(
        "Processing %s utterance(s) with %s worker(s)", num_utterances, max_workers
    )

    num_failed = 0
    missing_phonemes: "Counter[str]" = Counter()
    with open(output_dir / "dataset.jsonl", "w", encoding="utf-8") as dataset_file:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=init_worker,
            initargs=(
                args.language,
                args.text_casing,
                cache_dir,
                args.sample_rate,
                args.skip_audio,
            ),
        ) as executor:
            results = executor.map(process_utterance, utterances, chunksize=chunksize)
            for utt in tqdm(results, total=num_utterances, desc="Preprocessing"):
                if utt is None:
                    num_failed += 1
                    continue

                utt_dict = {
                    "text": utt.text,
                    "audio_path": utt.audio_path,
                    "phonemes": utt.phonemes,
                    "phoneme_ids": utt.phoneme_ids,
                    "audio_norm_path": utt.audio_norm_path,
                    "audio_spec_path": utt.audio_spec_path,
                    "audio_f0_path": utt.audio_f0_path,
                }
                json.dump(utt_dict, dataset_file, ensure_ascii=False, cls=PathEncoder)
                print("", file=dataset_file)

                missing_phonemes.update(utt.missing_phonemes)

    if missing_phonemes:
        for phoneme, count in missing_phonemes.most_common():
            _LOGGER.warning("Missing %s (%s)", phoneme, count)

        _LOGGER.warning("Missing %s phoneme(s)", len(missing_phonemes))

    if num_failed:
        _LOGGER.warning("Failed to process %s utterance(s)", num_failed)


if __name__ == "__main__":
    main()
