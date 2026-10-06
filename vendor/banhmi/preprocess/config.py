"""Writes the `config.json` training config for a single-speaker voice."""
import json
from pathlib import Path

from banhmi_phonemize import get_espeak_map, get_max_phonemes


def write_config(
    output_dir: Path,
    dataset_name: str,
    audio_quality: str,
    sample_rate: int,
    language: str,
    version: str,
) -> None:
    config = {
        "dataset": dataset_name,
        "audio": {
            "sample_rate": sample_rate,
            "quality": audio_quality,
        },
        "espeak": {
            "voice": language,
        },
        "language": {
            "code": language,
        },
        "inference": {"noise_scale": 0.667, "length_scale": 1, "noise_w": 0.8},
        "phoneme_type": "espeak",
        "phoneme_map": {},
        "phoneme_id_map": get_espeak_map(),
        "num_symbols": get_max_phonemes(),
        "banhmi_version": version,
    }

    with open(output_dir / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)
