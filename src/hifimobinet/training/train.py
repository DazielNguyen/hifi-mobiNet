"""Train a selected model from a preprocessed dataset.

    python -m hifimobinet.training.train \\
        --model-id Piper_no_VITS2_cpn \\
        --config configs/training/piper-no-vits2-cpn.yaml \\
        --dataset-dir "$DATASET_DIR" --data-root "$DATA_ROOT" \\
        --split results/manifests/canonical_split.json \\
        --run-dir "$RUNS/piper-no-vits2-cpn" \\
        --accelerator gpu --devices 1 --max_epochs 1500 --checkpoint-epochs 1

The model is chosen only by --model-id, which must equal the config's
model_id. Precision, gradient clipping and seed come only from the config;
passing them on the command line is an error. PyTorch Lightning 1.7 Trainer
flags (--accelerator, --devices, --strategy, --max_epochs, ...) are accepted.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import torch

from .config import TrainingConfig, load_config
from .data import load_split, load_training_data, spectrogram_lengths
from .env import environment_record
from .registry import TRAINING_MODELS, get_training_model
from .vendor import repository_root

_LOGGER = logging.getLogger("hifimobinet.training")
_CONFIG_ONLY_FLAGS = ("--precision", "--gradient_clip_val", "--seed")


def build_module(config: TrainingConfig, dataset_dir: Path, data_root: Path, split_path: Path,
                 run_dir: Path, read_only_length_caches: Sequence[Path] = ()):
    """Construct the selected LightningModule with its data attached."""
    spec = get_training_model(config.model_id)
    with open(Path(dataset_dir) / "config.json", "r", encoding="utf-8") as handle:
        dataset_config = json.load(handle)
    num_symbols = int(dataset_config["num_symbols"])
    sample_rate = int(dataset_config["audio"]["sample_rate"])
    if sample_rate != int(config.sections["model"]["sample_rate"]):
        raise ValueError(f"dataset sample_rate {sample_rate} != config model.sample_rate")

    split = load_split(split_path)
    data = load_training_data(spec.family, Path(dataset_dir) / "dataset.jsonl", Path(data_root), split,
                              max_phoneme_ids=config.sections["data"]["max_phoneme_ids"])
    module_cls = spec.builder()
    module = module_cls(num_symbols=num_symbols, **config.flat_hparams())
    cache_dir = Path(run_dir) / "cache"
    module.attach_data(data, lambda subset: spectrogram_lengths(data, subset, cache_dir, read_only_length_caches))
    return module, data


def checkpoint_callback(run_dir: Path, every_n_epochs: int):
    from pytorch_lightning.callbacks import ModelCheckpoint
    # Same selection rule as BanhmiTTS train.py: top-3 by validation-only
    # val_loss_mel plus a rolling last.ckpt, in a resume-stable directory.
    return ModelCheckpoint(
        dirpath=str(Path(run_dir) / "checkpoints"),
        every_n_epochs=every_n_epochs,
        monitor="val_loss_mel",
        mode="min",
        save_top_k=3,
        save_last=True,
        filename="best-{epoch}-{val_loss_mel:.4f}",
    )


def make_parser() -> argparse.ArgumentParser:
    from pytorch_lightning import Trainer
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", required=True, choices=sorted(TRAINING_MODELS))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path, help="Directory with dataset.jsonl and config.json")
    parser.add_argument("--data-root", required=True, type=Path, help="Base for relative tensor paths in dataset.jsonl")
    parser.add_argument("--split", required=True, type=Path, help="JSON with train/val/test audio_norm_path lists")
    parser.add_argument("--run-dir", required=True, type=Path, help="New directory for checkpoints, logs and records")
    parser.add_argument("--checkpoint-epochs", type=int, default=1)
    parser.add_argument("--length-cache", type=Path, action="append", default=[],
                        help="Existing spectrogram-length cache to read (never written)")
    parser.add_argument("--resume", type=Path, help="Checkpoint to resume from (full Lightning state)")
    Trainer.add_argparse_args(parser)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO)
    argv = list(sys.argv[1:] if argv is None else argv)
    for flag in _CONFIG_ONLY_FLAGS:
        if any(a == flag or a.startswith(flag + "=") for a in argv):
            raise SystemExit(f"{flag} comes from the config file only")
    args = make_parser().parse_args(argv)
    config = load_config(args.config)
    if config.model_id != args.model_id:
        raise SystemExit(f"--model-id {args.model_id!r} does not match config model_id {config.model_id!r}")

    run_dir = args.run_dir.resolve()
    if args.resume is None and run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} is not empty; use a new --run-dir or --resume")
    try:
        run_dir.relative_to(repository_root().resolve())
        raise SystemExit("--run-dir must be outside the Git checkout")
    except ValueError:
        pass
    run_dir.mkdir(parents=True, exist_ok=True)

    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger

    trainer_section = config.sections["trainer"]
    # Same seeding as both original entry points (torch.manual_seed only).
    torch.manual_seed(int(trainer_section["seed"]))
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    module, data = build_module(config, args.dataset_dir, args.data_root, args.split, run_dir, args.length_cache)

    args.precision = "bf16" if str(trainer_section["precision"]) == "bf16" else 32
    args.gradient_clip_val = trainer_section["gradient_clip_val"]
    args.default_root_dir = str(run_dir)
    args.replace_sampler_ddp = config.sections["data"]["sampler"] != "length_bucket"
    trainer = Trainer.from_argparse_args(
        args,
        logger=TensorBoardLogger(save_dir=str(run_dir), name="lightning_logs"),
        callbacks=[checkpoint_callback(run_dir, args.checkpoint_epochs)],
    )

    record = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_id": config.model_id,
        "config_path": config.path,
        "config_sha256": config.sha256,
        "config": json.loads(config.to_json()),
        "split_sha256": data.split.sha256,
        "split_sizes": {k: len(v) for k, v in data.split.partitions.items()},
        "dataset_jsonl_sha256": data.dataset_sha256,
        "argv": argv,
        "resume": str(args.resume) if args.resume else None,
        "environment": environment_record(),
    }
    name = "run_record.json" if args.resume is None else f"resume_record_{datetime.datetime.now():%Y%m%dT%H%M%S}.json"
    (run_dir / name).write_text(json.dumps(record, indent=2), encoding="utf-8")
    trainer.fit(module, ckpt_path=str(args.resume) if args.resume else None)


if __name__ == "__main__":
    main()
