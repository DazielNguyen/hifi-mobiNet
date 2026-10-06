"""Explicit training model identities.

Training recipe IDs are distinct from the released checkpoint IDs in
models/manifest.json: ``baseline-resblock2`` and ``piper-original`` name
existing artifacts and are never produced or overwritten by this harness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict


@dataclass(frozen=True)
class TrainingModel:
    model_id: str
    family: str  # dataset/collate family: 'banhmi' or 'piper'
    description: str
    builder: Callable[[], type]


def _baseline():
    from .baseline_module import BaselineVits2Module
    return BaselineVits2Module


TRAINING_MODELS: Dict[str, TrainingModel] = {
    "baseline-resblock2-vits2": TrainingModel(
        "baseline-resblock2-vits2", "banhmi",
        "Internal baseline recipe: HiFi-GAN ResBlock2 decoder with the project's VITS2 components "
        "(Transformer-conditioned flow, duration discriminator, noise-scaled MAS)",
        _baseline,
    ),
}


def get_training_model(model_id: str) -> TrainingModel:
    try:
        return TRAINING_MODELS[model_id]
    except KeyError:
        raise KeyError(f"Unknown training model {model_id!r}; choose one of {sorted(TRAINING_MODELS)}") from None
