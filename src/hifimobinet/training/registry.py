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


def _piper():
    from .piper_module import PiperNoVits2Module
    return PiperNoVits2Module


def _piper_ir():
    from .piper_ir_module import PiperIRNoVits2Module
    return PiperIRNoVits2Module


TRAINING_MODELS: Dict[str, TrainingModel] = {
    "baseline-resblock2-vits2": TrainingModel(
        "baseline-resblock2-vits2", "banhmi",
        "Internal baseline recipe: HiFi-GAN ResBlock2 decoder with the project's VITS2 components "
        "(Transformer-conditioned flow, duration discriminator, noise-scaled MAS)",
        _baseline,
    ),
    "Piper_no_VITS2_cpn": TrainingModel(
        "Piper_no_VITS2_cpn", "edgetts",
        "EdgeTTS Config A: vanilla Piper (EdgeTTS a73a897, fork of rhasspy/piper 73c04d8) with "
        "BigVGAN, VITS2 and F0 flags off, trained from scratch in the project harness",
        _piper,
    ),
    "Piper_Low_no_VITS2_cpn": TrainingModel(
        "Piper_Low_no_VITS2_cpn", "edgetts",
        "Piper Low: Config A with the Medium architecture at 16 kHz (not x-low), trained from scratch",
        _piper,
    ),
    "Piper_High_no_VITS2_cpn": TrainingModel(
        "Piper_High_no_VITS2_cpn", "edgetts",
        "Piper High: Config A with the upstream High ResBlock1/512-channel decoder at 22.05 kHz, trained from scratch",
        _piper,
    ),
    "Piper_ParIR_no_VITS2_cpn": TrainingModel(
        "Piper_ParIR_no_VITS2_cpn", "edgetts",
        "Piper-ParIR: the Piper_no_VITS2_cpn backbone (EdgeTTS Config A) with the released Parallel-IR "
        "decoder bundle (three parallel inverted-residual branches, kernels 3/5/7, expansion 1, SnakeBeta, MRD)",
        _piper_ir,
    ),
    "Piper_SeqIR_no_VITS2_cpn": TrainingModel(
        "Piper_SeqIR_no_VITS2_cpn", "edgetts",
        "Piper-SeqIR: the Piper_no_VITS2_cpn backbone (EdgeTTS Config A) with the released Sequential-IR "
        "decoder bundle (two sequential inverted-residual blocks, kernel 3, expansion 1, SnakeBeta, MRD)",
        _piper_ir,
    ),
}


def get_training_model(model_id: str) -> TrainingModel:
    try:
        return TRAINING_MODELS[model_id]
    except KeyError:
        raise KeyError(f"Unknown training model {model_id!r}; choose one of {sorted(TRAINING_MODELS)}") from None
