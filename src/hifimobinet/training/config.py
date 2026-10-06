"""Training configuration files: explicit model ID, strict key validation.

A config is a YAML mapping with four sections::

    model_id: <one of registry.TRAINING_MODELS>
    model:    {architecture hyper-parameters}
    optim:    {optimizer / loss weights / schedule}
    trainer:  {precision, gradient_clip_val, seed, ...}
    data:     {batch_size, sampler, num_workers, ...}
    harness:  {skip_nonfinite_updates, dp_health_gate, ...}

Unknown keys are rejected instead of being passed through a ``**kwargs``
catch-all, so a typo can never silently fall back to a default.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

SECTIONS = ("model", "optim", "trainer", "data", "harness")

_COMMON_MODEL = {
    "inter_channels", "hidden_channels", "filter_channels", "n_heads", "n_layers", "kernel_size",
    "p_dropout", "resblock", "resblock_kernel_sizes", "resblock_dilation_sizes", "upsample_rates",
    "upsample_initial_channel", "upsample_kernel_sizes", "filter_length", "hop_length", "win_length",
    "mel_channels", "sample_rate", "mel_fmin", "mel_fmax", "segment_size", "use_spectral_norm",
}
MODEL_KEYS = {
    "baseline-resblock2-vits2": _COMMON_MODEL | {
        "posterior_encoder_kernel_size", "posterior_encoder_dilation_rate", "posterior_encoder_layers",
        "flow_kernel_size", "flow_dilation_rate", "flow_n_flows",
        "mas_noise_scale_initial", "mas_noise_scale_decay",
    },
    # EdgeTTS (Piper fork) hard-codes posterior-encoder/flow sizes inside
    # SynthesizerTrn; only the knobs its VitsModel exposes are accepted here.
    "Piper_no_VITS2_cpn": _COMMON_MODEL | {"use_sdp", "use_bigvgan", "use_vits2", "use_f0"},
}
OPTIM_KEYS = {"learning_rate", "betas", "eps", "lr_decay", "c_mel", "c_kl"}
# EdgeTTS clips gradients itself inside its manual-optimization training_step.
EXTRA_OPTIM_KEYS = {"Piper_no_VITS2_cpn": {"grad_clip"}}
TRAINER_KEYS = {"precision", "gradient_clip_val", "seed"}
DATA_KEYS = {"batch_size", "sampler", "num_workers", "max_phoneme_ids", "num_audio_samples"}
HARNESS_KEYS = {"skip_nonfinite_updates", "dp_health_gate", "save_rng_state"}
SECTION_KEYS = {"optim": OPTIM_KEYS, "trainer": TRAINER_KEYS, "data": DATA_KEYS, "harness": HARNESS_KEYS}

SAMPLERS = ("length_bucket", "upstream_sequential")
PRECISIONS = ("32", "bf16")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class TrainingConfig:
    model_id: str
    sections: Mapping[str, Mapping[str, Any]]
    path: str
    sha256: str

    def section(self, name: str) -> Dict[str, Any]:
        return copy.deepcopy(dict(self.sections[name]))

    def flat_hparams(self) -> Dict[str, Any]:
        """All model/optim/data/harness values in one mapping (no trainer keys)."""
        out: Dict[str, Any] = {}
        for name in ("model", "optim", "data", "harness"):
            out.update(self.section(name))
        out["seed"] = self.sections["trainer"]["seed"]
        return out

    def to_json(self) -> str:
        return json.dumps({"model_id": self.model_id, **{k: dict(v) for k, v in self.sections.items()}},
                          indent=2, sort_keys=True)


def load_config(path: Path) -> TrainingConfig:
    raw = Path(path).read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a YAML mapping")
    model_id = data.get("model_id")
    if model_id not in MODEL_KEYS:
        raise ConfigError(f"{path}: model_id must be one of {sorted(MODEL_KEYS)}, got {model_id!r}")
    unknown_top = set(data) - {"model_id", *SECTIONS}
    if unknown_top:
        raise ConfigError(f"{path}: unknown top-level keys {sorted(unknown_top)}")
    sections = {}
    for name in SECTIONS:
        values = data.get(name)
        if not isinstance(values, dict):
            raise ConfigError(f"{path}: section {name!r} is required and must be a mapping")
        if name == "model":
            allowed = MODEL_KEYS[model_id]
        elif name == "optim":
            allowed = OPTIM_KEYS | EXTRA_OPTIM_KEYS.get(model_id, set())
        else:
            allowed = SECTION_KEYS[name]
        missing, unknown = allowed - set(values), set(values) - allowed
        if unknown:
            raise ConfigError(f"{path}: unknown keys in {name!r}: {sorted(unknown)}")
        if missing:
            raise ConfigError(f"{path}: missing keys in {name!r}: {sorted(missing)} (no implicit defaults)")
        sections[name] = values
    _check_values(path, sections)
    if model_id == "Piper_no_VITS2_cpn":
        _check_config_a(path, sections)
    return TrainingConfig(model_id, sections, str(path), hashlib.sha256(raw).hexdigest())


def _check_config_a(path: Path, s: Mapping[str, Mapping[str, Any]]) -> None:
    on = [k for k in ("use_bigvgan", "use_vits2", "use_f0") if s["model"][k]]
    if on:
        raise ConfigError(f"{path}: Piper_no_VITS2_cpn is EdgeTTS Config A; set {on} to false")
    if s["trainer"]["gradient_clip_val"] is not None:
        raise ConfigError(f"{path}: manual optimization clips via optim.grad_clip; trainer.gradient_clip_val must be null")
    if s["harness"]["skip_nonfinite_updates"]:
        raise ConfigError(f"{path}: skip_nonfinite_updates is not available with EdgeTTS manual optimization")


def _check_values(path: Path, s: Mapping[str, Mapping[str, Any]]) -> None:
    precision = str(s["trainer"]["precision"])
    if precision not in PRECISIONS:
        raise ConfigError(f"{path}: trainer.precision must be one of {PRECISIONS}")
    if s["data"]["sampler"] not in SAMPLERS:
        raise ConfigError(f"{path}: data.sampler must be one of {SAMPLERS}")
    if int(s["data"]["batch_size"]) <= 0:
        raise ConfigError(f"{path}: data.batch_size must be positive")
    clip = s["trainer"]["gradient_clip_val"]
    if clip is not None and float(clip) <= 0:
        raise ConfigError(f"{path}: trainer.gradient_clip_val must be null or positive")
    if s["model"]["segment_size"] % s["model"]["hop_length"]:
        raise ConfigError(f"{path}: model.segment_size must be a multiple of hop_length")
