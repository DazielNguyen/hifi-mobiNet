"""Harness behaviour shared by every training model.

These hooks are training-harness policy, not model architecture. Each one is
switched on or off explicitly in the config's ``harness`` section so the same
policy can be applied to both models of a comparison.
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List

import numpy as np
import torch

_LOGGER = logging.getLogger(__name__)
RNG_KEY = "hifimobinet_rng_state"
SKIP_RATE_ALARM = 0.01


def _world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def capture_rng_state() -> Dict[str, Any]:
    # Only tensors and plain Python values: checkpoints must stay loadable with
    # torch.load(weights_only=True), which Lightning 1.7 uses implicitly on
    # torch >= 2.6. A raw NumPy state array would make resume fail.
    name, keys, pos, has_gauss, cached = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {"name": str(name), "keys": torch.from_numpy(keys.astype(np.int64)), "pos": int(pos),
                  "has_gauss": int(has_gauss), "cached_gaussian": float(cached)},
        "torch": torch.get_rng_state(),
        "world_size": _world_size(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> bool:
    """Restore a single-process RNG snapshot. Returns False (and restores
    nothing) under multi-process training: the checkpoint holds rank 0's state
    only, and giving every rank the same stream would correlate their
    dropout/segment sampling."""
    if state.get("world_size", 1) != 1 or _world_size() != 1:
        _LOGGER.warning("RNG state not restored: only single-process RNG restore is supported")
        return False
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n["name"], n["keys"].numpy().astype(np.uint32), n["pos"], n["has_gauss"], n["cached_gaussian"]))
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])
    return True


class HarnessMixin:
    """Mixed into a LightningModule. Reads ``self.hparams`` keys
    ``skip_nonfinite_updates`` and ``save_rng_state``."""

    _skipped_steps = 0
    _total_steps = 0
    rng_restored = None

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric) -> None:
        # Version compatibility, not a behaviour change: PyTorch Lightning 1.7.7
        # only accepts schedulers that subclass torch's legacy `_LRScheduler`,
        # but since torch 2.0 ExponentialLR derives from `LRScheduler`, so
        # Lightning rejects it unless this hook is overridden. The body is
        # Lightning 1.7.7's own default implementation, called at the same point.
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.hparams.save_rng_state:
            checkpoint[RNG_KEY] = capture_rng_state()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self._pending_rng = checkpoint.get(RNG_KEY) if self.hparams.save_rng_state else None

    def on_train_start(self) -> None:
        # Restored here (after Lightning finished restoring the loops) so no
        # later restore step can consume random numbers first.
        pending = getattr(self, "_pending_rng", None)
        if pending is not None:
            self.rng_restored = restore_rng_state(pending)
            if self.rng_restored:
                self._rng_after_restore = torch.get_rng_state().clone()
            self._pending_rng = None

    def nonfinite_skipping_step(self, optimizer, optimizer_idx, optimizer_closure, batch_info) -> None:
        """BanhmiTTS's discard-non-finite-update policy (vits/training.py
        VitsModel.optimizer_step), factored out so either model can opt in."""
        optimizer_closure()
        nonfinite = any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for group in optimizer.param_groups for p in group["params"]
        )
        if optimizer_idx == 0:
            self._total_steps += 1
        if not nonfinite:
            optimizer.step()
            return
        optimizer.zero_grad(set_to_none=True)
        self._skipped_steps += 1
        rate = self._skipped_steps / max(self._total_steps, 1)
        _LOGGER.warning(
            "Discarded optimizer step (optimizer_idx=%d): non-finite gradients. Skipped %d of %d "
            "steps so far (%.4f%%). Batch fingerprint=%r",
            optimizer_idx, self._skipped_steps, self._total_steps, 100 * rate, batch_info,
        )
        if rate > SKIP_RATE_ALARM and self._total_steps > 200:
            _LOGGER.error("Discarded-step rate %.2f%% exceeds %.2f%%; the run is likely diverging",
                          100 * rate, 100 * SKIP_RATE_ALARM)
        self.log("skipped_steps", float(self._skipped_steps))


def free_running_healthy(module, dataset, sample_rate: int, log_audio: bool = True) -> bool:
    """Synthesize each utterance in ``dataset`` with free-running inference.
    Returns False when any output is non-finite or shorter than 0.2 s (BanhmiTTS's
    collapsed-duration-predictor check, vits/training.py _log_audio_samples)."""
    min_samples = int(0.2 * sample_rate)
    healthy = True
    for idx, utt in enumerate(dataset):
        text = utt.phoneme_ids.unsqueeze(0).to(module.device)
        lengths = torch.LongTensor([len(utt.phoneme_ids)]).to(module.device)
        audio = module(text, lengths, [0.667, 1.0, 0.8]).detach()
        if audio.shape[-1] < min_samples or not torch.isfinite(audio).all():
            healthy = False
            _LOGGER.warning("Free-running inference produced %d samples for utterance %d",
                            audio.shape[-1], idx)
        if log_audio and module.logger is not None:
            audio = audio * (1.0 / max(0.01, abs(audio.max())))
            module.logger.experiment.add_audio(utt.text or str(idx), audio, sample_rate=sample_rate)
    return healthy


def parameter_groups(module) -> List[str]:
    return [name for name, _ in module.named_children()]
