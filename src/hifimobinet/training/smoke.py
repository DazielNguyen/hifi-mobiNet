"""Short real-training smoke test for one model (functional checks only).

    python -m hifimobinet.training.smoke --model-id Piper_no_VITS2_cpn \\
        --config configs/training/piper-no-vits2-cpn.yaml \\
        --dataset-dir "$SMOKE_DATA" --data-root "$SMOKE_DATA" --split "$SMOKE_DATA/split.json" \\
        --run-dir "$RUNS/smoke-piper" --batch-size 4

Phase A trains ``--epochs-a`` epochs from scratch and saves checkpoints;
phase B resumes from phase A's last.ckpt and trains to ``--epochs-b``. The
number of optimizer updates is checked against ``--max-updates``. Writes
``smoke_report.json`` with pass/fail/not_checked per check and one PyTorch
inference WAV. Outputs are functional evidence, not research results.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import inspect
import json
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from .config import TrainingConfig, load_config
from .env import environment_record
from .train import build_module, checkpoint_callback

_LOGGER = logging.getLogger("hifimobinet.training.smoke")
VITS2_CLASS_MARKERS = ("DurationDiscriminator", "TransformerCoupling", "MultiResolutionDiscriminator")
VITS2_METRICS = ("mas_noise_scale", "loss_dur_gen", "loss_disc_dur")


class Report:
    def __init__(self):
        self.checks: Dict[str, Dict[str, Any]] = {}

    def set(self, name: str, status: str, detail: Any = None) -> None:
        if self.checks.get(name, {}).get("status") == "fail" and status == "pass":
            return  # a failure observed earlier is never overwritten
        self.checks[name] = {"status": status, "detail": detail}


def _snapshot(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().to("cpu", copy=True) for k, v in module.state_dict().items()}


class SmokeChecks(Callback):
    def __init__(self, report: Report, num_symbols: int, hop_length: int, segment_size: int):
        self.r, self.num_symbols, self.hop, self.segment = report, num_symbols, hop_length, segment_size
        self.updates = {0: 0, 1: 0}
        self.grad_norms: Dict[int, List[float]] = {0: [], 1: []}
        self.first_batch_checked = False
        self.nonfinite_metrics: List[str] = []
        self.before: Dict[str, Dict[str, torch.Tensor]] = {}
        self.val_before: Optional[Dict[str, torch.Tensor]] = None
        self.val_runs = 0

    def on_fit_start(self, trainer, pl_module):
        self.before = {name: _snapshot(child) for name, child in pl_module.named_children()}
        groups = [set(id(p) for g in opt.param_groups for p in g["params"]) for opt in trainer.optimizers]
        gen = {id(p) for p in pl_module.model_g.parameters()}
        disc = {id(p) for name, child in pl_module.named_children() if name != "model_g" for p in child.parameters()}
        ok = len(groups) == 2 and groups[0] == gen and groups[1] == disc and not (groups[0] & groups[1])
        self.r.set("optimizer_param_coverage", "pass" if ok else "fail",
                   {"generator_params": len(gen), "discriminator_params": len(disc),
                    "optimizer_sizes": [len(g) for g in groups]})

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, *args):
        if self.first_batch_checked:
            return
        self.first_batch_checked = True
        issues = []
        b = batch
        n = b.phoneme_ids.shape[0]
        if getattr(b, "f0s", None) is not None:
            issues.append("batch carries F0")
        if b.phoneme_ids.min() < 0 or b.phoneme_ids.max() >= self.num_symbols:
            issues.append("phoneme id out of range")
        for name, lengths, dim in (("phoneme", b.phoneme_lengths, b.phoneme_ids.shape[1]),
                                   ("spec", b.spectrogram_lengths, b.spectrograms.shape[2]),
                                   ("audio", b.audio_lengths, b.audios.shape[2])):
            if lengths.shape[0] != n or lengths.max() > dim or lengths.min() <= 0:
                issues.append(f"{name} lengths inconsistent with padding")
        if b.spectrograms.shape[1] != pl_module.hparams.filter_length // 2 + 1:
            issues.append("spectrogram channels != filter_length//2+1")
        if b.audios.shape[2] < self.segment:
            issues.append("padded audio shorter than segment_size")
        if not torch.equal(b.spectrogram_lengths, b.audio_lengths // self.hop):
            issues.append("spectrogram_lengths != audio_lengths // hop_length")
        for i in range(n):  # padding beyond each length must be zero
            if b.phoneme_ids[i, b.phoneme_lengths[i]:].abs().sum() or b.spectrograms[i, :, b.spectrogram_lengths[i]:].abs().sum():
                issues.append(f"non-zero padding in item {i}")
                break
        self.r.set("batch_shapes_lengths_padding", "fail" if issues else "pass",
                   {"issues": issues, "phoneme_ids": list(b.phoneme_ids.shape), "spectrograms": list(b.spectrograms.shape),
                    "audios": list(b.audios.shape)})

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, opt_idx):
        grads = [p.grad for g in optimizer.param_groups for p in g["params"] if p.grad is not None]
        finite = all(torch.isfinite(g).all() for g in grads)
        norm = torch.norm(torch.stack([g.detach().float().norm() for g in grads])).item() if grads else 0.0
        self.grad_norms[opt_idx].append(norm)
        self.updates[opt_idx] += 1
        self.r.set("gradients_finite", "pass" if finite and grads else "fail",
                   {"optimizer": opt_idx, "num_grads": len(grads)})

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args):
        for key, value in trainer.callback_metrics.items():
            if key.startswith("loss") and not torch.isfinite(torch.as_tensor(value)).all():
                self.nonfinite_metrics.append(key)
        self.r.set("losses_finite", "fail" if self.nonfinite_metrics else "pass",
                   {"nonfinite": sorted(set(self.nonfinite_metrics))})

    def on_validation_start(self, trainer, pl_module):
        self.val_before = _snapshot(pl_module)

    def on_validation_end(self, trainer, pl_module):
        after = _snapshot(pl_module)
        same = all(torch.equal(self.val_before[k], after[k]) for k in after)
        self.val_runs += 1
        self.r.set("validation_does_not_update_weights", "pass" if same else "fail", {"validation_runs": self.val_runs})

    def on_fit_end(self, trainer, pl_module):
        changed = {}
        for name, child in pl_module.named_children():
            now = _snapshot(child)
            params = {n for n, _ in child.named_parameters()}
            diff = [k for k in params if not torch.equal(self.before[name][k], now[k])]
            changed[name] = {"changed": len(diff), "total": len(params)}
        self.r.set("generator_updates", "pass" if changed["model_g"]["changed"] > 0 else "fail", changed["model_g"])
        self.r.set("waveform_discriminator_updates", "pass" if changed["model_d"]["changed"] > 0 else "fail",
                   changed["model_d"])
        if "model_d_dur" in changed:
            self.r.set("duration_discriminator_updates",
                       "pass" if changed["model_d_dur"]["changed"] > 0 else "fail", changed["model_d_dur"])
        self.changed = changed


class ResumeChecks(Callback):
    def __init__(self, report: Report, ckpt: Dict[str, Any]):
        self.r, self.ckpt = report, ckpt

    def on_train_start(self, trainer, pl_module):
        ck = self.ckpt
        state = pl_module.state_dict()
        model_ok = set(state) == set(ck["state_dict"]) and all(
            torch.equal(state[k].cpu(), ck["state_dict"][k].cpu()) for k in state)
        opt_ok = True
        for live, saved in zip(trainer.optimizers, ck["optimizer_states"]):
            ls = live.state_dict()
            if [g["lr"] for g in ls["param_groups"]] != [g["lr"] for g in saved["param_groups"]]:
                opt_ok = False
            for pid, pstate in saved["state"].items():
                for key, value in pstate.items():
                    if torch.is_tensor(value) and not torch.equal(ls["state"][pid][key].cpu(), value.cpu()):
                        opt_ok = False
        sched_live = [c.scheduler.state_dict() for c in trainer.lr_scheduler_configs]
        sched_ok = all(l["last_epoch"] == s["last_epoch"] and l["_last_lr"] == s["_last_lr"]
                       for l, s in zip(sched_live, ck["lr_schedulers"]))
        self.r.set("resume_model_state", "pass" if model_ok else "fail")
        self.r.set("resume_optimizer_state", "pass" if opt_ok else "fail")
        self.r.set("resume_scheduler_state", "pass" if sched_ok else "fail",
                   {"live": [s["last_epoch"] for s in sched_live], "saved": [s["last_epoch"] for s in ck["lr_schedulers"]]})
        self.r.set("resume_global_step", "pass" if trainer.global_step == ck["global_step"] else "fail",
                   {"live": trainer.global_step, "saved": ck["global_step"], "saved_epoch": ck["epoch"],
                    "live_epoch": trainer.current_epoch})

    def on_fit_end(self, trainer, pl_module):
        saved = self.ckpt.get("hifimobinet_rng_state")
        after = getattr(pl_module, "_rng_after_restore", None)
        if saved is None:
            self.r.set("resume_rng_state", "not_checked", "no RNG state in checkpoint")
        elif after is None:
            self.r.set("resume_rng_state", "fail", "RNG state present but not restored")
        else:
            self.r.set("resume_rng_state", "pass" if torch.equal(after, saved["torch"]) else "fail",
                       "torch CPU RNG compared immediately after restore; CUDA/NumPy/Python restored, not compared")


def vits2_absence(module, report: Report) -> None:
    """Config A: no VITS2 (Transformer flow, duration discriminator, noised MAS),
    no BigVGAN parts (Snake, MRD) and no F0 branch."""
    names = sorted({type(m).__name__ for m in module.modules()})
    flagged = [n for n in names if any(marker in n for marker in VITS2_CLASS_MARKERS + ("Snake", "F0Predictor"))]
    flow_attention = sorted({type(m).__name__ for m in module.model_g.flow.modules() if "Attention" in type(m).__name__})
    text_attention = sorted({type(m).__name__ for m in module.model_g.enc_p.modules() if "Attention" in type(m).__name__})
    g = module.model_g
    switches = {"use_noised_mas": getattr(g, "use_noised_mas", None), "use_f0": getattr(g, "use_f0", None),
                "dec.use_snake": getattr(g.dec, "use_snake", None),
                "flow.use_transformer_flows": getattr(g.flow, "use_transformer_flows", None)}
    absent = {"model_d_dur": getattr(module, "model_d_dur", None) is None,
              "model_d_mrd": getattr(module, "model_d_mrd", None) is None}
    ok = not flagged and not flow_attention and not any(switches.values()) and all(absent.values())
    report.set("no_vits2_modules_or_optimizer_groups", "pass" if ok else "fail", {
        "flagged_classes": flagged, "flow_attention_classes": flow_attention,
        "text_encoder_attention_classes_kept": text_attention, "switches": switches,
        "discriminators_absent": absent, "children": [n for n, _ in module.named_children()],
    })


def vits2_presence(module, report: Report) -> None:
    names = {type(m).__name__ for m in module.modules()}
    present = {marker: any(marker in n for n in names) for marker in ("DurationDiscriminator", "TransformerCoupling")}
    sig = inspect.signature(module.model_g.forward).parameters
    ok = all(present.values()) and "mas_noise_scale" in sig
    report.set("vits2_components_present", "pass" if ok else "fail", {"classes": present, "forward_parameters": list(sig)})


def _with_overrides(config: TrainingConfig, data_overrides: Dict[str, Any]) -> TrainingConfig:
    sections = copy.deepcopy({k: dict(v) for k, v in config.sections.items()})
    sections["data"].update(data_overrides)
    return dataclasses.replace(config, sections=sections)


def _trainer(run_dir: Path, config: TrainingConfig, epochs: int, callbacks, devices: int) -> Trainer:
    precision = "bf16" if str(config.sections["trainer"]["precision"]) == "bf16" else 32
    return Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=devices, max_epochs=epochs,
        precision=precision, gradient_clip_val=config.sections["trainer"]["gradient_clip_val"],
        default_root_dir=str(run_dir), logger=TensorBoardLogger(save_dir=str(run_dir), name="lightning_logs"),
        callbacks=[checkpoint_callback(run_dir, 1), *callbacks], num_sanity_val_steps=0,
        log_every_n_steps=1, enable_progress_bar=False, replace_sampler_ddp=False,
    )


def write_wav(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    import numpy as np
    import soundfile
    samples = audio.squeeze().float().cpu().numpy()
    peak = max(0.01, float(np.abs(samples).max()))
    soundfile.write(str(path), (samples / peak * 32767).astype(np.int16), sample_rate, subtype="PCM_16")


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", required=True)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--dataset-dir", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--split", required=True, type=Path)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--epochs-a", type=int, default=2)
    p.add_argument("--epochs-b", type=int, default=4)
    p.add_argument("--max-updates", type=int, default=50)
    p.add_argument("--devices", type=int, default=1)
    args = p.parse_args(argv)

    config = load_config(args.config)
    if config.model_id != args.model_id:
        raise SystemExit("--model-id does not match the config")
    overrides = {"batch_size": args.batch_size, "num_workers": args.num_workers, "num_audio_samples": 2}
    config = _with_overrides(config, overrides)
    run_dir = args.run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} is not empty")
    run_dir.mkdir(parents=True)
    report = Report()
    out: Dict[str, Any] = {"model_id": config.model_id, "config": config.path, "config_sha256": config.sha256,
                           "smoke_overrides": overrides, "environment": environment_record(),
                           "classification": "functional smoke test; not a research result"}
    torch.manual_seed(int(config.sections["trainer"]["seed"]))
    torch.set_float32_matmul_precision("high")
    try:
        module, data = build_module(config, args.dataset_dir, args.data_root, args.split, run_dir)
        out["data"] = {"train": len(data.train), "val": len(data.val), "split_sha256": data.split.sha256,
                       "test_items_in_split_file": len(data.split.partitions["test"])}
        report.set("ids_match_dataset", "pass" if all(
            torch.equal(data.train[i].phoneme_ids, torch.LongTensor(data.full.utterances[data.train.indices[i]].phoneme_ids))
            for i in range(len(data.train))) else "fail", "tensor IDs equal dataset.jsonl rows")
        if config.model_id == "Piper_no_VITS2_cpn":
            vits2_absence(module, report)
        else:
            vits2_presence(module, report)
        hop, seg = module.hparams.hop_length, module.hparams.segment_size

        checks_a = SmokeChecks(report, module.hparams.num_symbols, hop, seg)
        trainer_a = _trainer(run_dir, config, args.epochs_a, [checks_a], args.devices)
        trainer_a.fit(module)
        out["phase_a"] = {"global_step": trainer_a.global_step, "updates": checks_a.updates,
                          "grad_norm_first_last": {k: [v[0], v[-1]] if v else None for k, v in checks_a.grad_norms.items()},
                          "parameter_changes": getattr(checks_a, "changed", None),
                          "metrics": {k: float(v) for k, v in trainer_a.callback_metrics.items()}}
        if config.model_id == "Piper_no_VITS2_cpn":
            # EdgeTTS always logs loss_dur_gen/loss_disc_dur; without a duration
            # discriminator they must be exactly zero. No MAS-noise schedule is logged.
            metrics = trainer_a.callback_metrics
            nonzero = [m for m in ("loss_dur_gen", "loss_disc_dur", "loss_gen_mrd", "loss_disc_mrd", "loss_f0")
                       if m in metrics and float(metrics[m]) != 0.0]
            noise = "mas_noise_scale" in metrics
            report.set("no_vits2_losses_logged", "fail" if nonzero or noise else "pass",
                       {"nonzero_component_losses": nonzero, "mas_noise_scale_logged": noise})
        else:
            needed = [m for m in VITS2_METRICS if m not in trainer_a.callback_metrics]
            report.set("vits2_losses_logged", "fail" if needed else "pass", {"missing": needed})

        last = run_dir / "checkpoints" / "last.ckpt"
        try:
            ckpt = torch.load(last, map_location="cpu", weights_only=True)
            report.set("checkpoint_loads_weights_only", "pass")
        except Exception as error:  # recorded, then the trusted own checkpoint is read fully
            report.set("checkpoint_loads_weights_only", "fail", repr(error))
            ckpt = torch.load(last, map_location="cpu", weights_only=False)

        module_b, _ = build_module(config, args.dataset_dir, args.data_root, args.split, run_dir)
        checks_b = SmokeChecks(report, module_b.hparams.num_symbols, hop, seg)
        trainer_b = _trainer(run_dir, config, args.epochs_b, [checks_b, ResumeChecks(report, ckpt)], args.devices)
        trainer_b.fit(module_b, ckpt_path=str(last))
        total = sum(checks_a.updates.values()) + sum(checks_b.updates.values())
        out["phase_b"] = {"global_step": trainer_b.global_step, "updates": checks_b.updates,
                          "metrics": {k: float(v) for k, v in trainer_b.callback_metrics.items()}}
        out["total_optimizer_updates"] = total
        report.set("update_budget", "pass" if total <= args.max_updates else "fail",
                   {"total": total, "limit": args.max_updates})

        module_b.eval()
        utt = data.val[0]
        with torch.no_grad():
            torch.manual_seed(0)
            audio = module_b(utt.phoneme_ids.unsqueeze(0).to(module_b.device),
                             torch.LongTensor([len(utt.phoneme_ids)]).to(module_b.device), [0.667, 1.0, 0.8])
        wav = run_dir / "smoke_inference.wav"
        write_wav(wav, audio, module_b.hparams.sample_rate)
        finite = bool(torch.isfinite(audio).all())
        report.set("pytorch_inference_wav", "pass" if finite and audio.numel() > 0 else "fail",
                   {"samples": int(audio.numel()), "seconds": audio.numel() / module_b.hparams.sample_rate,
                    "file": wav.name, "note": "not a quality judgement"})
    except Exception as error:
        out["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        report.set("run_completed", "fail", repr(error))
    else:
        report.set("run_completed", "pass")
    for name in ("duration_discriminator_updates",):
        if config.model_id == "Piper_no_VITS2_cpn":
            report.checks.setdefault(name, {"status": "not_applicable", "detail": "model has no duration discriminator"})
    out["checks"] = report.checks
    out["summary"] = {s: sorted(k for k, v in report.checks.items() if v["status"] == s)
                      for s in ("pass", "fail", "not_checked", "not_applicable")}
    (run_dir / "smoke_report.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out["summary"], indent=2))
    if out["summary"]["fail"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
