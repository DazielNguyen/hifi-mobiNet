"""Synthesize Harvard-720 with a checkpoint trained by the hifi-mobiNet harness.

Same inference protocol as the historical Harvard runs of the four released models
(BanhmiTTS scripts/harvard/synth_banhmi.py and synth_piper.py, archived in the
journal evidence package):

- PyTorch FP32 on CPU, default thread count; decoder weight norm removed;
- scales noise 0.667, length 1.0, noise_w 0.8; no speaker id;
- ``torch.manual_seed(1234)`` before every utterance, so each sample's noise does
  not depend on order;
- the timed region is the forward call only; RTF = synthesis time / audio duration;
- audio peak-normalised to int16 and written as mono 22,050 Hz PCM16 WAV.

The module is built from the training config and the checkpoint's state dict is
loaded strictly with ``torch.load(weights_only=True)``. Hyper-parameters stored in
the checkpoint must equal the config's, otherwise the run stops. Paths are stored
as given; run from the checkout root with relative paths so that records contain
no host-specific paths.

    python scripts/evaluation/harvard/synthesize.py \
        --config configs/training/piper-no-vits2-cpn.yaml \
        --checkpoint <run>/checkpoints/best-epoch=...ckpt \
        --manifest results/manifests/harvard720.json \
        --wav-dir <outside Git>/wavs --output <outside Git>/synth.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
NOISE_SCALE, LENGTH_SCALE, NOISE_SCALE_W = 0.667, 1.0, 0.8
SEED = 1234


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _plain(value):
    """Normalise tuples/lists for comparing YAML values with stored hparams."""
    return json.loads(json.dumps(value))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--wav-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args(argv)
    if a.output.exists() or (a.wav_dir.exists() and any(a.wav_dir.iterdir())):
        p.error("output or WAV directory already exists; choose new paths")

    os.environ.setdefault("HIFIMOBINET_HOME", str(REPO))
    sys.path.insert(0, str(REPO / "src"))
    import numpy as np
    import torch

    from hifimobinet.training.config import load_config
    from hifimobinet.training.registry import get_training_model
    from hifimobinet.training.vendor import vendor_module

    cfg = load_config(a.config)
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    stored = ckpt["hyper_parameters"]
    expected = cfg.flat_hparams()
    mismatch = {k: [_plain(stored.get(k)), _plain(v)] for k, v in expected.items() if _plain(stored.get(k)) != _plain(v)}
    if mismatch:
        raise SystemExit(f"checkpoint hyper-parameters differ from {a.config}: {mismatch}")
    module_cls = get_training_model(cfg.model_id).builder()
    module = module_cls(num_symbols=stored["num_symbols"], **expected)
    module.load_state_dict(ckpt["state_dict"], strict=True)
    module.eval()
    with torch.no_grad():
        module.model_g.dec.remove_weight_norm()
    sample_rate = int(stored["sample_rate"])

    utils = vendor_module("edgetts_vits", "utils")
    wavfile = vendor_module("edgetts_vits", "wavfile")
    samples = json.loads(a.manifest.read_text(encoding="utf-8"))
    a.wav_dir.mkdir(parents=True, exist_ok=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    t_all = time.perf_counter()
    for i, s in enumerate(samples):
        x = torch.LongTensor(s["phoneme_ids"]).unsqueeze(0)
        x_lengths = torch.LongTensor([x.shape[1]])
        torch.manual_seed(SEED)
        t0 = time.perf_counter()
        with torch.no_grad():
            audio = module(x, x_lengths, [NOISE_SCALE, LENGTH_SCALE, NOISE_SCALE_W])
        synth_time = time.perf_counter() - t0
        wave = audio.squeeze().numpy().astype("float32")
        duration = len(wave) / sample_rate
        wav_path = a.wav_dir / f"sample_{i:04d}.wav"
        wavfile.write(str(wav_path), sample_rate, utils.audio_float_to_int16(wave))
        rows.append({"idx": i, "text": s["text"], "wav": wav_path.name, "wav_sha256": sha256(wav_path),
                     "synth_time_s": synth_time, "audio_duration_s": duration,
                     "rtf": synth_time / duration if duration > 0 else float("nan"),
                     "finite": bool(np.isfinite(wave).all())})
        if (i + 1) % 100 == 0:
            print(f"[synth {i + 1}/{len(samples)}] {(time.perf_counter() - t_all) / 60:.1f} min", flush=True)

    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": cfg.model_id, "config": str(a.config), "config_sha256": cfg.sha256,
        "checkpoint": a.checkpoint.name, "checkpoint_sha256": sha256(a.checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"), "checkpoint_global_step": ckpt.get("global_step"),
        "state_dict_entries": len(ckpt["state_dict"]), "strict_load": True,
        "manifest": str(a.manifest), "manifest_sha256": sha256(a.manifest), "n": len(rows),
        "wav_dir": str(a.wav_dir),
        "protocol": {"device": "cpu", "dtype": "float32", "torch_threads": torch.get_num_threads(),
                     "scales": [NOISE_SCALE, LENGTH_SCALE, NOISE_SCALE_W], "seed_per_utterance": SEED,
                     "decoder_weight_norm_removed": True, "sample_rate": sample_rate,
                     "timed_region": "forward call only"},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "numpy": np.__version__,
                        "cpu": platform.processor() or None},
        "wall_time_s": time.perf_counter() - t_all,
        "rows": rows,
    }
    a.output.write_text(json.dumps(record, indent=1), encoding="utf-8")
    rtf = np.array([r["rtf"] for r in rows])
    print(f"done: n={len(rows)} mean RTF={rtf.mean():.5f} non-finite={sum(not r['finite'] for r in rows)} -> {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
