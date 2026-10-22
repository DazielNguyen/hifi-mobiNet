"""Synthesize a Harvard manifest with the external Piper checkpoint (frontend-mismatch test).

Same procedure as the historical ``synth_piper.py`` (journal evidence package, B_harvard):
- Piper's unmodified VITS code (``vendor/piper``; its model code equals the upstream checkout);
- ``VitsModel.load_from_checkpoint(dataset=None)``, which loads with ``weights_only=True``; only
  ``pathlib.PosixPath`` is allowed, as for the historical checkpoints;
- decoder weight norm removed; ``torch.manual_seed(1234)`` before each sentence; scales
  0.667 / 1.0 / 0.8; peak-normalized int16 WAV at 22,050 Hz.

It writes a record in the ``synthesize.py`` schema, so ``score.py`` scores it unchanged. With the
en-us manifest it should reproduce the historical Piper WAVs; ``--reference-inventory`` compares
the WAV hashes with the stored inventory.

    python scripts/evaluation/harvard/synthesize_piper.py --checkpoint <lj-med_1000.ckpt> \
        --manifest results/manifests/harvard720_espeak_en.json \
        --wav-dir <out>/piper_en/wavs --output <out>/piper_en/synth.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SEED, SCALES, SR = 1234, (0.667, 1.0, 0.8), 22050


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--wav-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--reference-inventory", type=Path,
                   help="wav_inventory.csv of the journal evidence package; compares hashes for model 'piper'")
    a = p.parse_args(argv)
    if a.output.exists() or (a.wav_dir.exists() and any(a.wav_dir.iterdir())):
        p.error("output or WAV directory already exists; choose new paths")

    import numpy as np
    import torch

    sys.path.insert(0, str(REPO / "src"))
    from hifimobinet.training.vendor import vendor_module

    lightning = vendor_module("piper_vits", "lightning")
    utils = vendor_module("piper_vits", "utils")
    wavfile = vendor_module("piper_vits", "wavfile")
    with torch.serialization.safe_globals([pathlib.PosixPath]):
        model = lightning.VitsModel.load_from_checkpoint(str(a.checkpoint), map_location="cpu", dataset=None)
    model.eval()
    with torch.no_grad():
        model.model_g.dec.remove_weight_norm()
    samples = json.loads(a.manifest.read_text(encoding="utf-8"))
    a.wav_dir.mkdir(parents=True, exist_ok=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    rows, t_all = [], time.perf_counter()
    for i, s in enumerate(samples):
        text = torch.LongTensor(s["phoneme_ids"]).unsqueeze(0)
        lengths = torch.LongTensor([text.shape[1]])
        torch.manual_seed(SEED)
        t0 = time.perf_counter()
        with torch.no_grad():
            audio = model(text, lengths, list(SCALES), sid=None).detach().numpy()
        dt = time.perf_counter() - t0
        wave = audio.squeeze().astype("float32")
        path = a.wav_dir / f"sample_{i:04d}.wav"
        wavfile.write(str(path), SR, utils.audio_float_to_int16(wave))
        dur = len(wave) / SR
        rows.append({"idx": i, "text": s["text"], "wav": path.name, "wav_sha256": sha256(path), "synth_time_s": dt,
                     "audio_duration_s": dur, "rtf": dt / dur if dur > 0 else float("nan"),
                     "finite": bool(np.isfinite(wave).all())})
    reference = None
    if a.reference_inventory:
        with open(a.reference_inventory, encoding="utf-8") as f:
            inv = {int(r["idx"]): r["sha256"] for r in csv.DictReader(f) if r["model"] == "piper"}
        reference = {"file": a.reference_inventory.name, "sha256": sha256(a.reference_inventory),
                     "wavs_identical": sum(inv.get(r["idx"]) == r["wav_sha256"] for r in rows), "n": len(rows)}
    ckpt_sha = sha256(a.checkpoint)
    record = {"created_utc": datetime.now(timezone.utc).isoformat(), "model_id": "piper_external_lj_med_1000",
              "checkpoint": a.checkpoint.name, "checkpoint_sha256": ckpt_sha, "manifest": str(a.manifest),
              "manifest_sha256": sha256(a.manifest), "n": len(rows), "wav_dir": str(a.wav_dir),
              "protocol": {"device": "cpu", "dtype": "float32", "torch_threads": torch.get_num_threads(), "scales": list(SCALES),
                           "seed_per_utterance": SEED, "decoder_weight_norm_removed": True, "sample_rate": SR,
                           "timed_region": "forward call only", "code": "vendor/piper (unmodified Piper VITS)",
                           "load": "VitsModel.load_from_checkpoint, weights_only with pathlib.PosixPath allowed"},
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "torch": torch.__version__, "numpy": np.__version__},
              "reference_inventory": reference, "script_sha256": sha256(Path(__file__)),
              "wall_time_s": time.perf_counter() - t_all, "rows": rows}
    a.output.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"done: n={len(rows)} non-finite={sum(not r['finite'] for r in rows)}"
          + (f" WAVs identical to the inventory {reference['wavs_identical']}/{len(rows)}" if reference else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
