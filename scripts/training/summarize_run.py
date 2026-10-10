"""Summarize a finished training run: checkpoints and the per-epoch val_loss_mel curve.

Reads TensorBoard event files (scalars only; nothing is trained or synthesized) of the
run and, optionally, of a reference run, and writes one JSON record with:

- the run's checkpoints (name, bytes, SHA-256) and run_record.json fields;
- val_loss_mel per epoch for each run (one validation per epoch; points are ordered by
  global step, and a later point at the same step replaces an earlier one, which handles
  resumed runs);
- best value and epoch, counts below thresholds, and window means/SD/min over identical
  epoch ranges for both runs.

    python scripts/training/summarize_run.py --run-dir training_output/piper-no-vits2-cpn \
        --reference-logdir <reference run>/lightning_logs --reference-label baseline-resblock2 \
        --output results/piper-no-vits2-cpn/training_summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

WINDOWS = ((0, 99), (500, 599), (900, 999), (1000, 1099), (1100, 1199), (1200, 1299), (1300, 1399), (1400, 1499),
           (1250, 1449), (1300, 1499))
THRESHOLDS = (20.0, 19.9, 19.8)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def val_mel_series(logdir: Path):
    import numpy as np
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

    points = {}
    files = sorted(logdir.rglob("events.out.tfevents*"))
    for f in files:
        for event in EventFileLoader(str(f)).Load():
            for v in event.summary.value:
                if v.tag == "val_loss_mel":
                    if v.HasField("simple_value"):
                        value = v.simple_value
                    elif v.tensor.tensor_content:
                        value = float(np.frombuffer(v.tensor.tensor_content, np.float32)[0])
                    else:
                        value = float(v.tensor.float_val[0])
                    points[event.step] = value
    steps = sorted(points)
    return [points[s] for s in steps], steps, [f.name for f in files]


def describe(series):
    import numpy as np

    v = np.array(series)
    out = {"epochs": len(v), "best": float(v.min()), "best_epoch": int(v.argmin()), "last": float(v[-1]),
           "below": {str(t): int((v < t).sum()) for t in THRESHOLDS}, "windows": {}}
    for lo, hi in WINDOWS:
        if len(v) > hi:
            w = v[lo:hi + 1]
            out["windows"][f"{lo}-{hi}"] = {"mean": float(w.mean()), "sd": float(w.std(ddof=1)), "min": float(w.min()),
                                            "below_20.0": int((w < 20.0).sum())}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--reference-logdir", type=Path)
    p.add_argument("--reference-label")
    p.add_argument("--reference-best-checkpoint", help="file name of the reference run's selected checkpoint")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args(argv)
    if a.output.exists():
        p.error("output exists; choose a new path")

    record = json.loads((a.run_dir / "run_record.json").read_text(encoding="utf-8"))
    record.pop("config", None)
    checkpoints = [{"name": c.name, "bytes": c.stat().st_size, "sha256": sha256(c)}
                   for c in sorted((a.run_dir / "checkpoints").glob("*.ckpt"))]
    series, steps, files = val_mel_series(a.run_dir / "lightning_logs")
    out = {"run_dir": str(a.run_dir), "run_record": record, "checkpoints": checkpoints,
           "run": {"event_files": files, "summary": describe(series), "val_loss_mel_per_epoch": series,
                   "global_step_per_epoch": steps},
           "note": "val_loss_mel on the 100-utterance canonical validation split; one training run; "
                   "best = minimum over epochs (an optimistic, noise-selected value)."}
    if a.reference_logdir:
        r_series, r_steps, r_files = val_mel_series(a.reference_logdir)
        out["reference"] = {"label": a.reference_label, "logdir": str(a.reference_logdir), "event_files": r_files,
                            "selected_checkpoint": a.reference_best_checkpoint, "summary": describe(r_series),
                            "val_loss_mel_per_epoch": r_series, "global_step_per_epoch": r_steps}
        n = min(len(series), len(r_series))
        import numpy as np
        d = np.array(series[:n]) - np.array(r_series[:n])
        out["run_minus_reference"] = {f"{lo}-{hi}": float(d[lo:hi + 1].mean()) for lo, hi in WINDOWS if n > hi}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"run": out["run"]["summary"], "reference": out.get("reference", {}).get("summary"),
                      "run_minus_reference": out.get("run_minus_reference")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
