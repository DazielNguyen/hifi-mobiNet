"""Same-session control for a Harvard-720 timing run.

RTF depends on the state of the workstation, so a new model's RTF is not comparable
with RTF values recorded on another day. The control re-synthesizes a historical
model with its original, unmodified synthesis script in the same session; this tool
then records:

- whether every control WAV is byte-identical to the historical WAV of the same
  sentence (synthesis reproducibility);
- the control's RTF against the historical RTF of that model (session drift);
- a paired RTF comparison of the candidate with the control (same session):
  mean difference, paired bootstrap 95% interval (10,000 resamples,
  ``default_rng(20261004)``), two-sided Wilcoxon, sentences where each is faster.

Host paths are not copied into the output.

    python scripts/evaluation/harvard/session_control.py \
        --candidate-synth <run>/eval/harvard720/synth.json \
        --control-synth <historical-format synth JSON of the control run> --control-wav-dir <dir> \
        --historical-wav-dir <dir> --historical-results results/historical/harvard/harvard_baseline_results.json \
        --control-script <original synth script> --output <session_control.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

N_BOOT, SEED = 10_000, 20261004


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidate-synth", required=True, type=Path)
    p.add_argument("--control-synth", required=True, type=Path)
    p.add_argument("--control-wav-dir", required=True, type=Path)
    p.add_argument("--historical-wav-dir", required=True, type=Path)
    p.add_argument("--historical-results", required=True, type=Path)
    p.add_argument("--control-script", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args(argv)
    if a.output.exists():
        p.error("output exists; choose a new path")

    import numpy as np
    from scipy.stats import wilcoxon

    cand = json.loads(a.candidate_synth.read_text(encoding="utf-8"))["rows"]
    ctrl = json.loads(a.control_synth.read_text(encoding="utf-8"))
    hist = json.loads(a.historical_results.read_text(encoding="utf-8"))
    for name, rows in (("control", ctrl), ("historical", hist)):
        if [(r["idx"], r["text"]) for r in rows] != [(r["idx"], r["text"]) for r in cand]:
            raise SystemExit(f"{name}: sentence ids/text/order differ from the candidate")

    identical = [sha256(a.control_wav_dir / f"sample_{i:04d}.wav") == sha256(a.historical_wav_dir / f"sample_{i:04d}.wav")
                 for i in range(len(ctrl))]
    durations_equal = all(abs(c["audio_duration_s"] - h["audio_duration_s"]) < 1e-12 for c, h in zip(ctrl, hist))

    x = np.array([r["rtf"] for r in cand], float)
    y = np.array([r["rtf"] for r in ctrl], float)
    h = np.array([r["rtf"] for r in hist], float)
    d = x - y
    boot = d[np.random.default_rng(SEED).integers(0, len(d), size=(N_BOOT, len(d)))].mean(1)
    out = {
        "classification": "new_measurement_same_session_control",
        "control_script_sha256": sha256(a.control_script),
        "control_synth_rows": [{k: r[k] for k in ("idx", "synth_time_s", "audio_duration_s", "rtf")} for r in ctrl],
        "historical_results": {"file": a.historical_results.name, "sha256": sha256(a.historical_results)},
        "reproducibility": {"wavs_byte_identical_to_historical": int(sum(identical)), "n": len(identical),
                            "audio_durations_equal_to_historical": durations_equal},
        "session_drift": {"control_mean_rtf": float(y.mean()), "historical_mean_rtf": float(h.mean()),
                          "control_over_historical": float(y.mean() / h.mean())},
        "candidate_vs_control_rtf": {
            "candidate_mean_rtf": float(x.mean()), "control_mean_rtf": float(y.mean()),
            "mean_diff_candidate_minus_control": float(d.mean()),
            "relative_reduction_1_minus_meanX_over_meanY": float(1 - x.mean() / y.mean()),
            "paired_bootstrap95_mean_diff": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "wilcoxon_p": float(wilcoxon(x, y).pvalue),
            "candidate_faster_count": int((d < 0).sum()), "control_faster_count": int((d > 0).sum()),
            "limits": "one timing pass per sentence, no warm-up, run back to back on the same workstation; "
                      "the two models run through different code bases (harness/EdgeTTS vs BanhmiTTS) with the same torch build",
        },
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("reproducibility", "session_drift", "candidate_vs_control_rtf")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
