"""Scorer control: rescore historical WAVs today and compare with the stored scores.

When a new model is scored in a later session, a difference from the historical
models could come from the scorer rather than the model. Rescoring byte-identical
historical WAVs with the same ``score.py`` separates the two: per sentence it
compares today's Whisper transcript, WER and UTMOSv2 with the stored values.

    python scripts/evaluation/harvard/scorer_control.py \
        --rescored <today's results for the historical WAVs> \
        --historical results/historical/harvard/harvard_baseline_results.json --output <scorer_control.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rescored", required=True, type=Path)
    p.add_argument("--historical", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args(argv)
    if a.output.exists():
        p.error("output exists; choose a new path")

    sys.path.insert(0, str(REPO / "src"))
    import numpy as np

    from hifimobinet.evaluation import check_pairs, load_rows

    new, old = load_rows(a.rescored), load_rows(a.historical)
    check_pairs(new, old)
    wn, wo = np.array([r["wer"] for r in new]), np.array([r["wer"] for r in old])
    un, uo = np.array([r["utmosv2"] for r in new]), np.array([r["utmosv2"] for r in old])
    du = un - uo
    same_text = sum(n["asr_text"] == o["asr_text"] for n, o in zip(new, old))
    out = {
        "classification": "new_measurement_scorer_control",
        "inputs": {k: {"file": f.name, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
                   for k, f in (("rescored", a.rescored), ("historical", a.historical))},
        "n": len(new),
        "whisper": {"identical_transcripts": int(same_text), "identical_sentence_wer": int((wn == wo).sum()),
                    "macro_wer_rescored": float(wn.mean()), "macro_wer_historical": float(wo.mean()),
                    "macro_wer_difference": float(wn.mean() - wo.mean()),
                    "max_abs_sentence_wer_difference": float(np.abs(wn - wo).max())},
        "utmosv2": {"mean_rescored": float(un.mean()), "mean_historical": float(uo.mean()),
                    "mean_difference": float(du.mean()), "sd_of_sentence_differences": float(du.std(ddof=1)),
                    "max_abs_sentence_difference": float(np.abs(du).max()),
                    "pearson_r": float(np.corrcoef(un, uo)[0, 1])},
        "interpretation": "Same WAV bytes, same scorer code and weights, different session. Differences measure "
                          "scorer run-to-run variation (UTMOSv2 random cropping is not seeded), not model differences.",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("whisper", "utmosv2")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
