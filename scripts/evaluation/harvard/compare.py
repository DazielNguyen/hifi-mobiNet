"""Compare one newly scored model with the historical Harvard-720 results, paired by sentence.

Uses the repository's audit functions (``hifimobinet.evaluation``): ``load_rows`` refuses
missing or non-finite observations, ``check_pairs`` requires identical IDs, text and
order, ``summarize`` gives the per-model aggregates and ``paired`` the two-sided Wilcoxon
signed-rank test with the historical settings. Added, as in the prior Harvard audit
(scripts/evaluation/original/recompute_B_stats.py): Mann-Whitney U with SciPy defaults
and a paired bootstrap 95% interval of the mean difference (10,000 resamples,
percentile). Each (pair, metric) uses a fresh ``numpy.random.default_rng(20261004)``,
so an interval does not depend on the order of comparisons (the prior audit shared one
stream across its 15 tests). The Holm adjustment over this file's comparisons is
exploratory. None of this is an equivalence test.

    python scripts/evaluation/harvard/compare.py \
        --candidate <results.json> --label piper_no_vits2_cpn \
        --reference-dir results/historical/harvard --output <comparison.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REFERENCES = ("baseline", "mrf", "seq", "piper")
METRICS = ("rtf", "utmosv2", "wer")
LOWER_IS_BETTER = {"rtf": True, "utmosv2": False, "wer": True}
N_BOOT, SEED = 10_000, 20261004


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def holm(pvalues: dict) -> dict:
    names = sorted(pvalues, key=pvalues.get)
    out, running = {}, 0.0
    for i, name in enumerate(names):
        running = max(running, min(1.0, (len(names) - i) * pvalues[name]))
        out[name] = running
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--label", required=True)
    p.add_argument("--reference-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args(argv)
    if a.output.exists():
        p.error("output exists; choose a new path")

    sys.path.insert(0, str(REPO / "src"))
    import numpy as np
    import scipy
    from scipy.stats import mannwhitneyu

    from hifimobinet.evaluation import check_pairs, load_rows, paired, summarize

    files = {a.label: a.candidate, **{m: a.reference_dir / f"harvard_{m}_results.json" for m in REFERENCES}}
    rows = {m: load_rows(f) for m, f in files.items()}
    for m in REFERENCES:
        check_pairs(rows[a.label], rows[m])

    summary = {}
    for m, r in rows.items():
        s = summarize(r)
        s["median_rtf"] = float(np.median([x["rtf"] for x in r]))
        s["sd_utmosv2"] = float(np.std([x["utmosv2"] for x in r], ddof=1))
        s["sentences_with_wer_0"] = sum(x["wer"] == 0 for x in r)
        summary[m] = s

    tests = {}
    for other in REFERENCES:
        for k in METRICS:
            x = np.array([r[k] for r in rows[a.label]], float)
            y = np.array([r[k] for r in rows[other]], float)
            d = x - y
            rng = np.random.default_rng(SEED)
            boot = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(1)
            w = paired(rows[a.label], rows[other], k)
            better = (d < 0) if LOWER_IS_BETTER[k] else (d > 0)
            worse = (d > 0) if LOWER_IS_BETTER[k] else (d < 0)
            t = {"mean_x": float(x.mean()), "mean_y": float(y.mean()), "mean_diff_x_minus_y": float(d.mean()),
                 "median_paired_diff": float(np.median(d)), "n_pairs": len(d), "n_zero_differences": int((d == 0).sum()),
                 "candidate_better_count": int(better.sum()), "candidate_worse_count": int(worse.sum()),
                 "wilcoxon_p": w["wilcoxon_p"], "wilcoxon_settings": w["settings"],
                 "mannwhitney_p_scipy_default": float(mannwhitneyu(x, y, alternative="two-sided").pvalue),
                 "paired_bootstrap95_mean_diff": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
            if k == "rtf":
                t["rtf_reduction_1_minus_meanX_over_meanY"] = float(1 - x.mean() / y.mean())
            tests[f"{a.label}_vs_{other}_{k}"] = t
    for name, value in holm({n: t["wilcoxon_p"] for n, t in tests.items()}).items():
        tests[name]["exploratory_holm_adjusted_wilcoxon_p"] = value

    out = {"classification": "new_comparison_of_new_and_historical_observations",
           "inputs": {m: {"file": f.name, "sha256": sha256(f)} for m, f in files.items()},
           "scipy": scipy.__version__, "numpy": np.__version__,
           "method": {"direction": "x = candidate, y = reference; difference = x - y",
                      "better": "lower RTF and WER, higher UTMOSv2",
                      "bootstrap": f"paired, {N_BOOT} resamples, percentile 2.5/97.5, default_rng({SEED}) per comparison",
                      "holm": f"exploratory, over the {len(tests)} comparisons in this file",
                      "equivalence": "not tested; no margin was pre-specified"},
           "summary": summary, "paired_tests": tests}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=1, allow_nan=False), encoding="utf-8")
    for other in REFERENCES:
        print(other, {k: round(tests[f"{a.label}_vs_{other}_{k}"]["mean_diff_x_minus_y"], 5) for k in METRICS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
