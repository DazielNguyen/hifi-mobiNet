"""RECOMPUTED in this evidence round from the copied per-sentence Harvard JSON files.
Does not rerun synthesis, ASR or UTMOSv2. Two blocks:

1. REPLICATION of the tests reported in the 2026-10-04 conversation (stats.py):
   scipy.stats.wilcoxon(x, y) and mannwhitneyu(x, y, alternative="two-sided") with
   scipy defaults, no multiple-comparison correction. Defaults are spelled out below.
2. EXPLORATORY additions, clearly labelled: zero/tie counts, Holm adjustment over the
   15 reported tests, paired bootstrap CIs of mean differences, RTF reduction vs
   throughput gain, ratio-of-sums RTF, corpus WER from stored transcripts.
   These are not a pre-specified protocol and do not establish equivalence.
Usage: recompute_B_stats.py <package_root>
"""
import json
import os
import sys

import jiwer
import numpy as np
import scipy
from scipy.stats import mannwhitneyu, wilcoxon

import argparse
parser = argparse.ArgumentParser(description="Reproduce the prior Harvard statistical audit; no new synthesis or scoring")
parser.add_argument("--input-dir", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
RAW = args.input_dir
OUT = args.output_dir
if os.path.abspath(RAW) == os.path.abspath(OUT):
    parser.error("Output must be separate from historical inputs")
for name in ("harvard_stats_recomputed.json", "harvard_stats_recomputed.md"):
    if os.path.exists(os.path.join(OUT, name)):
        parser.error("Output exists; select a new output directory")
os.makedirs(OUT, exist_ok=True)
MODELS = ["baseline", "mrf", "seq", "piper"]
LABEL = {"baseline": "HiFi-GAN ResBlock2", "mrf": "Parallel-IR", "seq": "Sequential-IR", "piper": "Piper (published)"}
R = {m: json.load(open(os.path.join(RAW, f"harvard_{m}_results.json"))) for m in MODELS}
A = {m: {k: np.array([r[k] for r in R[m]], float) for k in ("rtf", "utmosv2", "wer", "synth_time_s", "audio_duration_s")} for m in MODELS}

tf = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(),
                    jiwer.Strip(), jiwer.ReduceToListOfListOfWords()])
out = {"scipy_version": scipy.__version__, "jiwer_version": jiwer.__version__ if hasattr(jiwer, "__version__") else None,
       "n_per_model": {m: len(R[m]) for m in MODELS}}

# Per-model summaries (WER is a fraction per sentence; macro = mean over sentences)
summ = {}
for m in MODELS:
    a = A[m]
    refs = [r["text"] for r in R[m]]
    hyps = [r["asr_text"] for r in R[m]]
    per = np.array([jiwer.wer(r, h, reference_transform=tf, hypothesis_transform=tf) for r, h in zip(refs, hyps)])
    o = jiwer.process_words(refs, hyps, reference_transform=tf, hypothesis_transform=tf)
    n_ref = o.hits + o.substitutions + o.deletions
    summ[m] = {
        "label": LABEL[m],
        "mean_rtf_per_sentence": a["rtf"].mean(), "median_rtf": float(np.median(a["rtf"])),
        "ratio_of_sums_rtf": a["synth_time_s"].sum() / a["audio_duration_s"].sum(),
        "mean_utmosv2": a["utmosv2"].mean(), "sd_utmosv2": a["utmosv2"].std(ddof=1),
        "macro_wer_fraction": a["wer"].mean(),
        "macro_wer_recomputed_from_transcripts": per.mean(),
        "max_abs_diff_stored_vs_recomputed_wer": float(np.abs(per - a["wer"]).max()),
        "corpus_wer_fraction_exploratory": (o.substitutions + o.deletions + o.insertions) / n_ref,
        "S_D_I_H": [o.substitutions, o.deletions, o.insertions, o.hits],
        "sentences_with_wer_0": int((a["wer"] == 0).sum()),
        "mean_audio_duration_s": a["audio_duration_s"].mean(),
    }
out["per_model"] = summ

PAIRS = [("seq", "mrf"), ("seq", "baseline"), ("seq", "piper"), ("mrf", "piper"), ("baseline", "piper")]
rng = np.random.default_rng(20261004)
tests, flat = {}, []
for a, b in PAIRS:
    for k in ("rtf", "utmosv2", "wer"):
        x, y = A[a][k], A[b][k]
        d = x - y
        w = wilcoxon(x, y)  # defaults: zero_method='wilcox' (drops zero diffs), correction=False, two-sided, method='auto'
        u = mannwhitneyu(x, y, alternative="two-sided")  # defaults: use_continuity=True, method='auto'
        idx = rng.integers(0, len(d), size=(10000, len(d)))
        boot = d[idx].mean(1)
        rec = {
            "mean_x": x.mean(), "mean_y": y.mean(), "mean_diff_x_minus_y": d.mean(), "median_paired_diff": float(np.median(d)),
            "wilcoxon_p_scipy_default": w.pvalue, "wilcoxon_statistic": float(w.statistic),
            "mannwhitney_p_scipy_default": u.pvalue,
            "n_pairs": len(d), "n_zero_differences_dropped_by_wilcoxon": int((d == 0).sum()),
            "n_tied_abs_differences": int(len(d[d != 0]) - len(np.unique(np.abs(d[d != 0])))),
            "exploratory_paired_bootstrap95_mean_diff": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        }
        if k == "rtf":
            rec["rtf_reduction_1_minus_meanX_over_meanY"] = 1 - x.mean() / y.mean()
            rec["throughput_gain_meanY_over_meanX_minus_1"] = y.mean() / x.mean() - 1
            rs_x = A[a]["synth_time_s"].sum() / A[a]["audio_duration_s"].sum()
            rs_y = A[b]["synth_time_s"].sum() / A[b]["audio_duration_s"].sum()
            rec["rtf_reduction_ratio_of_sums"] = 1 - rs_x / rs_y
            rec["median_per_sentence_rtf_ratio_x_over_y"] = float(np.median(x / y))
        tests[f"{a}_vs_{b}_{k}"] = rec
        flat.append((f"{a}_vs_{b}_{k}", w.pvalue))

# Exploratory Holm adjustment over the 15 Wilcoxon tests reported together
order = sorted(range(len(flat)), key=lambda i: flat[i][1])
m_ = len(flat)
adj, running = [None] * m_, 0.0
for rank, i in enumerate(order):
    running = max(running, min(1.0, (m_ - rank) * flat[i][1]))
    adj[i] = running
for (name, p), pa in zip(flat, adj):
    tests[name]["exploratory_holm_adjusted_wilcoxon_p_over_15_tests"] = pa
out["paired_tests"] = tests
out["notes"] = {
    "unit_of_analysis": "sentence (720 paired observations); one synthesis draw per sentence and model; one scorer pass; one timing pass; one training run per model",
    "wer_unit": "fraction of reference words per sentence, averaged over sentences (macro); 0.05 = 5 percentage points absolute",
    "rtf": "per-sentence synth_time_s / audio_duration_s, then arithmetic mean over sentences",
    "equivalence": "no margin was pre-specified; non-significant or borderline p-values do not establish equivalence",
}
json.dump(out, open(os.path.join(OUT, "harvard_stats_recomputed.json"), "w"), indent=1, default=float)

lines = ["| Model | mean RTF | ratio-of-sums RTF | mean UTMOSv2 | macro WER | corpus WER (expl.) | WER=0 sentences |", "|---|---:|---:|---:|---:|---:|---:|"]
for m in MODELS:
    s = summ[m]
    lines.append(f"| {s['label']} | {s['mean_rtf_per_sentence']:.5f} | {s['ratio_of_sums_rtf']:.5f} | {s['mean_utmosv2']:.4f} | {s['macro_wer_fraction']:.4f} | {s['corpus_wer_fraction_exploratory']:.4f} | {s['sentences_with_wer_0']} |")
lines += ["", "| Pair / metric | mean diff (x−y) | Wilcoxon p | MWU p | Holm p (expl.) | zero diffs | paired bootstrap 95% (expl.) |", "|---|---:|---:|---:|---:|---:|---|"]
for name, t in tests.items():
    lines.append(f"| {name} | {t['mean_diff_x_minus_y']:+.5f} | {t['wilcoxon_p_scipy_default']:.3g} | {t['mannwhitney_p_scipy_default']:.3g} | {t['exploratory_holm_adjusted_wilcoxon_p_over_15_tests']:.3g} | {t['n_zero_differences_dropped_by_wilcoxon']} | [{t['exploratory_paired_bootstrap95_mean_diff'][0]:+.5f}, {t['exploratory_paired_bootstrap95_mean_diff'][1]:+.5f}] |")
for p in ("seq_vs_mrf_rtf", "seq_vs_piper_rtf", "seq_vs_baseline_rtf"):
    t = tests[p]
    lines.append(f"\n{p}: RTF reduction (1 − mean_x/mean_y) = {t['rtf_reduction_1_minus_meanX_over_meanY']*100:.2f}%; "
                 f"throughput gain (mean_y/mean_x − 1) = {t['throughput_gain_meanY_over_meanX_minus_1']*100:.2f}%; "
                 f"ratio-of-sums reduction = {t['rtf_reduction_ratio_of_sums']*100:.2f}%; median per-sentence ratio = {t['median_per_sentence_rtf_ratio_x_over_y']:.4f}")
open(os.path.join(OUT, "harvard_stats_recomputed.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
