"""Harvard (out-of-domain, n=720) vs canonical LJSpeech test (n=500) summary,
with Mann-Whitney U (same test as the canonical comparison) and paired
Wilcoxon signed-rank (all models read the same sentences) for key pairs,
plus bootstrap 95% CIs of the means.
"""
import json
import os

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

MODELS = ["baseline", "mrf", "seq", "piper"]
CANON = {
    "baseline": "/tmp/eval_baseline_fp32_seeded_500_results.json",
    "mrf": "/tmp/eval_mrf_fp32_seeded_500_results.json",
    "seq": "/tmp/eval_seq111_fp32_seeded_500_results.json",
    "piper": "/tmp/eval_piper_original_500_full_results.json",
}
METRICS = ["rtf", "utmosv2", "wer"]
rng = np.random.default_rng(0)


def load(path):
    rs = json.load(open(path))
    return {m: np.array([r[m] for r in rs], dtype=float) for m in METRICS}


def ci(x, n=5000):
    b = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return np.percentile(b, [2.5, 97.5])


H = {m: load(f"/tmp/harvard_{m}_results.json") for m in MODELS if os.path.exists(f"/tmp/harvard_{m}_results.json")}
C = {m: load(p) for m, p in CANON.items() if os.path.exists(p)}

print("=== Harvard n=720 (out-of-domain, unseen by every model) ===")
for m, d in H.items():
    row = [f"{m:9s} n={len(d['rtf'])}"]
    for k in METRICS:
        lo, hi = ci(d[k])
        row.append(f"{k}={d[k].mean():.4f} [{lo:.4f},{hi:.4f}]")
    print("  ".join(row))

print("\n=== Canonical LJSpeech test n=500 (for reference) ===")
for m, d in C.items():
    print(f"{m:9s} " + "  ".join(f"{k}={d[k].mean():.4f}" for k in METRICS))

print("\n=== Pairwise tests on Harvard (two-sided) ===")
for a, b in [("seq", "mrf"), ("seq", "baseline"), ("seq", "piper"), ("mrf", "piper"), ("baseline", "piper")]:
    if a not in H or b not in H:
        continue
    for k in METRICS:
        x, y = H[a][k], H[b][k]
        p_mw = mannwhitneyu(x, y, alternative="two-sided").pvalue
        d = x - y
        p_w = wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0
        print(f"{a:8s} vs {b:8s} {k:8s} diff={x.mean()-y.mean():+.4f}  MWU p={p_mw:.2e}  Wilcoxon p={p_w:.2e}")

print("\n=== Ranking per metric (Harvard / canonical) ===")
for k, better_low in [("rtf", True), ("utmosv2", False), ("wer", True)]:
    for name, D in (("harvard", H), ("canonical", C)):
        order = sorted(D, key=lambda m: D[m][k].mean(), reverse=not better_low)
        print(f"{k:8s} {name:9s}: " + " > ".join(order))
