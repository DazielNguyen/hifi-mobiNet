import json
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

def load(path):
    with open(path) as f:
        return json.load(f)

seq_fp32 = load("/tmp/eval_seq_final_fp32_onnx_results.json")
seq_int8 = load("/tmp/eval_seq_final_int8_onnx_results.json")

# baseline/MRF result files from the earlier investigation (same n=500 canonical test set)
paths = {
    "baseline_fp32": "/tmp/eval_baseline_fp32_onnx_500_results.json",
    "baseline_int8": "/tmp/eval_baseline_ptq_onnx_500_results.json",
    "mrf_fp32": "/tmp/eval_mrf_fp32_onnx_500_results.json",
    "mrf_int8": "/tmp/eval_mrf_ptq_onnx_500_rerun_results.json",
}
others = {k: load(v) for k, v in paths.items()}

def arr(results, key):
    return np.array([r[key] for r in results])

def bootstrap_ci(a, b, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    # unpaired (different test-set orderings historically for baseline/MRF vs SEQ's own run)
    n_a, n_b = len(a), len(b)
    diffs = []
    for _ in range(n_boot):
        sa = rng.choice(a, size=n_a, replace=True)
        sb = rng.choice(b, size=n_b, replace=True)
        diffs.append(sa.mean() - sb.mean())
    diffs = np.array(diffs)
    return (a.mean() - b.mean()), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

def compare(name_seq, seq_results, name_other, other_results, metric):
    a = arr(seq_results, metric)
    b = arr(other_results, metric)
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    mean_diff, lo, hi = bootstrap_ci(a, b)
    direction = "SEQ tot hon" if (mean_diff < 0 if metric in ("rtf", "wer") else mean_diff > 0) else f"{name_other} tot hon"
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
    print(f"  {metric:10s} {name_seq}={a.mean():.4f}  {name_other}={b.mean():.4f}  diff={mean_diff:+.4f} CI=[{lo:+.4f},{hi:+.4f}]  p={p:.3e} {sig}  -> {direction}")

print("=== SEQ vs baseline/MRF, FP32 (Mann-Whitney U, unpaired, n=500 each) ===")
for metric in ["rtf", "utmosv2", "wer"]:
    compare("SEQ", seq_fp32, "baseline", others["baseline_fp32"], metric)
    compare("SEQ", seq_fp32, "MRF", others["mrf_fp32"], metric)

print("\n=== SEQ vs baseline/MRF, PTQ INT8 ===")
for metric in ["rtf", "utmosv2", "wer"]:
    compare("SEQ", seq_int8, "baseline", others["baseline_int8"], metric)
    compare("SEQ", seq_int8, "MRF", others["mrf_int8"], metric)
