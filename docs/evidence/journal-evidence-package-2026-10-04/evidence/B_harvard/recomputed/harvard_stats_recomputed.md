| Model | mean RTF | ratio-of-sums RTF | mean UTMOSv2 | macro WER | corpus WER (expl.) | WER=0 sentences |
|---|---:|---:|---:|---:|---:|---:|
| HiFi-GAN ResBlock2 | 0.04177 | 0.04154 | 3.0260 | 0.1476 | 0.1437 | 318 |
| Parallel-IR | 0.03388 | 0.03373 | 3.5680 | 0.1073 | 0.1033 | 381 |
| Sequential-IR | 0.03081 | 0.03058 | 3.5443 | 0.1222 | 0.1186 | 351 |
| Piper (published) | 0.03916 | 0.03897 | 3.4507 | 0.1747 | 0.1696 | 228 |

| Pair / metric | mean diff (x−y) | Wilcoxon p | MWU p | Holm p (expl.) | zero diffs | paired bootstrap 95% (expl.) |
|---|---:|---:|---:|---:|---:|---|
| seq_vs_mrf_rtf | -0.00308 | 7e-37 | 6.94e-34 | 7e-36 | 0 | [-0.00357, -0.00257] |
| seq_vs_mrf_utmosv2 | -0.02373 | 0.0471 | 0.039 | 0.0943 | 0 | [-0.04584, -0.00099] |
| seq_vs_mrf_wer | +0.01488 | 0.0523 | 0.0798 | 0.0943 | 323 | [+0.00188, +0.02768] |
| seq_vs_baseline_rtf | -0.01096 | 8.08e-105 | 7.21e-170 | 1.13e-103 | 0 | [-0.01156, -0.01040] |
| seq_vs_baseline_utmosv2 | +0.51822 | 3.77e-108 | 8.33e-159 | 5.65e-107 | 0 | [+0.49134, +0.54529] |
| seq_vs_baseline_wer | -0.02547 | 0.00116 | 0.00965 | 0.00465 | 295 | [-0.04018, -0.01103] |
| seq_vs_piper_rtf | -0.00835 | 8.1e-98 | 9.54e-141 | 1.05e-96 | 0 | [-0.00886, -0.00783] |
| seq_vs_piper_utmosv2 | +0.09357 | 2.87e-13 | 6.13e-12 | 1.72e-12 | 0 | [+0.06977, +0.11737] |
| seq_vs_piper_wer | -0.05253 | 4.72e-11 | 8.48e-11 | 2.36e-10 | 248 | [-0.06777, -0.03688] |
| mrf_vs_piper_rtf | -0.00528 | 8.49e-67 | 4.07e-82 | 9.34e-66 | 0 | [-0.00582, -0.00474] |
| mrf_vs_piper_utmosv2 | +0.11729 | 1.11e-20 | 1.44e-18 | 8.92e-20 | 0 | [+0.09432, +0.14024] |
| mrf_vs_piper_wer | -0.06741 | 1.96e-17 | 1.15e-16 | 1.37e-16 | 263 | [-0.08269, -0.05245] |
| baseline_vs_piper_rtf | +0.00261 | 3.3e-24 | 2.83e-26 | 2.97e-23 | 0 | [+0.00199, +0.00322] |
| baseline_vs_piper_utmosv2 | -0.42466 | 5.98e-96 | 1.34e-120 | 7.17e-95 | 0 | [-0.45140, -0.39721] |
| baseline_vs_piper_wer | -0.02706 | 0.00194 | 0.000235 | 0.00582 | 225 | [-0.04345, -0.01080] |

seq_vs_mrf_rtf: RTF reduction (1 − mean_x/mean_y) = 9.08%; throughput gain (mean_y/mean_x − 1) = 9.98%; ratio-of-sums reduction = 9.34%; median per-sentence ratio = 0.9027

seq_vs_piper_rtf: RTF reduction (1 − mean_x/mean_y) = 21.33%; throughput gain (mean_y/mean_x − 1) = 27.11%; ratio-of-sums reduction = 21.55%; median per-sentence ratio = 0.7788

seq_vs_baseline_rtf: RTF reduction (1 − mean_x/mean_y) = 26.25%; throughput gain (mean_y/mean_x − 1) = 35.59%; ratio-of-sums reduction = 26.40%; median per-sentence ratio = 0.7206
