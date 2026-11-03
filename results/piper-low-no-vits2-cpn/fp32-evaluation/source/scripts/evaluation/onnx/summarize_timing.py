"""Summarize the ONNX timing session (protocol section 5): means, order check, paired tests.

Reads every ``time_graphs.py`` record in ``--timing-dir`` and the PyTorch control synthesis of
the same session. Comparisons stay inside one thread setting and use round A; round B (reverse
order) is reported as an order check (B/A ratio of mean RTF per graph). Paired tests follow
``harvard/compare.py``: difference x - y per sentence, paired bootstrap 95% interval of the mean
difference (10,000 resamples, ``default_rng(20261004)`` per comparison), two-sided Wilcoxon.

    python scripts/evaluation/onnx/summarize_timing.py --timing-dir training_output/onnx-ptq/timing \
        --pytorch-control training_output/onnx-ptq/timing/pytorch_ctrl/synth.json \
        --earlier-control results/piper-ir-timing-session/piper_no_vits2_cpn_ctrl_synth_record.json \
        --output <dir>/timing_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

CONTROL = "piper-no-vits2-cpn"
N_BOOT, BOOT_SEED = 10_000, 20261004


def paired(x_rows, y_rows) -> dict:
    import numpy as np
    from scipy.stats import wilcoxon

    if [r["text"] for r in x_rows] != [r["text"] for r in y_rows]:
        raise SystemExit("timing rows are not paired")
    x = np.array([r["rtf"] for r in x_rows], float)
    y = np.array([r["rtf"] for r in y_rows], float)
    d = x - y
    boot = d[np.random.default_rng(BOOT_SEED).integers(0, len(d), size=(N_BOOT, len(d)))].mean(1)
    return {"mean_x": float(x.mean()), "mean_y": float(y.mean()), "mean_diff_x_minus_y": float(d.mean()),
            "relative_change_meanX_over_meanY_minus_1": float(x.mean() / y.mean() - 1),
            "paired_bootstrap95_mean_diff": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "wilcoxon_p": float(wilcoxon(x, y).pvalue), "x_faster_count": int((d < 0).sum()),
            "y_faster_count": int((d > 0).sum()), "n_pairs": len(d)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timing-dir", required=True, type=Path)
    p.add_argument("--pytorch-control", required=True, type=Path)
    p.add_argument("--earlier-control", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args(argv)
    if a.output.exists():
        p.error("output exists; choose a new path")
    import numpy as np
    import scipy

    files = sorted(a.timing_dir.glob("t*_*.json"))
    runs = {}
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        key = (r["protocol"]["intra_op_threads"], r["round"], r["label"], r["precision"])
        if key in runs:
            raise SystemExit(f"duplicate timing run {key}")
        runs[key] = (f, r)
    if any(r["summary"]["finite"] != r["summary"]["n"] for _, r in runs.values()):
        raise SystemExit("non-finite output in a timing run")
    threads = sorted({k[0] for k in runs})
    labels = sorted({k[2] for k in runs})
    out = {"classification": "new_measurement_summary_same_session", "scipy": scipy.__version__, "numpy": np.__version__,
           "inputs": {f.name: C.sha256(f) for f, _ in runs.values()},
           "method": {"comparisons": "within one thread setting, round A; round B reverses the order",
                      "bootstrap": f"paired, {N_BOOT} resamples, percentile 2.5/97.5, default_rng({BOOT_SEED}) per comparison",
                      "wilcoxon": "scipy.stats.wilcoxon defaults, two-sided",
                      "equivalence": "not tested"},
           "runs": [{"file": f.name, "threads": k[0], "round": k[1], "label": k[2], "precision": k[3],
                     "graph_sha256": r["graph_sha256"], "graph_bytes": r["graph_bytes"], **r["summary"],
                     "session_open_s": r["session_open_s"], "load_average_start": r["load_average_start"],
                     "load_average_end": r["load_average_end"]} for k, (f, r) in sorted(runs.items())],
           "order_check": {}, "int8_vs_fp32": {}, "vs_control": {}}
    for t in threads:
        rows = lambda rnd, lab, prec: runs[(t, rnd, lab, prec)][1]["rows"]  # noqa: E731
        out["order_check"][f"t{t}"] = {
            f"{lab}_{prec}": {"round_A_mean_rtf": runs[(t, "A", lab, prec)][1]["summary"]["mean_rtf"],
                              "round_B_mean_rtf": runs[(t, "B", lab, prec)][1]["summary"]["mean_rtf"],
                              "B_over_A": runs[(t, "B", lab, prec)][1]["summary"]["mean_rtf"]
                              / runs[(t, "A", lab, prec)][1]["summary"]["mean_rtf"]}
            for lab in labels for prec in ("FP32", "INT8")}
        out["int8_vs_fp32"][f"t{t}"] = {lab: {"round_A": paired(rows("A", lab, "INT8"), rows("A", lab, "FP32")),
                                              "round_B": paired(rows("B", lab, "INT8"), rows("B", lab, "FP32"))}
                                        for lab in labels}
        out["vs_control"][f"t{t}"] = {prec: {lab: paired(rows("A", lab, prec), rows("A", CONTROL, prec))
                                             for lab in labels if lab != CONTROL} for prec in ("FP32", "INT8")}
    ctrl = json.loads(a.pytorch_control.read_text(encoding="utf-8"))
    earlier = json.loads(a.earlier_control.read_text(encoding="utf-8"))
    c_rtf = float(np.mean([r["rtf"] for r in ctrl["rows"]]))
    e_rtf = float(np.mean([r["rtf"] for r in earlier["rows"]]))
    out["pytorch_control"] = {"file": a.pytorch_control.name, "sha256": C.sha256(a.pytorch_control),
                              "torch_threads": ctrl["protocol"]["torch_threads"], "mean_rtf": c_rtf,
                              "earlier_session_file": a.earlier_control.name, "earlier_session_mean_rtf": e_rtf,
                              "this_over_earlier": c_rtf / e_rtf,
                              "wavs_identical_to_earlier_session": sum(x["wav_sha256"] == y["wav_sha256"]
                                                                       for x, y in zip(ctrl["rows"], earlier["rows"]))}
    if 12 in threads:
        out["onnx_fp32_t12_over_pytorch_control"] = runs[(12, "A", CONTROL, "FP32")][1]["summary"]["mean_rtf"] / c_rtf
    C.write_json(a.output, out)
    for t in threads:
        for lab in labels:
            r = out["int8_vs_fp32"][f"t{t}"][lab]["round_A"]
            print(f"t{t} {lab}: FP32 {r['mean_y']:.5f} INT8 {r['mean_x']:.5f} ({100 * r['relative_change_meanX_over_meanY_minus_1']:+.1f}%)")
    print(f"PyTorch control {c_rtf:.5f} (earlier session {e_rtf:.5f}, x{c_rtf / e_rtf:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
