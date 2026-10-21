"""Per-node ONNX Runtime profile of deployment graphs: where the time goes, by module and op type.

Follow-up to the timing session of docs/evaluation-onnx-ptq.md, to explain why the decoders
rank differently at 2 and 12 intra-op threads. For each graph and thread count:
- the session settings of ``time_graphs.py`` (sequential, inter-op 1, ``set_seed(1234)``), plus
  ORT's own profiler;
- a fixed subset of Harvard-720 (every 24th sentence, 30 sentences), after one untimed
  warm-up run of the first sentence (the warm-up is dropped from the totals);
- the kernel time of every node is summed by top-level module (``enc_p``, ``dp``, ``flow``,
  ``dec``, other) and by op type within each module;
- the optimized graph ORT actually runs is saved once per graph and counted by op type, so that
  fusions are visible.

Profiling adds overhead; compare the shares and the 2-vs-12-thread ratios, not the absolute
RTF values with ``time_graphs.py``.

    python scripts/evaluation/onnx/profile_graphs.py --graph <label>_fp32.onnx --threads 2 12 \
        --out-dir training_output/onnx-ptq/profile
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

STEP = 24


def optimized_op_counts(graph: Path, threads: int, tmp: Path) -> dict:
    import onnx
    import onnxruntime as ort

    opt = ort.SessionOptions()
    opt.intra_op_num_threads = threads
    opt.optimized_model_filepath = str(tmp / "optimized.onnx")
    ort.InferenceSession(str(graph), sess_options=opt, providers=["CPUExecutionProvider"])
    model = onnx.load(opt.optimized_model_filepath)
    by_module = defaultdict(Counter)
    fused_activation = Counter()
    for n in model.graph.node:
        by_module[C.module_of(n.name) or "other"][n.op_type] += 1
        if n.op_type == "FusedConv":
            act = next((a.s.decode() for a in n.attribute if a.name == "activation"), "")
            fused_activation[f"{C.module_of(n.name) or 'other'}:{act}"] += 1
    return {"by_module": {m: dict(sorted(c.items())) for m, c in sorted(by_module.items())},
            "fused_conv_activation": dict(sorted(fused_activation.items()))}


def profile(graph: Path, threads: int, samples, tmp: Path, spinning: bool) -> dict:
    import numpy as np
    import onnxruntime as ort

    opt = ort.SessionOptions()
    opt.intra_op_num_threads = threads
    opt.inter_op_num_threads = 1
    opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # ORT's default lets idle intra-op threads spin; "0" makes them wait instead
    opt.add_session_config_entry("session.intra_op.allow_spinning", "1" if spinning else "0")
    opt.enable_profiling = True
    opt.profile_file_prefix = str(tmp / f"prof_t{threads}")
    ort.set_seed(C.SEED)
    sess = ort.InferenceSession(str(graph), sess_options=opt, providers=["CPUExecutionProvider"])
    scales = np.array(C.SCALES, dtype=np.float32)
    seconds = 0
    for s in samples:
        ids = np.array(s["phoneme_ids"], dtype=np.int64)[None, :]
        out = sess.run(None, {"input": ids, "input_lengths": np.array([ids.shape[1]], dtype=np.int64), "scales": scales})[0]
        seconds += out.shape[-1] / 22050
    trace = json.loads(Path(sess.end_profiling()).read_text())
    runs = sorted((e for e in trace if e.get("cat") == "Session" and e.get("name") == "model_run"), key=lambda e: e["ts"])
    if len(runs) != len(samples):
        raise SystemExit(f"expected {len(samples)} model_run events, found {len(runs)}")
    first_end = runs[0]["ts"] + runs[0]["dur"]
    kernels = [e for e in trace if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time") and e["ts"] > first_end]
    module_us, op_us = Counter(), defaultdict(Counter)
    for e in kernels:
        node = e["name"][: -len("_kernel_time")]
        mod = C.module_of(node) or "other"
        module_us[mod] += e["dur"]
        op_us[mod][e["args"].get("op_name", "?")] += e["dur"]
    run_us = sum(r["dur"] for r in runs[1:])
    timed_audio = seconds - samples[0]["_seconds"]
    return {"threads": threads, "intra_op_spinning": spinning,
            "sentences_profiled": len(samples) - 1, "audio_seconds": timed_audio,
            "model_run_us": run_us, "kernel_us": sum(module_us.values()),
            "rtf_under_profiling": run_us / 1e6 / timed_audio,
            "module_us": dict(sorted(module_us.items())),
            "module_share": {m: v / sum(module_us.values()) for m, v in sorted(module_us.items())},
            "op_us_by_module": {m: dict(c.most_common()) for m, c in sorted(op_us.items())}}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph", required=True, type=Path, action="append")
    p.add_argument("--threads", required=True, type=int, nargs="+")
    p.add_argument("--manifest", default=Path("results/manifests/harvard720.json"), type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--no-spinning", action="store_true",
                   help="set session.intra_op.allow_spinning=0 (ORT default: idle threads spin)")
    a = p.parse_args(argv)
    import numpy as np
    import onnxruntime as ort

    harvard = json.loads(a.manifest.read_text(encoding="utf-8"))
    subset = [harvard[0]] + harvard[::STEP]  # warm-up (dropped) + 30 sentences
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for graph in a.graph:
        out = a.out_dir / f"profile_{graph.stem}{'_nospin' if a.no_spinning else ''}.json"
        if out.exists():
            raise SystemExit(f"{out} exists; choose a new directory")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # audio length of the warm-up run, to exclude it from the RTF denominator
            ort.set_seed(C.SEED)
            probe = C.ort_session(graph, 2)
            ids = np.array(subset[0]["phoneme_ids"], dtype=np.int64)[None, :]
            first = probe.run(None, {"input": ids, "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
                                     "scales": np.array(C.SCALES, dtype=np.float32)})[0]
            samples = [dict(s) for s in subset]
            samples[0]["_seconds"] = first.shape[-1] / 22050
            record = {"created_utc": datetime.now(timezone.utc).isoformat(), "graph": graph.name,
                      "graph_sha256": C.sha256(graph), "subset": {"rule": f"warm-up idx 0, then idx 0::{STEP}",
                                                                  "idx": [s["idx"] for s in subset[1:]]},
                      "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                      "onnxruntime": ort.__version__, "cpu_count": os.cpu_count()},
                      "load_average_start": list(os.getloadavg()), "script_sha256": C.sha256(Path(__file__)),
                      "optimized_graph": optimized_op_counts(graph, 2, tmp),
                      "profiles": [profile(graph, t, samples, tmp, not a.no_spinning) for t in a.threads],
                      "load_average_end": list(os.getloadavg())}
        C.write_json(out, record)
        line = ", ".join(f"t{r['threads']}: " + " ".join(f"{m} {100 * v:.0f}%" for m, v in r["module_share"].items())
                         + f" RTF {r['rtf_under_profiling']:.4f}" for r in record["profiles"])
        print(f"{graph.name}: {line}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
