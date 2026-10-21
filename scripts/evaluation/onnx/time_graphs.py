"""Time one ONNX deployment graph on Harvard-720 (protocol section 5).

ONNX Runtime CPU, sequential, inter-op 1, ``--threads`` intra-op threads; ``onnxruntime.set_seed
(1234)`` before the session is created; one ``session.run`` per sentence in manifest order, no
warm-up; the timed region is ``session.run`` only. No WAV is written. Run it only on an idle
machine; the record keeps the load average at start and end.

    python scripts/evaluation/onnx/time_graphs.py --graph <label>_int8.onnx --threads 2 --round A \
        --output training_output/onnx-ptq/timing/t2_A_<label>_int8.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph", required=True, type=Path)
    p.add_argument("--threads", required=True, type=int)
    p.add_argument("--round", required=True)
    p.add_argument("--manifest", default=Path("results/manifests/harvard720.json"), type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args(argv)
    if a.output.exists():
        p.error("output exists; choose a new path")

    import numpy as np
    import onnx
    import onnxruntime as ort

    export = json.loads((a.graph.parent / "export_record.json").read_text(encoding="utf-8"))
    sr = export["model"]["sample_rate"]
    props = {x.key: x.value for x in onnx.load(a.graph, load_external_data=False).metadata_props}
    samples = json.loads(a.manifest.read_text(encoding="utf-8"))
    load_start = os.getloadavg()
    ort.set_seed(C.SEED)
    t_open = time.perf_counter()
    sess = C.ort_session(a.graph, a.threads)
    open_s = time.perf_counter() - t_open
    scales = np.array(C.SCALES, dtype=np.float32)
    rows = []
    t_all = time.perf_counter()
    for i, s in enumerate(samples):
        ids = np.array(s["phoneme_ids"], dtype=np.int64)[None, :]
        feeds = {"input": ids, "input_lengths": np.array([ids.shape[1]], dtype=np.int64), "scales": scales}
        t0 = time.perf_counter()
        out = sess.run(None, feeds)[0]
        dt = time.perf_counter() - t0
        duration = out.shape[-1] / sr
        rows.append({"idx": i, "text": s["text"], "synth_time_s": dt, "audio_duration_s": duration,
                     "rtf": dt / duration if duration > 0 else float("nan"), "finite": bool(np.isfinite(out).all())})
    rtf = np.array([r["rtf"] for r in rows])
    record = {"created_utc": datetime.now(timezone.utc).isoformat(), "label": export["label"],
              "model_id": export["model"]["model_id"], "checkpoint_sha256": export["model"]["checkpoint_sha256"],
              "graph": a.graph.name, "graph_sha256": C.sha256(a.graph), "graph_bytes": a.graph.stat().st_size,
              "precision": "INT8" if "INT8" in props.get("precision", "") else "FP32", "round": a.round,
              "protocol": {"runtime": "onnxruntime", "provider": "CPUExecutionProvider", "intra_op_threads": a.threads,
                           "inter_op_threads": 1, "execution_mode": "sequential", "ort_seed": C.SEED,
                           "warm_up": False, "timed_region": "session.run", "scales": list(C.SCALES),
                           "noise": "native RandomNormalLike", "document": "docs/evaluation-onnx-ptq.md"},
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "onnxruntime": ort.__version__, "numpy": np.__version__, "cpu_count": os.cpu_count()},
              "load_average_start": list(load_start), "load_average_end": list(os.getloadavg()),
              "session_open_s": open_s, "wall_time_s": time.perf_counter() - t_all,
              "manifest_sha256": C.sha256(a.manifest), "script_sha256": C.sha256(Path(__file__)),
              "summary": {"n": len(rows), "finite": sum(r["finite"] for r in rows), "mean_rtf": float(rtf.mean()),
                          "median_rtf": float(np.median(rtf))},
              "rows": rows}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"timed {a.graph.name} t{a.threads} round {a.round}: mean RTF {rtf.mean():.5f}, median {np.median(rtf):.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
