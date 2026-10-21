"""Thread scaling of single elementwise ONNX ops on decoder-sized tensors (ONNX Runtime CPU).

Checks whether an op's CPU kernel uses the intra-op thread pool. Each op runs alone in a
one-node graph on float32 input of the decoder's shapes (``--shape``; default the last decoder
stage, 32 channels x 53,760 samples, about 2.4 s of audio). For every thread count, with ORT's
default idle-thread spinning and with spinning off, the median of ``--repeats`` timed runs
after 5 warm-up runs is recorded.

    python scripts/evaluation/onnx/profile_ops.py --output training_output/onnx-ptq/profile/ops.json
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

OPS = ("Sin", "Cos", "Mul", "Pow", "Add", "LeakyRelu", "Tanh")


def graph(op: str, shape):
    from onnx import TensorProto, helper

    inputs = ["X"]
    inits = []
    if op in ("Mul", "Add"):
        inputs = ["X", "X"]
    elif op == "Pow":
        inits = [helper.make_tensor("two", TensorProto.FLOAT, [], [2.0])]
        inputs = ["X", "two"]
    node = helper.make_node(op, inputs, ["Y"], name=op)
    g = helper.make_graph([node], op, [helper.make_tensor_value_info("X", TensorProto.FLOAT, list(shape))],
                          [helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(shape))], initializer=inits)
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shape", type=int, nargs="+", default=[1, 32, 53760])
    p.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 12])
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args(argv)
    if a.output.exists():
        p.error("output exists; choose a new path")
    import numpy as np
    import onnxruntime as ort

    x = np.random.default_rng(C.SEED).standard_normal(a.shape).astype(np.float32)
    load_start = list(os.getloadavg())
    rows = []
    for op in OPS:
        model = graph(op, a.shape).SerializeToString()
        for spinning in (True, False):
            for t in a.threads:
                opt = ort.SessionOptions()
                opt.intra_op_num_threads = t
                opt.inter_op_num_threads = 1
                opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                opt.add_session_config_entry("session.intra_op.allow_spinning", "1" if spinning else "0")
                sess = ort.InferenceSession(model, sess_options=opt, providers=["CPUExecutionProvider"])
                for _ in range(5):
                    sess.run(None, {"X": x})
                times = []
                for _ in range(a.repeats):
                    t0 = time.perf_counter()
                    sess.run(None, {"X": x})
                    times.append(time.perf_counter() - t0)
                rows.append({"op": op, "threads": t, "spinning": spinning, "median_ms": 1e3 * float(np.median(times)),
                             "min_ms": 1e3 * float(np.min(times))})
    base = {(r["op"], r["spinning"]): r["median_ms"] for r in rows if r["threads"] == a.threads[0]}
    for r in rows:
        r[f"speedup_vs_{a.threads[0]}_thread"] = base[(r["op"], r["spinning"])] / r["median_ms"]
    record = {"created_utc": datetime.now(timezone.utc).isoformat(), "shape": a.shape, "repeats": a.repeats,
              "warm_up_runs": 5, "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                                 "onnxruntime": ort.__version__, "numpy": np.__version__,
                                                 "cpu_count": os.cpu_count()},
              "load_average_start": load_start, "load_average_end": list(os.getloadavg()),
              "script_sha256": C.sha256(Path(__file__)), "rows": rows}
    C.write_json(a.output, record)
    for op in OPS:
        line = "  ".join(f"t{r['threads']}{'' if r['spinning'] else 'ns'} {r['median_ms']:.2f}ms" for r in rows if r["op"] == op)
        print(f"{op:9s} {line}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
