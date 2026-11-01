"""Read-only runtime probe of the suspected empty branch; run only on N100."""
import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

sys.dont_write_bytecode = True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    bundle = args.bundle.resolve()
    sys.path.insert(0, str(bundle))
    spec = importlib.util.spec_from_file_location("_audit_suite", bundle / "openvino_suite.py")
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    env = suite.environment()
    record = {"status": "running", "environment": env, "started_utc": suite.utc(),
              "purpose": "test the empty masked-branch hypothesis; no quality/timing certification",
              "node": "/dp/flows.7/Slice_23", "model": "piper-seqir-no-vits2-cpn",
              "rows": [], "empty_count": 0,
              "rng": "fresh native OpenVINO RNG; not identical to the earlier calibration"}
    try:
        import numpy as np
        import openvino as ov
        import onnx_reference as reference
        data = suite.manifest(bundle)
        item = next(m for m in data["models"] if m["id"] == record["model"])
        source = suite.verify_file(bundle, item["fp32"])
        record["source_sha256"] = suite.sha(source)
        cal = [json.loads(s) for s in suite.verify_file(bundle, data["calibration"]).read_text().splitlines() if s.strip()]
        subset = [json.loads(s) for s in suite.verify_file(bundle, data["calibration_dataset_rows"]).read_text().splitlines() if s.strip()]
        check = reference.check_calibration(cal, suite.read(suite.verify_file(bundle, data["split"])), subset)
        if not check["passed"] or check["n"] != 60:
            raise ValueError("not the fixed 60 train-only calibration rows")
        record["calibration"] = check
        record["calibration_sha256"] = data["calibration"]["sha256"]
        core = ov.Core()
        model = core.read_model(str(source))
        nodes = [n for n in model.get_ops() if n.get_friendly_name() == record["node"]]
        if len(nodes) != 1 or nodes[0].get_output_size() != 1:
            raise ValueError("cannot find the single Slice_23 output; no guessed replacement")
        record["ov_op_type"] = nodes[0].get_type_name()
        record["partial_shape"] = str(nodes[0].output(0).get_partial_shape())
        probe_index = len(model.outputs)
        model.add_outputs([nodes[0].output(0)])
        compiled = core.compile_model(model, "CPU", suite.config(4))
        request = compiled.create_infer_request()
        for i, row in enumerate(cal):
            request.infer(suite.feeds(row))
            a = np.asarray(request.get_output_tensor(probe_index).data)
            finite = bool(np.isfinite(a).all())
            entry = {"calibration_index": i, "shape": list(a.shape), "size": int(a.size),
                     "dtype": str(a.dtype), "all_finite": finite}
            if a.size and finite:
                entry.update(minimum=float(a.min()), maximum=float(a.max()))
            record["rows"].append(entry)
            if a.size == 0:
                record["empty_count"] += 1
            if not finite:
                raise ValueError("non-finite actual branch output; empty-only hypothesis is insufficient")
            suite.audio_info(request.get_output_tensor(0).data, item["sample_rate"])
            print(f"AUDIT {i+1}/60: size={a.size}, finite={finite}", flush=True)
        if suite.sha(source) != record["source_sha256"]:
            raise ValueError("source graph changed during audit")
        record["status"] = "completed"
        print("EMPTY BRANCH OBSERVATIONS:", record["empty_count"], "/ 60", flush=True)
    except BaseException:
        record.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        record["finished_utc"] = suite.utc()
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(record, f, indent=2, allow_nan=False)
            f.write("\n")

if __name__ == "__main__":
    main()

