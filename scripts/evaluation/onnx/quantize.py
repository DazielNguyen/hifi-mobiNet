"""Static INT8 PTQ of one FP32 deployment graph with the fixed recipe (protocol section 3).

Recipe as the historical baseline/MRF INT8 graphs (J-C024): ``quantize_static``, QOperator,
weights QInt8 per channel, activations QUInt8, MinMax calibration, ``Conv`` nodes under
``/enc_p/``, ``/dp/`` and ``/flow/`` only. Calibration: the 60 canonical-train sentences of
``calibration60.candidate.jsonl``, checked against the split and ``dataset.jsonl`` first.

Writes into ``--out-dir``: ``<label>_int8.onnx`` (deployment), ``<label>_int8_matched_noise.onnx``
(validation / quality evaluation) and ``quantize_record.json``.

    python scripts/evaluation/onnx/quantize.py --label piper-seqir-no-vits2-cpn \
        --fp32 training_output/onnx-ptq/piper-seqir-no-vits2-cpn/piper-seqir-no-vits2-cpn_fp32.onnx \
        --dataset data/ljspeech-medium/training_output/ljspeech/medium/dataset.jsonl \
        --out-dir training_output/onnx-ptq/piper-seqir-no-vits2-cpn
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def initializers_used_by(model, module: str) -> dict:
    from onnx import numpy_helper

    names = {i for n in model.graph.node if C.module_of(n.name) == module for i in n.input}
    return {i.name: numpy_helper.to_array(i).tobytes() for i in model.graph.initializer if i.name in names}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", required=True)
    p.add_argument("--fp32", required=True, type=Path)
    p.add_argument("--calibration", default=C.CALIBRATION, type=Path)
    p.add_argument("--split", default=Path("results/manifests/canonical_split.json"), type=Path)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--manifest", default=Path("results/manifests/harvard720.json"), type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    a = p.parse_args(argv)
    int8_path = a.out_dir / f"{a.label}_int8.onnx"
    matched_path = a.out_dir / f"{a.label}_int8_matched_noise.onnx"
    record_path = a.out_dir / "quantize_record.json"
    if any(x.exists() for x in (int8_path, matched_path, record_path)):
        p.error("INT8 outputs exist; choose a new directory")

    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)

    record = {"label": a.label, "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
              "protocol": "docs/evaluation-onnx-ptq.md", "script_sha256": C.sha256(Path(__file__)),
              "common_sha256": C.sha256(Path(C.__file__)),
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "onnx": onnx.__version__, "onnxruntime": ort.__version__, "numpy": np.__version__}}

    def check(name, ok, **detail):
        record.setdefault("checks", {})[name] = {"passed": bool(ok), **detail}
        C.write_json(record_path, record)
        if not ok:
            raise SystemExit(f"check failed: {name} {detail}")

    try:
        # 1. calibration set: canonical train only, phoneme IDs equal to dataset.jsonl
        rows = [json.loads(line) for line in a.calibration.read_text(encoding="utf-8").splitlines() if line.strip()]
        split = json.loads(a.split.read_text(encoding="utf-8"))
        dataset = [json.loads(line) for line in a.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
        cal = C.check_calibration(rows, split, dataset)
        record["calibration"] = {"file": C.CALIBRATION.relative_to(C.REPO).as_posix() if a.calibration == C.CALIBRATION
                                 else str(a.calibration), "sha256": C.sha256(a.calibration),
                                 "split": str(a.split), "split_sha256": C.sha256(a.split),
                                 "dataset_sha256": C.sha256(a.dataset), "order": "file order",
                                 "scales": list(C.SCALES), "ort_seed": C.SEED, "ids": [r["id"] for r in rows],
                                 **{k: v for k, v in cal.items() if k != "passed"}}
        check("calibration_train_only", cal["passed"] and cal["n"] == 60, n=cal["n"], problems=cal["problems"])

        # 2. scope
        fp32 = onnx.load(a.fp32)
        quantize_nodes, exclude_nodes = C.ptq_nodes([(n.name, n.op_type) for n in fp32.graph.node])
        record["fp32"] = {"file": a.fp32.name, "sha256": C.sha256(a.fp32), "bytes": a.fp32.stat().st_size}
        record["recipe"] = {"method": "onnxruntime.quantization.quantize_static", "quant_format": "QOperator",
                            "weight_type": "QInt8", "activation_type": "QUInt8", "per_channel": True,
                            "calibrate_method": "MinMax", "op_types_to_quantize": list(C.PTQ_OP_TYPES),
                            "scope_modules": list(C.PTQ_SCOPE),
                            "conv_in_scope_by_module": dict(sorted(Counter(C.module_of(n) for n in quantize_nodes).items())),
                            "conv_excluded_by_module": dict(sorted(Counter(C.module_of(n) for n in exclude_nodes).items())),
                            "nodes_to_exclude": exclude_nodes}
        C.write_json(record_path, record)

        class Reader(CalibrationDataReader):
            def __init__(self):
                self.rows = list(rows)

            def get_next(self):
                if not self.rows:
                    return None
                r = self.rows.pop(0)
                ids = np.array(r["phoneme_ids"], dtype=np.int64)[None, :]
                return {"input": ids, "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
                        "scales": np.array(C.SCALES, dtype=np.float32)}

        def run(out: Path):
            ort.set_seed(C.SEED)
            quantize_static(str(a.fp32), str(out), calibration_data_reader=Reader(), quant_format=QuantFormat.QOperator,
                            weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                            op_types_to_quantize=list(C.PTQ_OP_TYPES), nodes_to_exclude=exclude_nodes,
                            per_channel=True, calibrate_method=CalibrationMethod.MinMax)

        # 3. quantize twice; the two raw outputs must be byte-identical
        a.out_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=a.out_dir) as tmp:
            first, second = Path(tmp) / "first.onnx", Path(tmp) / "second.onnx"
            run(first)
            run(second)
            check("quantization_reproducible", C.sha256(first) == C.sha256(second), raw_sha256=C.sha256(first))
            int8 = onnx.load(first)
        record["raw_quantized_sha256"] = record["checks"]["quantization_reproducible"]["raw_sha256"]

        # 4. structure: quantized convs by module, decoder untouched, random nodes kept
        qconv = Counter(C.module_of(n.name) for n in int8.graph.node if n.op_type == "QLinearConv")
        conv_left = Counter(C.module_of(n.name) for n in int8.graph.node if n.op_type == "Conv")
        record["int8_structure"] = {"qlinearconv_by_module": dict(sorted(qconv.items())),
                                    "fp32_conv_left_by_module": dict(sorted(conv_left.items())),
                                    "op_counts": dict(sorted(Counter(n.op_type for n in int8.graph.node).items()))}
        check("every_in_scope_conv_quantized", dict(qconv) == record["recipe"]["conv_in_scope_by_module"],
              qlinearconv=dict(qconv))
        check("no_quantized_node_under_dec",
              not any(C.module_of(n.name) == "dec" and ("QLinear" in n.op_type or "Quantize" in n.op_type)
                      for n in int8.graph.node))
        check("decoder_initializers_byte_identical_to_fp32",
              initializers_used_by(int8, "dec") == initializers_used_by(fp32, "dec"),
              decoder_initializers=len(initializers_used_by(fp32, "dec")))
        C.random_nodes(int8.graph)
        check("two_random_nodes_kept", True)

        # 5. final deployment graph (metadata added) and its matched-noise variant
        props = {p.key: p.value for p in fp32.metadata_props}
        props.update(precision="INT8 (static PTQ, Conv in enc_p/dp/flow)", fp32_parent_sha256=record["fp32"]["sha256"],
                     calibration_sha256=record["calibration"]["sha256"], purpose="deployment graph")
        onnx.helper.set_model_props(int8, props)
        onnx.checker.check_model(int8, full_check=True)
        onnx.save(int8, int8_path)
        export_record = json.loads((a.fp32.parent / "export_record.json").read_text(encoding="utf-8"))
        matched, surgery = C.matched_noise_graph(int8, export_record["model"]["inter_channels"])
        check("matched_noise_strips_to_deployment_without_random",
              C.strip_matched_noise(matched).SerializeToString() == C.without_random_nodes(int8).SerializeToString())
        onnx.helper.set_model_props(matched, {**props, "purpose": "validation graph with external noise; not a deployment artifact"})
        onnx.checker.check_model(matched, full_check=True)
        onnx.save(matched, matched_path)
        record["int8"] = {"file": int8_path.name, "sha256": C.sha256(int8_path), "bytes": int8_path.stat().st_size,
                          "bytes_ratio_to_fp32": int8_path.stat().st_size / record["fp32"]["bytes"]}
        record["matched_noise_graph"] = {"file": matched_path.name, "sha256": C.sha256(matched_path), **surgery}

        # 6. native-random smoke on three Harvard sentences
        samples = json.loads(a.manifest.read_text(encoding="utf-8"))
        sess = C.ort_session(int8_path)
        record["native_random_smoke"] = []
        for s in (samples[0], samples[359], samples[719]):
            ids = np.array(s["phoneme_ids"], dtype=np.int64)[None, :]
            wave = sess.run(None, {"input": ids, "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
                                   "scales": np.array(C.SCALES, dtype=np.float32)})[0]
            ok = wave.ndim == 3 and wave.shape[2] > 4410 and bool(np.isfinite(wave).all()) and float(np.std(wave)) > 1e-6
            record["native_random_smoke"].append({"idx": s["idx"], "shape": list(wave.shape), "std": float(np.std(wave)),
                                                  "passed": ok})
            check(f"native_random_idx{s['idx']}", ok)
        record.update(status="passed", finished_utc=datetime.now(timezone.utc).isoformat())
        C.write_json(record_path, record)
        print(f"PASSED {a.label}: {dict(qconv)} QLinearConv, {record['int8']['bytes'] / 1e6:.1f} MB "
              f"({record['int8']['bytes_ratio_to_fp32']:.3f} of FP32)")
        return 0
    except BaseException as error:
        record.update(status="failed", error="".join(traceback.format_exception(error)),
                      finished_utc=datetime.now(timezone.utc).isoformat())
        C.write_json(record_path, record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
