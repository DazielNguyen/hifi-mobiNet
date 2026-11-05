"""Static inspection only: parse existing ONNX/JSON; write report evidence.

No checkpoint loading, model construction, conversion, compilation, inference,
calibration or quantization. Requires an already installed ONNX reader.
Outputs use exclusive creation so earlier evidence cannot be overwritten.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import onnx


ROOT = Path(__file__).resolve().parents[3]
SESSION = ROOT / "results/edge-n100-openvino/2026-11-01"
BUNDLE = SESSION / ".local/originals/final"
IDS = ("piper-no-vits2-cpn", "piper-parir-no-vits2-cpn", "piper-seqir-no-vits2-cpn")
SOURCE_FILES = [
    *(f"configs/training/{mid}.yaml" for mid in IDS),
    "src/hifimobinet/training/registry.py", "src/hifimobinet/training/vendor.py",
    "src/hifimobinet/training/config.py", "src/hifimobinet/training/train.py",
    "src/hifimobinet/training/piper_module.py", "src/hifimobinet/training/piper_ir_module.py",
    "vendor/edgetts/vits/lightning.py", "vendor/edgetts/vits/models.py",
    "vendor/edgetts/vits/modules.py", "vendor/edgetts/vits/attentions.py",
    "vendor/edgetts/vits/transforms.py", "vendor/edgetts/vits/commons.py",
    "src/hifimobinet/architecture/vits/modules/generator.py",
    "src/hifimobinet/architecture/vits/utils/resblocks.py",
    "src/hifimobinet/architecture/vits/utils/normalization.py",
    "src/hifimobinet/architecture/vits/modules/discriminators.py",
    "scripts/evaluation/onnx/common.py", "scripts/evaluation/onnx/export.py",
    "demo/n100_models.json", "results/edge-n100-openvino/2026-11-01/evidence/models.json",
    "results/edge-n100-openvino/2026-11-01/evidence/openvino_suite.py",
    "docs/plan-onnx-ptq-v2.vi.md",
]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def module(name):
    return next((m for m in ("enc_p", "dp", "flow", "dec") if f"/{m}/" in name), "alignment_or_other")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Source checkout git HEAD, supplied by caller")
    args = parser.parse_args()
    out = Path(__file__).resolve().parent / "evidence"
    out.mkdir(exist_ok=True)
    watched = {}

    def record(path, expected=None):
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Hash mismatch: {path}")
        watched[path] = actual
        return {"path": path.relative_to(ROOT).as_posix(), "sha256": actual, "bytes": path.stat().st_size}

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "git_head": args.revision,
        "inspection_only": True, "new_model_execution_or_conversion": False,
        "python": platform.python_version(), "onnx_reader": onnx.__version__,
        "sources": [record(ROOT / f) for f in SOURCE_FILES],
        "inspection_script": record(Path(__file__).resolve()), "models": [],
    }
    manifest = read(SESSION / "evidence/models.json")
    catalog = {m["id"]: m for m in read(ROOT / "demo/n100_models.json")["models"]}
    weighted, constants, operations, historical = [], [], [], []
    for item in manifest["models"]:
        mid = item["id"]
        if mid not in IDS:
            continue
        cat = catalog[mid]
        source = BUNDLE / item["fp32"]["file"]
        model_report = {"id": mid, "checkpoint_provenance_only": item["checkpoint"],
                        "graph": record(source, item["fp32"]["sha256"]),
                        "export_config": record(BUNDLE / item["config"]["file"], item["config"]["sha256"])}
        model_report["current_config_matches_export_config"] = digest(ROOT / f"configs/training/{mid}.yaml") == item["config"]["sha256"]
        graph = onnx.load(str(source), load_external_data=False)
        if any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):
            raise ValueError("External initializers require separate inspection")
        if any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS) for n in graph.graph.node for a in n.attribute):
            raise ValueError("Nested graphs require recursive inspection")
        onnx.checker.check_model(graph)
        initializers = {t.name: t for t in graph.graph.initializer}
        consumers = defaultdict(list)
        counts = Counter(n.op_type for n in graph.graph.node)
        by_module = Counter((module(n.name), n.op_type) for n in graph.graph.node)
        model_report.update({"onnx_checker": "passed_static_only", "onnx_nodes": len(graph.graph.node),
                             "op_counts": dict(sorted(counts.items())),
                             "graph_inputs": [x.name for x in graph.graph.input],
                             "quantized_or_weight_norm_ops": {k: counts[k] for k in ("ReduceL2", "QuantizeLinear", "DequantizeLinear", "QLinearConv")}})
        for n in graph.graph.node:
            for inp in n.input:
                consumers[inp].append(n.name)
            if n.op_type not in ("Conv", "ConvTranspose", "MatMul", "Gemm"):
                continue
            weight = initializers.get(n.input[1])
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
            group = int(attrs.get("group", 1))
            if n.op_type in ("Conv", "ConvTranspose"):
                if weight is None or weight.data_type != onnx.TensorProto.FLOAT:
                    raise ValueError(f"Convolution lacks direct FP32 weights: {mid}: {n.name}")
                eligibility = "W8A8_candidate_requires_validation"
                if n.op_type == "ConvTranspose" or group > 1:
                    eligibility = "W8A8_candidate_backend_pattern_confirmation_required"
            else:
                eligibility = "activation_or_derived_operand_quantization_conditional"
                if module(n.name) == "alignment_or_other":
                    eligibility = "alignment_matmul_possible_in_principle_low_priority_unverified"
            weighted.append({"model_id": mid, "module": module(n.name), "node": n.name,
                             "op": n.op_type, "group": group,
                             "kernel": json.dumps(attrs.get("kernel_shape", [])),
                             "dilations": json.dumps(attrs.get("dilations", [])),
                             "weight_initializer": weight.name if weight is not None else "",
                             "weight_shape": json.dumps(list(weight.dims)) if weight is not None else "dynamic_or_derived",
                             "weight_dtype": onnx.TensorProto.DataType.Name(weight.data_type) if weight is not None else "not_direct_initializer",
                             "weight_elements": math.prod(weight.dims) if weight is not None else 0,
                             "eligibility": eligibility})
        for t in graph.graph.initializer:
            constants.append({"model_id": mid, "initializer": t.name, "shape": json.dumps(list(t.dims)),
                              "dtype": onnx.TensorProto.DataType.Name(t.data_type), "elements": math.prod(t.dims),
                              "direct_consumers": json.dumps(consumers[t.name])})
        for (mod, op), count in sorted(by_module.items()):
            operations.append({"model_id": mid, "module": mod, "op": op, "count": count})
        prep_path = SESSION / "evidence" / cat["preparation"]["file"]
        model_report["historical_preparation"] = record(prep_path, cat["preparation"]["sha256"])
        prep = read(prep_path)
        if prep["model"]["fp32"]["sha256"] != item["fp32"]["sha256"]:
            raise ValueError("Historical preparation references a different source graph")
        runtime_path = prep_path.parent / "int8_runtime_nodes.json"
        model_report["historical_runtime_metadata"] = record(runtime_path)
        runtime_counts = Counter()
        for node in read(runtime_path):
            if node.get("layerType") not in ("Convolution", "GroupConvolution", "FullyConnected", "Deconvolution", "MatMul"):
                continue
            precision = node.get("runtimePrecision", "")
            is_int8 = any(s in (precision + " " + node.get("execType", "")).lower() for s in ("i8", "u8", "int8"))
            mod = module(node["name"])
            runtime_counts[(mod, "int8" if is_int8 else precision)] += 1
            historical.append({"model_id": mid, "historical_only": True, "module": mod,
                               **{key: node.get(key, "") for key in ("name", "layerType", "runtimePrecision", "execType", "originalLayersNames")}})
        model_report["historical_weighted_kernels_by_module_precision"] = {f"{m}/{p}": n for (m,p),n in sorted(runtime_counts.items())}
        report["models"].append(model_report)
    if len(report["models"]) != 3 or len(weighted) != 483:
        raise ValueError("Unexpected graph inventory; investigate before publishing report")
    for path, before in watched.items():
        if digest(path) != before:
            raise ValueError(f"Source changed during inspection: {path}")
    report["read_sources_unchanged_after_inspection"] = True
    report["weighted_operator_rows"] = len(weighted)
    report["initializer_rows"] = len(constants)
    write_csv(out / "weighted-operators.csv", weighted)
    write_csv(out / "initializers.csv", constants)
    write_csv(out / "operation-counts.csv", operations)
    write_csv(out / "historical-runtime-kernels.csv", historical)
    with (out / "inspection.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"models": len(report["models"]), "weighted_operators": len(weighted),
                      "initializers": len(constants), "model_execution": False}))


if __name__ == "__main__":
    main()
