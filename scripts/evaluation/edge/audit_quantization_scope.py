"""Read-only model inspection; writes a new report, never executes a model.

Requires the existing ONNX package, plus Python's standard library. Reads the six
hash-bound deployment graphs, saved OpenVINO XML/BIN and previous execution
metadata. No PyTorch checkpoint loading, OpenVINO compilation, inference or PTQ.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SESSION = ROOT / "results/edge-n100-openvino/2026-11-01"
POLICY = ROOT / "configs/quantization/n100-common-flow-conv-int8.json"
WEIGHTED = {"Conv", "ConvTranspose", "MatMul", "Gemm"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return {"sha256": actual, "bytes": path.stat().st_size}


def module(name):
    return next((s for s in ("enc_p", "dp", "flow", "dec") if f"/{s}/" in name), "alignment_or_other")


def component(name, op, group=1):
    m = module(name)
    if m == "enc_p":
        if "/attn_layers." in name:
            return "encoder_attention_projection" if op == "Conv" else "encoder_attention_other"
        if "/ffn_layers." in name:
            return "encoder_ffn"
        if "/proj/" in name:
            return "encoder_statistics_projection"
        return "encoder_embedding_norm_or_other"
    if m == "dp":
        if op == "Conv":
            return "duration_depthwise" if group > 1 else "duration_pointwise_pre_or_projection"
        return "duration_spline_norm_or_other"
    if m == "flow":
        if "/attn/" in name:
            return "flow_attention_projection_or_ffn" if op == "Conv" else "flow_attention_other"
        if op == "Conv":
            if "/enc/" in name:
                return "flow_wavenet"
            return "flow_pre_or_post"
        return "flow_gating_or_other"
    if m == "dec":
        if op == "ConvTranspose":
            return "decoder_upsampling"
        if "/conv_pre/" in name or "/conv_post/" in name:
            return "decoder_input_output"
        if op == "Conv":
            if "/dw/" in name:
                return "decoder_ir_depthwise"
            if "/pw_expand/" in name or "/pw_project/" in name:
                return "decoder_ir_pointwise"
            return "decoder_resblock_convolution"
        return "decoder_activation_or_other"
    return "alignment_or_shape_or_other"


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=SESSION / ".local/originals/final")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error("Output already exists; use a new directory to preserve reports")
    import onnx

    policy = read(POLICY)
    manifest_path = SESSION / "evidence/models.json"
    manifest = read(manifest_path)
    manifest_check = verify(manifest_path, policy["source_manifest_sha256"])
    verify(args.bundle / "models.json", policy["source_manifest_sha256"])
    catalog_path = ROOT / "demo/n100_models.json"
    catalog = {m["id"]: m for m in read(catalog_path)["models"]}
    if {m["id"] for m in manifest["models"]} != set(policy["model_ids"]):
        raise ValueError("Model set does not match policy")
    choose = re.compile(policy["selection"]["onnx_name_fullmatch"])
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inspection_only": True,
        "new_quantization_inference_scoring_or_benchmark": False,
        "python": platform.python_version(), "onnx_reader": onnx.__version__,
        "script_sha256": sha(Path(__file__)), "policy_sha256": sha(POLICY),
        "manifest": manifest_check, "catalog_sha256": sha(catalog_path),
        "models": [],
    }
    rows, operations, runtime_rows, selections = [], [], [], []
    recheck = []
    for item in manifest["models"]:
        mid = item["id"]
        cat = catalog[mid]
        source = args.bundle / item["fp32"]["file"]
        checked = verify(source, item["fp32"]["sha256"])
        config = verify(args.bundle / item["config"]["file"], item["config"]["sha256"])
        prep_path = args.bundle / cat["preparation"]["file"]
        prep_check = verify(prep_path, cat["preparation"]["sha256"])
        prep = read(prep_path)
        if prep["status"] != "prepared" or prep["model"]["fp32"]["sha256"] != checked["sha256"]:
            raise ValueError(f"Inconsistent previous preparation: {mid}")
        files = {}
        for prec in ("fp32", "int8"):
            for spec in cat["artifacts"][prec]:
                path = args.bundle / spec["file"]
                files[f"{prec}.{path.suffix[1:]}"] = verify(path, spec["sha256"])
                recheck.append((path, spec["sha256"]))
        model = onnx.load(str(source), load_external_data=False)
        if any(t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer):
            raise ValueError("External tensors need explicit hash-bound inspection")
        onnx.checker.check_model(model)
        if any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
               for n in model.graph.node for a in n.attribute):
            raise ValueError("Subgraphs need recursive inspection")
        if any(n.op_type in {"ReduceL2", "QuantizeLinear", "DequantizeLinear", "QLinearConv"}
               for n in model.graph.node):
            raise ValueError("Unexpected weight-norm/quantized op in FP32 source")
        initializers = {t.name: t for t in model.graph.initializer}
        if any(t.data_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.INT64)
               for t in initializers.values()):
            raise ValueError("Unexpected initializer precision")
        ov = ET.parse(prep_path.parent / "fp32.xml")
        ov_layers = {n.get("name"): n for n in ov.findall("./layers/layer")}
        old_allowed = set(prep["quantization"]["allowed_convolutions"])
        selected, mod_ops, components = [], Counter(), Counter()
        for n in model.graph.node:
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
            group = int(attrs.get("group", 1))
            comp = component(n.name, n.op_type, group)
            mod_ops[(module(n.name), n.op_type)] += 1
            components[(comp, n.op_type)] += 1
            if n.op_type not in WEIGHTED:
                continue
            weight = initializers.get(n.input[1]) if len(n.input) > 1 else None
            if n.op_type in ("Conv", "ConvTranspose") and weight is None:
                raise ValueError(f"Convolution weights are not direct constants: {mid} {n.name}")
            picked = n.op_type == "Conv" and bool(choose.fullmatch(n.name))
            ov_name = n.name + "/WithoutBiases" if n.name + "/WithoutBiases" in ov_layers else n.name
            # OpenVINO names deconvolution as ConvolutionBackpropData_N;
            # the original /dec/ups.N/ConvTranspose name belongs to its bias Add.
            ov_type = ov_layers[ov_name].get("type") if ov_name in ov_layers else "not_name_mapped"
            if picked:
                if weight is None or weight.data_type != onnx.TensorProto.FLOAT or ov_type != "Convolution":
                    raise ValueError(f"Candidate cannot be resolved: {mid} {n.name}")
                if ov_name not in old_allowed:
                    raise ValueError("Candidate has no previous in-scope evidence")
                selected.append({"onnx_name": n.name, "openvino_name": ov_name,
                                 "weight_shape": list(weight.dims), "weight_elements": math.prod(weight.dims)})
            rows.append({
                "model_id": mid, "module": module(n.name), "component": comp,
                "onnx_name": n.name, "op_type": n.op_type, "group": group,
                "weight_initializer": weight.name if weight is not None else "",
                "weight_shape": json.dumps(list(weight.dims)) if weight is not None else "dynamic_or_derived",
                "weight_elements": math.prod(weight.dims) if weight is not None else 0,
                "openvino_name": ov_name, "openvino_named_node_type": ov_type,
                "previous_allowed_conv": ov_name in old_allowed, "candidate_int8": picked,
                "status": "candidate_requires_validation" if picked else "keep_original_precision",
            })
        selected.sort(key=lambda x: x["onnx_name"])
        if len(selected) != policy["selection"]["expected_count_per_model"]:
            raise ValueError(f"Wrong candidate count: {mid}")
        selections.append([(x["onnx_name"], x["weight_shape"]) for x in selected])
        for (mod, op), count in sorted(mod_ops.items()):
            operations.append({"model_id": mid, "module": mod, "onnx_op_type": op, "count": count})
        nodes_path = prep_path.parent / "int8_runtime_nodes.json"
        # Compare local metadata against the preserved Git evidence, then parse it.
        archived_nodes = SESSION / "evidence" / Path(cat["preparation"]["file"]).parent / nodes_path.name
        node_check = verify(nodes_path, sha(archived_nodes))
        nodes = read(nodes_path)
        kernels = [n for n in nodes if n.get("layerType") in ("Convolution", "GroupConvolution", "FullyConnected")
                   and any(s in (n.get("runtimePrecision", "") + " " + n.get("execType", "")).lower()
                           for s in ("i8", "u8", "int8"))]
        if len(kernels) != prep["int8_weighted_kernel_count"]:
            raise ValueError("Stored kernel count does not match parsed runtime metadata")
        selected_runtime = []
        for s in selected:
            matches = [n for n in kernels if s["openvino_name"] in n.get("originalLayersNames", "").split(",")
                       or s["openvino_name"] == n["name"]]
            if len(matches) != 1:
                raise ValueError(f"Expected one previous INT8 execution kernel: {mid} {s['onnx_name']}")
            selected_runtime.append(matches[0]["name"])
        for n in nodes:
            if n.get("layerType") in ("Convolution", "GroupConvolution", "FullyConnected", "Deconvolution", "MatMul"):
                runtime_rows.append({"model_id": mid, **{k: n.get(k, "") for k in
                                    ("name", "layerType", "runtimePrecision", "execType", "originalLayersNames")},
                                     "previous_kernel_for_candidate": n["name"] in selected_runtime})
        report["models"].append({
            "id": mid, "label": item["label"], "checkpoint": item["checkpoint"]["filename"],
            "fp32_onnx": checked, "config": config, "preparation": prep_check,
            "artifacts": files, "previous_runtime_metadata": node_check,
            "onnx_checker": "passed", "onnx_node_count": len(model.graph.node),
            "op_counts": dict(sorted(Counter(n.op_type for n in model.graph.node).items())),
            "component_counts": [{"component": c, "op": o, "count": v} for (c, o), v in sorted(components.items())],
            "initializer_fp32_elements": sum(math.prod(t.dims) for t in initializers.values() if t.data_type == 1),
            "previous_int8_weighted_kernels": len(kernels),
            "previous_int8_kernels_by_module": dict(Counter(module(n['name']+' '+n.get('originalLayersNames','')) for n in kernels)),
            "previous_allowed_convs": len(old_allowed),
            "previous_quality_status": prep["quality_status"],
            "candidate": {"count": len(selected), "nodes": selected,
                          "weight_elements": sum(x["weight_elements"] for x in selected),
                          "all_had_previous_int8_kernel": True,
                          "quality_and_speed_of_new_scope": "not_measured"},
        })
        recheck.extend([(source, checked["sha256"]), (prep_path, prep_check["sha256"]),
                        (args.bundle / item["config"]["file"], config["sha256"]),
                        (nodes_path, node_check["sha256"])])
    if not all(s == selections[0] for s in selections):
        raise ValueError("Selected names or weight shapes differ across models")
    for path, expected in recheck:
        verify(path, expected)
    report["checks"] = {"six_models_verified": True, "common_names_and_weight_shapes": True,
                        "source_hashes_unchanged_after_inspection": True,
                        "source_files_rehashed": len(recheck),
                        "new_candidate_quality_pass": None, "new_candidate_speed_pass": None}
    args.out.mkdir(parents=True)
    (args.out / "inventory.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    write_csv(args.out / "weighted-operations.csv", rows)
    write_csv(args.out / "operation-counts.csv", operations)
    write_csv(args.out / "previous-runtime-weighted-operations.csv", runtime_rows)
    (args.out / "SHA256SUMS").write_text("".join(f"{sha(f)}  {f.name}\n" for f in sorted(args.out.iterdir()) if f.is_file()), encoding="utf-8")
    print(json.dumps({"inspection": "passed", "models": 6, "weighted_operations": len(rows),
                      "candidate_convs_per_model": 40, "candidate_quality": "not_measured",
                      "candidate_speed": "not_measured", "output": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
