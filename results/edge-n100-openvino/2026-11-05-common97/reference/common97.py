"""Versioned Piper PTQ policy; protects duration/alignment by construction.

Only the three hash-pinned Piper graphs in the report qualify. This module is
also copied into the standalone N100 kit; heavy dependencies are lazy imports.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

RECIPE = "piper-common97-v1"
MODEL_IDS = ("piper-no-vits2-cpn", "piper-parir-no-vits2-cpn", "piper-seqir-no-vits2-cpn")
SCOPES = ("/enc_p/", "/dp/", "/flow/")
EXPECTED = {"/enc_p/": 37, "/dp/": 20, "/flow/": 40}


def policy_path():
    portable = Path(__file__).with_name("common97-policy.json")
    return portable if portable.is_file() else Path(__file__).resolve().parents[2] / "configs/quantization/piper-common97.json"


def policy():
    return json.loads(policy_path().read_text(encoding="utf-8"))


def identity():
    return {"recipe": RECIPE, "policy_sha256": hashlib.sha256(policy_path().read_bytes()).hexdigest(),
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def check_source(model_id, source_hash):
    item = policy()["models"].get(model_id)
    if item is None or item["fp32_onnx_sha256"] != source_hash:
        raise ValueError("common97 requires one of the three verified Piper FP32 ONNX sources")


def scope(model, model_id):
    expected = set(policy()["models"][model_id]["allowed_node_names"])
    ops = list(model.get_ops())
    names = [n.get_friendly_name() for n in ops]
    if len(names) != len(set(names)):
        raise ValueError("duplicate graph node names: cannot apply an exact ignored scope")
    if any(n.get_type_name() == "FakeQuantize" for n in ops):
        raise ValueError("PTQ input must be FP32, not a previously quantized graph")
    allowed = {n.get_friendly_name() for n in ops if n.get_type_name() == "Convolution"
               and n.get_friendly_name().startswith(SCOPES)}
    counts = {p: sum(n.startswith(p) for n in allowed) for p in SCOPES}
    if allowed != expected or counts != EXPECTED:
        raise ValueError(f"common97 graph scope drift: counts={counts}; missing={sorted(expected-allowed)}; extra={sorted(allowed-expected)}")
    ignored = sorted(n.get_friendly_name() for n in ops
                     if n.get_type_name() not in ("Constant", "Parameter", "Result")
                     and n.get_friendly_name() not in allowed)
    return {**identity(), "allowed_convolutions": sorted(allowed), "ignored_nodes": ignored,
            "breakdown": counts, "subset_size": 60, "preset": "MIXED", "target_device": "CPU",
            "fast_bias_correction": True, "compress_to_fp16": False}


def audit_fq_edges(model, allowed, expected_count=96):
    """Inspect destinations, not just FQ names/counts or executed INT8 kernels.

    Input FQ may follow an encoder Pad/Reshape. Every consumer must be the
    activation input of an explicitly selected Conv; mask/CumSum paths cannot
    acquire FQ even if a quantizer is renamed.
    """
    import numpy as np
    rows = []
    for node in model.get_ops():
        if node.get_type_name() != "FakeQuantize":
            continue
        targets = list(node.output(0).get_target_inputs())
        if not targets or any(t.get_node().get_type_name() != "Convolution"
                              or t.get_node().get_friendly_name() not in allowed
                              or t.get_index() != 0 for t in targets):
            raise ValueError(f"FQ outside selected Conv activation edges: {node.get_friendly_name()}")
        ranges = []
        for i in range(1, 5):
            constant = node.input_value(i).get_node()
            if constant.get_type_name() != "Constant":
                raise ValueError("FQ range is not a constant")
            value = np.asarray(constant.get_data())
            if not np.isfinite(value).all():
                raise ValueError("nonfinite FQ range")
            ranges.append(value)
        if node.get_levels() != 256 or np.any(ranges[0] >= ranges[1]) or np.any(ranges[2] >= ranges[3]):
            raise ValueError("invalid common97 FQ levels/ranges")
        rows.append({"name": node.get_friendly_name(),
                     "source": node.input_value(0).get_node().get_friendly_name(),
                     "consumers": sorted(t.get_node().get_friendly_name() for t in targets)})
    if len(rows) != expected_count:
        raise ValueError(f"FQ inventory drift: {len(rows)} != {expected_count}")
    return {"passed": True, "fake_quantize_count": len(rows), "edges": sorted(rows, key=lambda r: r["name"])}


def audit_graph(model, model_id):
    if model_id not in MODEL_IDS:
        raise ValueError("common97 is only validated for the three Piper models")
    if {p.get_any_name() for p in model.inputs} != {"input", "input_lengths", "scales"} or len(model.outputs) != 1:
        raise ValueError("deployment requires the native three-input / one-output graph")
    allowed = set(policy()["models"][model_id]["allowed_node_names"])
    return audit_fq_edges(model, allowed)


def audit_runtime(nodes, model_id):
    expected = set(policy()["models"][model_id]["allowed_node_names"])
    kernels = [n for n in nodes if n.get("layerType") in
               ("Convolution", "GroupConvolution", "FullyConnected", "Deconvolution")
               and any(t in (n.get("runtimePrecision", "") + " " + n.get("execType", "")).lower()
                       for t in ("i8", "u8", "int8"))]
    seen = set()
    for node in kernels:
        names = set(node.get("originalLayersNames", "").split(",")) | {node["name"]}
        matched = names & expected
        if node.get("layerType") != "Convolution" or not matched or any("/dec/" in n for n in names):
            raise ValueError("INT8 runtime kernel outside common97")
        seen.update(matched)
    if len(kernels) != 97 or seen != expected:
        raise ValueError(f"common97 runtime drift: {len(kernels)} kernels; {len(seen)} identified Conv nodes")
    return {"passed": True, "int8_weighted_kernel_count": len(kernels)}


def validate_alignment(model, core, config, rows, make_feeds, output, sample_rate):
    """Probe a clone, leaving deployment I/O intact. Native RNG, not paired WER."""
    import numpy as np
    import wave
    debug = model.clone()
    byname = {n.get_friendly_name(): n for n in debug.get_ops()}
    if byname.get("/CumSum") is None or byname["/CumSum"].get_type_name() != "CumSum" or "/Sub" not in byname:
        raise ValueError("cannot identify the verified duration/alignment outputs")
    debug.add_outputs([byname["/CumSum"].output(0), byname["/Sub"].output(0)])
    request = core.compile_model(debug, "CPU", config).create_infer_request()
    records = []
    for row in rows:
        for length_scale in (1.0, 1.3):
            inputs = make_feeds(row)
            inputs["scales"] = inputs["scales"].copy()
            inputs["scales"][1] = length_scale
            request.infer(inputs)
            audio = np.array(request.get_output_tensor(0).data, copy=True)
            cumulative = np.array(request.get_output_tensor(1).data, copy=True)
            alignment = np.array(request.get_output_tensor(2).data, copy=True)
            coverage = alignment.sum(axis=-2).reshape(-1)
            if not coverage.size or not np.isfinite(cumulative).all() or not np.all(coverage == 1):
                raise ValueError(f"lost alignment: {row['id']} length_scale={length_scale}")
            if audio.ndim != 3 or audio.shape[:2] != (1, 1) or not np.isfinite(audio).all() or np.std(audio, dtype=np.float64) <= 1e-6:
                raise ValueError("invalid alignment-check waveform")
            if audio.shape[-1] != coverage.size * 256 or cumulative.max() != coverage.size:
                raise ValueError("audio/duration/alignment length disagreement")
            name = f"alignment-{row['id']}-length{length_scale:.1f}.wav"
            pcm = np.clip(audio.reshape(-1) * (32767 / max(.01, float(np.abs(audio).max()))), -32767, 32767).astype("<i2")
            with wave.open(str(output / name), "wb") as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(sample_rate); f.writeframes(pcm.tobytes())
            records.append({"case": row["id"], "length_scale": length_scale, "frames": int(coverage.size),
                            "min_coverage": float(coverage.min()), "max_coverage": float(coverage.max()),
                            "zero_alignment_frames": int(np.sum(coverage == 0)),
                            "audio_seconds": audio.shape[-1] / sample_rate, "wav": name,
                            "wav_sha256": hashlib.sha256((output / name).read_bytes()).hexdigest()})
            (output / "alignment-validation.json").write_text(json.dumps({"status": "running", "cases": records}, indent=2) + "\n")
    if not records or max(r["audio_seconds"] for r in records) <= 10:
        raise ValueError("regression check did not exercise audio longer than 10 seconds")
    result = {"status": "passed", "cases": records, "rng": "native, not matched; length scales 1.0 and 1.3",
              "quality_status": "alignment regression only; not WER/MOS certification"}
    (output / "alignment-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
