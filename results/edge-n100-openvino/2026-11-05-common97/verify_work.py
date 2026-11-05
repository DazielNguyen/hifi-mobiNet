"""Audit the received N100 work directory without loading or running a model.

Requires the external work directory. --write only writes a derived report here.
All source files remain read-only; hashes bind metadata to the measured session.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
from pathlib import Path
import struct
import wave
import xml.etree.ElementTree as ET

from verify import ROOT, MODELS, close, read, require, runtime_audit, safe, sha


def inspect_ir(xml_path, allowed, expected_edges=None):
    tree = ET.parse(xml_path).getroot()
    layers = {n.attrib["id"]: n for n in tree.find("layers")}
    edges = list(tree.find("edges"))
    incoming, outgoing = {}, collections.defaultdict(list)
    for edge in edges:
        a = edge.attrib
        key = (a["to-layer"], int(a["to-port"]))
        require(key not in incoming, "duplicate IR input edge")
        incoming[key] = layers[a["from-layer"]]
        outgoing[a["from-layer"]].append((layers[a["to-layer"]], int(a["to-port"])))
    require({n.attrib["name"] for n in layers.values() if n.attrib["type"] == "Parameter"} ==
            {"input", "input_lengths", "scales"}, "deployment IR input signature")
    require(sum(n.attrib["type"] == "Result" for n in layers.values()) == 1, "IR output count")
    convs = {n.attrib["name"] for n in layers.values() if n.attrib["type"] == "Convolution"}
    require(allowed <= convs, "selected Convolution missing from IR")
    binary = xml_path.with_suffix(".bin").read_bytes()
    for node in layers.values():
        if node.attrib["type"] == "Const":
            data = node.find("data").attrib
            offset, size = int(data["offset"]), int(data["size"])
            require(offset >= 0 and size >= 0 and offset + size <= len(binary), "Const exceeds BIN bounds")
    rows = []
    for node_id, node in layers.items():
        if node.attrib["type"] != "FakeQuantize":
            continue
        require(node.find("data").attrib["levels"] == "256", "FQ levels")
        consumers = outgoing[node_id]
        require(consumers and all(n.attrib["type"] == "Convolution" and n.attrib["name"] in allowed
                                 and port == 0 for n, port in consumers), "FQ leaks outside selected Conv activation edges")
        ranges = []
        for port in range(1, 5):
            parent = incoming[node_id, port]
            require(parent.attrib["type"] == "Const", "nonconstant FQ range")
            data = parent.find("data").attrib
            require(data["element_type"] == "f32" and int(data["size"]) == 4 and
                    data["shape"] in ("", "1"), "unexpected FQ range dtype/shape; manual review required")
            value = struct.unpack_from("<f", binary, int(data["offset"]))[0]
            require(math.isfinite(value), "nonfinite FQ range")
            ranges.append(value)
        require(ranges[0] < ranges[1] and ranges[2] < ranges[3], "degenerate FQ range")
        rows.append({"name": node.attrib["name"], "source": incoming[node_id, 0].attrib["name"],
                     "consumers": sorted(n.attrib["name"] for n, _ in consumers)})
    rows.sort(key=lambda r: r["name"])
    require(len(rows) == (96 if expected_edges is not None else 0), "FQ count")
    if expected_edges is not None:
        require(rows == expected_edges, "actual FQ edges differ from preparation audit")
    return {"fake_quantize_count": len(rows), "range_constants_checked": 4 * len(rows),
            "all_ranges_finite_and_ordered": True, "fq_only_on_selected_conv_activation_inputs": True,
            "constant_bin_bounds_checked": True}


def inspect_wav(path, samples, seconds):
    with wave.open(str(path), "rb") as stream:
        require((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getcomptype()) ==
                (1, 2, 22050, "NONE"), "unexpected WAV format: " + path.name)
        count = stream.getnframes()
        pcm = stream.readframes(count)
    require(count == samples and len(pcm) == count * 2 and close(count / 22050, seconds),
            "WAV sample count/duration differs: " + path.name)
    require(any(pcm), "silent/empty WAV: " + path.name)
    return {"file": path.name, "samples": count, "audio_seconds": count / 22050,
            "sha256": sha(path), "format": "mono PCM16 22050 Hz"}


def audit(root, source):
    inventory = read(root / "work-manifest.json")
    entries = inventory["files"]
    require(len(entries) == 84 and len({e["path"] for e in entries}) == 84, "work manifest count")
    require(source.is_dir(), "external work directory is missing")
    actual = {p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()}
    require(actual == {e["path"] for e in entries}, "missing/unexpected work source file")
    for entry in entries:
        original = safe(source, entry["path"])
        require(original.stat().st_size == entry["size_bytes"] and sha(original) == entry["sha256"],
                "work source hash/size differs: " + entry["path"])
        if entry["repository_copy"]:
            require(safe(root, entry["repository_copy"]).read_bytes() == original.read_bytes(),
                    "imported metadata differs: " + entry["path"])
    require(sum(e["size_bytes"] for e in entries) == inventory["total_bytes"], "work total bytes")
    refs = root / "reference"
    policy, models = read(refs / "piper-common97.json"), read(refs / "models.json")
    for key in ("harvard", "split", "calibration", "calibration_dataset_rows"):
        require(sha(safe(refs, models[key]["file"])) == models[key]["sha256"], "input metadata hash")
    calibration = [json.loads(line) for line in (refs / models["calibration"]["file"]).read_text(encoding="utf-8").splitlines()]
    dataset = [json.loads(line) for line in (refs / models["calibration_dataset_rows"]["file"]).read_text(encoding="utf-8").splitlines()]
    dataset_by_path = {r["audio_norm_path"]: r for r in dataset}
    splits = read(refs / models["split"]["file"])
    require(len(calibration) == len({r["id"] for r in calibration}) == 60, "calibration IDs/count")
    for row in calibration:
        path = row["audio_norm_path"]
        require([k for k, paths in splits.items() if path in paths] == ["train"], "calibration split leak")
        require(row["phoneme_ids"] == dataset_by_path[path]["phoneme_ids"], "calibration tokens differ")
    report_models = []
    for model in MODELS:
        directory = source / model
        prep = read(directory / "preparation.json")
        prep_hash = sha(directory / "preparation.json")
        cells = [read(p) for p in (root / "evidence/results").glob(f"r*_{model}_*_t4/result.json")]
        require(len(cells) == 6 and all(r["preparation_sha256"] == prep_hash for r in cells), "benchmark preparation binding")
        require(prep["status"] == "prepared", "preparation not complete")
        require(prep["model"] == next(m for m in models["models"] if m["id"] == model), "model/source identity")
        require(prep["policy"] == {"recipe": "piper-common97-v1", "policy_sha256": sha(refs / "piper-common97.json"),
                                   "implementation_sha256": sha(refs / "common97.py")}, "policy/implementation binding")
        for field, filename in (("numeric_guard_sha256", "ptq_numeric_guard.py"),
                                ("regression_sha256", "piper-common97-regression.json"),
                                ("script_sha256", "openvino_suite.py"), ("manifest_sha256", "models.json")):
            require(prep[field] == sha(refs / filename), "preparation source binding: " + field)
        require(prep["calibration"] == {"n": 60, "train": 60, "val": 0, "test": 0, "problems": [], "passed": True},
                "calibration recorded result differs")
        allowed = set(policy["models"][model]["allowed_node_names"])
        quant = prep["quantization"]
        require(len(quant["allowed_convolutions"]) == len(allowed) == 97 and
                set(quant["allowed_convolutions"]) == allowed and quant["breakdown"] == {"/enc_p/":37, "/dp/":20, "/flow/":40},
                "quantization selection differs")
        require(all(quant[k] == v for k,v in prep["policy"].items()) and quant["subset_size"] == 60 and
                quant["compress_to_fp16"] is False, "quantization policy/settings")
        for key in ("platform", "python", "cpu_names", "logical_cpus", "packages"):
            require(all(prep["environment"][key] == c["environment_start"][key] for c in cells), "preparation environment differs")
        started, finished = [dt.datetime.fromisoformat(prep[k]) for k in ("started_utc", "finished_utc")]
        require(started <= finished < min(dt.datetime.fromisoformat(c["environment_start"]["utc"]) for c in cells),
                "preparation/benchmark chronology")
        parity = read(directory / "fp32_openvino_parity.json")
        require(parity == prep["fp32_parity"] and [r["idx"] for r in parity] ==
                [0,1,2,59,119,179,239,299,359,419,479,599,719], "FP32 parity record/IDs")
        for row in parity:
            require(row["same_shape"] is True and row["finite"] is True and row["passed"] is True and
                    all(math.isfinite(row[k]) and row[k] >= 0 for k in ("rmse","nrmse","reference_rms","max_abs_error")) and
                    close(row["nrmse"], row["rmse"] / max(row["reference_rms"], 1e-12)) and
                    row["nrmse"] <= 1e-3 and row["max_abs_error"] <= 5e-3, "FP32 parity recorded thresholds/formula")
        graphs, runtime, wavs = {}, {}, []
        for precision in ("fp32", "int8"):
            files = prep["artifacts"][precision]["files"]
            require({f["file"] for f in files} == {precision + ".xml", precision + ".bin"} and len(files) == 2, "IR inventory")
            for f in files:
                require(sha(safe(directory, f["file"])) == f["sha256"], "actual IR hash differs from preparation")
            graphs[precision] = inspect_ir(directory / (precision + ".xml"), allowed,
                                          prep["graph_audit"]["edges"] if precision == "int8" else None)
            runtime[precision] = runtime_audit(read(directory / (precision + "_runtime_nodes.json")), precision, allowed)
            smoke = prep["artifacts"][precision]["smoke"]
            require([r["idx"] for r in smoke] == [0,359,719], "smoke IDs")
            for row in smoke:
                wavs.append(inspect_wav(directory / f"{precision}_{row['idx']:04d}.wav", row["samples"], row["audio_seconds"]))
        require(prep["graph_audit"]["passed"] is True and prep["graph_audit"]["fake_quantize_count"] == 96 and
                prep["runtime_audit"] == {"passed": True, "int8_weighted_kernel_count": 97} and
                prep["int8_weighted_kernel_count"] == 97, "recorded graph/runtime audit")
        alignment = read(directory / "alignment-validation.json")
        require(alignment == prep["alignment_validation"] and alignment["status"] == "passed", "alignment record differs")
        cases = alignment["cases"]
        require(len(cases) == 12 and {(r["case"], r["length_scale"]) for r in cases} ==
                {(case, scale) for case in ("reported","first_sentence","second_sentence","long_a","long_b","stress_300")
                 for scale in (1.,1.3)}, "alignment cases")
        for row in cases:
            require(row["min_coverage"] == row["max_coverage"] == 1 and row["zero_alignment_frames"] == 0,
                    "alignment reported uncovered frames")
            path = safe(directory, row["wav"])
            require(sha(path) == row["wav_sha256"], "alignment WAV hash differs from N100 record")
            wavs.append(inspect_wav(path, row["frames"] * 256, row["audio_seconds"]))
        require(len({r["file"] for r in wavs}) == 18 and
                {r["file"] for r in wavs} == {p.name for p in directory.glob("*.wav")}, "WAV inventory")
        report_models.append({"model": model, "preparation_sha256": prep_hash, "benchmark_bindings_checked": 6,
                              "ir_files_hash_checked": 4, "graphs": graphs, "runtime_record_audit": runtime,
                              "recorded_fp32_parity_cases_consistent": len(parity),
                              "recorded_alignment_cases_consistent": len(cases),
                              "alignment_wav_original_hashes_matched": 12,
                              "alignment_wavs_over_10_seconds": sum(r["audio_seconds"] > 10 for r in cases),
                              "longest_alignment_wav_seconds": max(r["audio_seconds"] for r in cases), "wavs": wavs})
    return {"status": "received_work_artifacts_verified_without_inference", "source_directory": inventory["source_directory"],
            "work_manifest_sha256": sha(root / "work-manifest.json"),
            "files_hashed": len(entries), "bytes_hashed": inventory["total_bytes"], "metadata_copies_checked": 15,
            "ir_files_bound_to_benchmark": 12, "calibration_train_only_rows_checked": 60,
            "wav_files_checked": 54, "alignment_wavs_bound_to_preparation": 36,
            "alignment_wavs_over_10_seconds": sum(m["alignment_wavs_over_10_seconds"] for m in report_models),
            "models": report_models, "inference_performed": False,
            "limitations": ["Coverage and FP32 parity records are internally consistent; their internal tensors/paired raw outputs were not saved, so the numerical inference checks were not rerun.",
                            "WAV hashes, headers, durations and complete PCM payload were checked; intelligibility, WER/CER/UTMOS/MOS and perceptual tail quality were not assessed.",
                            "18 smoke WAVs and 3 matched-validation ONNX files have fresh inventory hashes but no original N100 file hash in preparation.json. Smoke dimensions match recorded outputs.",
                            "The original deployment ONNX/checkpoints are not in this work directory. The auxiliary matched-validation ONNX is not the original deployment model. The separately received demo catalog is checked by verify.py.",
                            "69 non-JSON source files remain external in Downloads; rerunning this audit requires that directory."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    try:
        report = audit(ROOT, args.source)
        target = ROOT / "verification/work-verification.json"
        if args.write:
            target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        else:
            require(read(target) == report, "derived work report differs")
        print(json.dumps({k: v for k, v in report.items() if k not in ("models", "limitations")}, indent=2))
    except (OSError, ValueError, KeyError, TypeError, StopIteration, ET.ParseError, wave.Error, struct.error) as exc:
        parser.exit(1, f"Work verification failed: {exc}\n")


if __name__ == "__main__":
    main()
