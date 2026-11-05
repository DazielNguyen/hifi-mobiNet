"""Recompute the received N100 timing evidence; standard library, no inference.

--write creates derived tables/verification only; it never edits received files.
--source additionally checks every original file against its imported copy.
--preparation-source additionally checks the separately received preparation log.
--demo-source-root checks the three supplemental demo directories under Downloads.
--catalog-source compares the separately received active common97 catalog.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import statistics

ROOT = Path(__file__).resolve().parent
MODELS = ("piper-no-vits2-cpn", "piper-parir-no-vits2-cpn", "piper-seqir-no-vits2-cpn")
FLOAT32 = {"f32", "fp32", "float32"}
WEIGHTED = {"Convolution", "GroupConvolution", "FullyConnected", "Deconvolution"}
DEMO_FILES = {
    "20261105T074110Z-check.8oqXmf": {"check.log", "doctor.json"},
    "20261105T074122Z-server.CkuEHv": {"server.log"},
    "20261105T074123Z-a3c2e898": {"server.json"},
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12)


def safe(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def stats(rows):
    latency = [r["latency_s"] for r in rows]
    duration = [r["audio_seconds"] for r in rows]
    rtf = [t / d for t, d in zip(latency, duration)]
    p95 = lambda values: sorted(values)[math.ceil(.95 * len(values)) - 1]
    return {"n": len(rows), "mean_rtf": statistics.mean(rtf), "median_rtf": statistics.median(rtf),
            "p95_rtf": p95(rtf), "duration_weighted_rtf": sum(latency) / sum(duration),
            "mean_latency_s": statistics.mean(latency), "p95_latency_s": p95(latency),
            "total_inference_s": sum(latency), "total_audio_s": sum(duration)}


def compare(actual, stored, label):
    for key, value in actual.items():
        require(key in stored and close(value, stored[key]), f"{label}: {key} mismatch")


def runtime_audit(nodes, precision, allowed):
    weighted = [n for n in nodes if n.get("layerType") in WEIGHTED]
    require(bool(weighted), "no weighted runtime nodes")
    if precision == "fp32":
        require(all(n.get("runtimePrecision", "").lower() in FLOAT32 for n in weighted),
                "FP32 weighted node is not FP32")
        kernels = []
    else:
        kernels = [n for n in weighted if any(p in (n.get("runtimePrecision", "") + " " + n.get("execType", "")).lower()
                                             for p in ("i8", "u8", "int8"))]
        seen = set()
        for node in kernels:
            names = set(node.get("originalLayersNames", "").split(",")) | {node["name"]}
            matched = names & allowed
            require(node["layerType"] == "Convolution" and matched and not any("/dec/" in n for n in names),
                    "INT8 weighted node outside common97")
            seen.update(matched)
        require(len(kernels) == 97 and seen == allowed, "INT8 scope/count mismatch")
        require(all(n.get("runtimePrecision", "").lower() in FLOAT32 for n in weighted if n not in kernels),
                "unexpected precision outside INT8 kernels")
    decoder = [n for n in weighted if "/dec/" in n["name"] + n.get("originalLayersNames", "")]
    require(decoder and all(n.get("runtimePrecision", "").lower() in FLOAT32 for n in decoder),
            "decoder weighted precision not FP32")
    return {"int8_weighted_kernels": len(kernels), "weighted_nodes": len(weighted),
            "decoder_weighted_nodes": len(decoder), "decoder_weighted_precision": "FP32"}


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def preparation_log_audit(root, entries, source=None):
    logs = [e for e in entries if e["origin"] == "received_preparation_log"]
    require(len(logs) == 1, "expected one separately received preparation log")
    entry = logs[0]
    directory = root / "preparation-evidence"
    require({p.relative_to(root).as_posix() for p in directory.rglob("*") if p.is_file()} ==
            {entry["path"]}, "preparation inventory changed")
    if source is not None:
        require({p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()} ==
                {entry["source_relative_path"]}, "preparation source inventory changed")
        original = safe(source, entry["source_relative_path"])
        require(original.stat().st_size == entry["size_bytes"] and sha(original) == entry["sha256"],
                "preparation source hash/size mismatch")
    text = safe(root, entry["path"]).read_text(encoding="utf-8")
    lines = text.splitlines()
    marker = "COMMON97 READY: three Piper models prepared and local catalog activated."
    require(lines.count(marker) == 1, "preparation success marker missing/duplicated")
    calibration = [line for line in lines if line.startswith("Statistics collection")]
    require(len(calibration) == 3 and all("100% 60/60" in line for line in calibration),
            "incomplete calibration log")
    warnings = re.findall(r"(?im)^.*(?:Traceback|RuntimeWarning|overflow encountered|invalid value encountered|divide by zero|^(?:ERROR|WARNING):).*$", text)
    require(not warnings, "preparation diagnostic requires review")
    paths = {}
    for label in ("Catalog", "Evidence/WAVs"):
        values = [line.split(": ", 1)[1] for line in lines if line.startswith(label + ": ")]
        require(len(values) == 1, "missing/ambiguous preparation path: " + label)
        paths[label] = values[0]
    return {"status": "log_reports_success_work_evidence_received_separately", "source_directory": entry["source_directory"],
            "path": entry["path"], "size_bytes": entry["size_bytes"], "sha256": entry["sha256"],
            "calibration_blocks_completed_60_of_60": len(calibration), "success_marker": marker,
            "numeric_warning_or_error_lines": warnings, "reported_n100_paths": paths,
            "benchmark_artifact_binding": "checked separately through received preparation hashes; log alone has no measured-artifact hashes"}


def demo_startup_audit(root, entries, environment, benchmark_start, source_root=None):
    received = [e for e in entries if e["origin"] == "received_demo_startup"]
    expected = {(directory, name) for directory, names in DEMO_FILES.items() for name in names}
    require(len(received) == 4 and
            {(e["source_directory"], e["source_relative_path"]) for e in received} == expected,
            "demo startup source inventory differs")
    directory = root / "demo-evidence"
    require({p.relative_to(root).as_posix() for p in directory.rglob("*") if p.is_file()} ==
            {e["path"] for e in received}, "demo startup imported inventory differs")
    if source_root is not None:
        for folder, names in DEMO_FILES.items():
            original_dir = safe(source_root, folder)
            require(original_dir.is_dir() and
                    {p.relative_to(original_dir).as_posix() for p in original_dir.rglob("*") if p.is_file()} == names,
                    "demo startup source inventory differs: " + folder)
        for entry in received:
            original = safe(source_root, entry["source_directory"] + "/" + entry["source_relative_path"])
            require(original.stat().st_size == entry["size_bytes"] and sha(original) == entry["sha256"],
                    "demo startup source hash/size mismatch: " + entry["path"])
    paths = {e["source_relative_path"]: safe(root, e["path"]) for e in received}
    require(paths["check.log"].read_bytes() == paths["doctor.json"].read_bytes(),
            "check.log differs from doctor.json")
    doctor, server = read(paths["doctor.json"]), read(paths["server.json"])
    for label, record in (("doctor", doctor), ("server", server)):
        require(record["status"] == "ready" and record["inference_performed"] is False,
                label + " is not a readiness-only record")
        require(record["model_count"] == 6, label + " catalog entry count differs")
        require(all(record["host"][k] == environment[k]
                    for k in ("cpu_names", "logical_cpus", "platform", "python", "packages")),
                label + " environment differs from benchmark")
        require(record["host"]["hostname"] == "edge-n100", label + " host differs")
        frontend = record["frontend"]
        require(frontend["matched"] == frontend["total"] == 720 and frontend["language"] == "en-us",
                label + " frontend check incomplete")
        for key in ("python_module_sha256", "native_module_sha256"):
            require(re.fullmatch(r"[0-9a-f]{64}", frontend[key]), label + " frontend hash malformed")
    require(doctor["frontend"] == server["frontend"] and doctor["bundle"] == server["bundle"],
            "doctor/server frontend or bundle differs")
    catalog_hash = doctor["catalog_sha256"]
    require(re.fullmatch(r"[0-9a-f]{64}", catalog_hash) and server["catalog_sha256"] == catalog_hash,
            "doctor/server catalog hash differs")
    checked, started = [dt.datetime.fromisoformat(r["checked_utc"]) for r in (doctor, server)]
    require(checked <= started < benchmark_start, "demo/benchmark chronology differs")
    require(server["source_sha256"] == sha(root / "reference/n100_demo.py"), "demo source hash differs")
    require(server["openvino_cpu"] == "Intel(R) N100" and
            server["config"] == {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": 1,
                                 "INFERENCE_NUM_THREADS": 4, "INFERENCE_PRECISION_HINT": "f32",
                                 "EXECUTION_MODE_HINT": "ACCURACY"} and server["scales"] == [.667, 1., .8],
            "demo runtime settings differ")
    server_lines = paths["server.log"].read_text(encoding="utf-8").splitlines()
    require(server_lines == ["Checking N100 environment, IR hashes and Harvard-720 frontend...",
                             "N100 DEMO READY on port 8501. Logs: /home/capstone/benchmarks/n100-demo-runs/20261105T074123Z-a3c2e898"],
            "server startup log differs; review diagnostics")
    return {"status": "readiness_records_consistent_inference_not_performed",
            "received_files": len(received), "received_bytes": sum(e["size_bytes"] for e in received),
            "source_directories": sorted(DEMO_FILES), "doctor_checked_utc": doctor["checked_utc"],
            "server_checked_utc": server["checked_utc"], "catalog_sha256": catalog_hash,
            "catalog_bytes_received": True, "catalog_entry_count": server["model_count"],
            "frontend": server["frontend"], "demo_source_sha256": server["source_sha256"],
            "environment_matches_benchmark": True, "inference_performed": False,
            "synthesis_json_or_wav_received": False,
            "limitations": ["Frontend binary bytes were not received. The catalog was received and its bindings are checked separately.",
                            "The six catalog entries comprise six historical FP32 selections and three common97 INT8 selections.",
                            "No synthesis records/audio are provided by the startup files; readiness is not an inference-quality result."]}


def catalog_audit(root, entries, demo_startup, preparations, source=None):
    catalog_entries = [e for e in entries if e["origin"] == "received_demo_catalog"]
    require(len(catalog_entries) == 1, "expected one active common97 catalog")
    entry = catalog_entries[0]
    path = safe(root, entry["path"])
    require({p.relative_to(root).as_posix() for p in (root / "catalog-evidence").rglob("*") if p.is_file()} ==
            {entry["path"]}, "catalog inventory differs")
    if source is not None:
        require(source.read_bytes() == path.read_bytes(), "catalog source/copy differs")
    require(sha(path) == demo_startup["catalog_sha256"], "catalog does not match doctor/server")
    catalog = read(path)
    refs = root / "reference"
    models = read(refs / "models.json")
    require(catalog["schema_version"] == 2 and len(catalog["models"]) == 6 and
            {m["id"] for m in catalog["models"]} == {m["id"] for m in models["models"]}, "catalog model set/schema")
    require(catalog["expected_packages"] == read(refs / "piper-common97.json")["runtime_versions"], "catalog package identity")
    for field, filename in (("manifest", "models.json"), ("harvard", "metadata/harvard720.json")):
        require(catalog[field] == {"file": filename, "sha256": sha(refs / filename)}, "catalog input metadata identity")
    identity = catalog["common97"]
    require(identity["recipe"] == "piper-common97-v1" and identity["policy_sha256"] == sha(refs / "piper-common97.json") and
            identity["implementation_sha256"] == sha(refs / "common97.py"), "catalog common97 policy/source")
    work_name = read(root / "work-manifest.json")["source_directory"]
    require(identity["work"] == work_name and identity["n100_benchmark"] == "not run by this command", "catalog work identity")
    activated = dt.datetime.fromisoformat(identity["activated_utc"])
    require(activated < dt.datetime.fromisoformat(demo_startup["doctor_checked_utc"]), "catalog activation after preflight")
    rows = []
    for item in catalog["models"]:
        model = item["id"]
        original = next(m for m in models["models"] if m["id"] == model)
        require(item["sample_rate"] == original["sample_rate"] and item["checkpoint"] == original["checkpoint"], "catalog model source identity")
        prefix = "work-n100-clean-20261101T075458Z/" + model + "/"
        require(item["preparation"]["file"] == prefix + "preparation.json" and
                {f["file"] for f in item["artifacts"]["fp32"]} == {prefix + "fp32.xml", prefix + "fp32.bin"},
                "historical FP32 catalog path changed")
        if model not in MODELS:
            require(set(item["artifacts"]) == {"fp32"} and "int8_preparation" not in item, "INT8 enabled outside three Piper models")
            continue
        prep = read(root / "work-evidence" / model / "preparation.json")
        require(item["int8_recipe"] == identity["recipe"] and set(item["artifacts"]) == {"fp32", "int8"}, "Piper catalog precision/recipe")
        require(item["int8_preparation"] == {"file": f"{work_name}/{model}/preparation.json", "sha256": preparations[model]},
                "catalog preparation differs from benchmark")
        require(dt.datetime.fromisoformat(prep["finished_utc"]) <= activated, "catalog activated before preparation finished")
        expected = [{"file": f"{work_name}/{model}/" + f["file"], "sha256": f["sha256"]} for f in prep["artifacts"]["int8"]["files"]]
        require(item["artifacts"]["int8"] == expected, "catalog INT8 IR differs from measured preparation")
        require(item["quality_status"] == prep["quality_status"], "catalog quality status changed")
        comparison = {ext: next(f["sha256"] for f in item["artifacts"]["fp32"] if f["file"].endswith("." + ext)) ==
                          next(f["sha256"] for f in prep["artifacts"]["fp32"]["files"] if f["file"].endswith("." + ext))
                      for ext in ("xml", "bin")}
        rows.append({"model": model, "int8_preparation_sha256": preparations[model], "int8_ir_files_bound": 2,
                     "fp32_demo_hash_equals_benchmark": comparison})
    require({r["model"] for r in rows} == set(MODELS), "common97 INT8 catalog coverage")
    return {"status": "catalog_bound_to_demo_startup_and_measured_int8", "path": entry["path"], "sha256": sha(path),
            "size_bytes": path.stat().st_size, "catalog_entries": 6, "fp32_entries": 6, "common97_int8_entries": 3,
            "activated_utc": identity["activated_utc"], "models": rows,
            "limitations": ["Historical FP32 catalog XML hashes differ from the November 5 measured FP32 XML hashes; matching BIN hashes alone do not establish identical graphs.",
                            "Catalog bindings establish selected artifacts at recorded startup, not a particular generated demo WAV or current live server state."]}


def work_metadata_audit(root, entries, preparations):
    """Check the archived metadata chain; external binaries require verify_work.py."""
    imported = [e for e in entries if e["origin"] == "received_work_metadata"]
    require(len(imported) == 15 and
            {p.relative_to(root).as_posix() for p in (root / "work-evidence").rglob("*") if p.is_file()} ==
            {e["path"] for e in imported}, "work metadata inventory differs")
    inventory = read(root / "work-manifest.json")
    work = {e["path"]: e for e in inventory["files"]}
    require(len(work) == len(inventory["files"]) == 84 and
            sum(e["size_bytes"] for e in work.values()) == inventory["total_bytes"], "external work inventory")
    for entry in imported:
        record = work[entry["source_relative_path"]]
        require(record["repository_copy"] == entry["path"] and
                all(record[k] == entry[k] for k in ("sha256", "size_bytes")), "work import binding")
    for model, expected in preparations.items():
        directory = root / "work-evidence" / model
        require(sha(directory / "preparation.json") == expected, "received preparation differs from measured hash")
        prep = read(directory / "preparation.json")
        require(prep["status"] == "prepared" and prep["model"]["id"] == model, "work preparation identity/status")
        require(prep["alignment_validation"] == read(directory / "alignment-validation.json") and
                prep["fp32_parity"] == read(directory / "fp32_openvino_parity.json"), "duplicated work records differ")
        for precision in ("fp32", "int8"):
            for f in prep["artifacts"][precision]["files"]:
                require(work[model + "/" + f["file"]]["sha256"] == f["sha256"], "IR manifest/preparation binding")
        for row in prep["alignment_validation"]["cases"]:
            require(work[model + "/" + row["wav"]]["sha256"] == row["wav_sha256"], "alignment WAV manifest/preparation binding")
    recorded = read(root / "verification/work-verification.json")
    require(recorded["status"] == "received_work_artifacts_verified_without_inference" and
            recorded["work_manifest_sha256"] == sha(root / "work-manifest.json") and
            {m["model"]: m["preparation_sha256"] for m in recorded["models"]} == preparations,
            "recorded external work audit binding")
    return {"status": "archived_preparations_bound_to_benchmark", "received_metadata_files": len(imported),
            "external_work_files": len(work), "external_work_bytes": inventory["total_bytes"],
            "preparation_sha256": preparations,
            "external_artifact_audit": "verification/work-verification.json",
            "external_artifact_audit_sha256": sha(root / "verification/work-verification.json"),
            "external_bytes_rehashed_by_this_command": False,
            "repeat_external_check": "verify_work.py --source <received-work-directory>"}


def calculate(root, source=None, preparation_source=None, demo_source_root=None, catalog_source=None):
    manifest = read(root / "manifest.json")
    raw = root / "evidence"
    entries = manifest["files"]
    received = [e for e in entries if e["origin"] == "received"]
    require(len(received) == 77, "received inventory must contain 77 files")
    require(len({e["path"] for e in entries}) == len(entries), "duplicate inventory path")
    require({p.relative_to(raw).as_posix() for p in raw.rglob("*") if p.is_file()} ==
            {e["source_relative_path"] for e in received}, "received file set changed")
    for entry in entries:
        path = safe(root, entry["path"])
        require(path.stat().st_size == entry["size_bytes"] and sha(path) == entry["sha256"],
                "import hash/size mismatch: " + entry["path"])
        if source is not None and entry["origin"] == "received":
            original = safe(source, entry["source_relative_path"])
            require(original.stat().st_size == entry["size_bytes"] and sha(original) == entry["sha256"],
                    "source changed: " + entry["source_relative_path"])
    if source is not None:
        require({p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()} ==
                {e["source_relative_path"] for e in received}, "source inventory changed")
    preparation_log = preparation_log_audit(root, entries, preparation_source)

    refs = root / "reference"
    models = read(refs / "models.json")
    policy = read(refs / "piper-common97.json")
    for key in ("harvard", "split", "calibration", "calibration_dataset_rows"):
        require(sha(safe(refs, models[key]["file"])) == models[key]["sha256"], "metadata identity: " + key)
    harvard = read(refs / models["harvard"]["file"])
    require(len(harvard) == 720 and [r["idx"] for r in harvard] == list(range(720)), "Harvard index")
    selected = [m for m in models["models"] if m["id"] in MODELS]
    require([m["id"] for m in selected] == list(MODELS), "model source identity")
    for model in selected:
        require(model["fp32"]["sha256"] == policy["models"][model["id"]]["fp32_onnx_sha256"], "policy source identity")
        names = policy["models"][model["id"]]["allowed_node_names"]
        require(len(names) == len(set(names)) == 97, "policy names")
        require([sum(n.startswith(p) for n in names) for p in ("/enc_p/", "/dp/", "/flow/")] == [37, 20, 40],
                "policy breakdown")

    run = raw / "results"
    schedule = read(run / "schedule.json")
    cells = [(m, p, 4) for m in MODELS for p in ("fp32", "int8")]
    random.Random(1234).shuffle(cells)
    expected = [[r, *c] for r in range(3) for c in (cells if r % 2 == 0 else reversed(cells))]
    require(schedule == expected, "schedule/order does not match three-model protocol")
    previous = read(run / "summary.json")
    require(previous["status"] == "timing_complete", "incomplete timing")
    bindings = {r["cell"]: r for r in previous["sources"]}
    require(len(bindings) == len(previous["sources"]) == 18, "summary sources")
    expected_names = [f"r{r}_{m}_{p}_t{t}" for r, m, p, t in schedule]
    require(set(bindings) == set(expected_names), "source cell identities")
    require({p.name for p in run.iterdir() if p.is_dir()} == set(expected_names), "cell directories")
    require((raw / "benchmark.log").read_text().splitlines() == expected_names, "benchmark log order")
    warning = re.compile(r"Traceback|RuntimeWarning|overflow encountered|invalid value encountered|divide by zero", re.I)
    logs = list(raw.rglob("*.log"))
    require(len(logs) == 19, "missing log")
    for path in logs:
        require(not warning.search(path.read_text(encoding="utf-8")), "warning in " + path.name)

    by_length = sorted(range(720), key=lambda i: len(harvard[i]["phoneme_ids"]))
    warmup = [by_length[round(i * 719 / 9)] for i in range(10)]
    protocol = {"batch": 1, "device": "CPU", "scales": [.667, 1., .8], "warmup_calls": 10,
                "timed_region": "synchronous InferRequest.infer only; output copy/check/write excluded",
                "native_random": True, "cross_runtime_RNG_equivalence": False,
                "configuration": {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": 1, "INFERENCE_NUM_THREADS": 4,
                                  "INFERENCE_PRECISION_HINT": "f32", "EXECUTION_MODE_HINT": "ACCURACY"}}
    aggregates = collections.defaultdict(list)
    rss = collections.defaultdict(list)
    preparations, environment_ids = {}, set()
    environments, per_cell, observations = [], [], []
    prior_end = None
    for rnd, model, precision, threads in schedule:
        name = f"r{rnd}_{model}_{precision}_t{threads}"
        path = run / name
        record = read(path / "result.json")
        require(sha(path / "result.json") == bindings[name]["result_sha256"] and
                sha(path / "utterances.jsonl") == bindings[name]["utterances_sha256"], "summary source hashes: " + name)
        require(record["status"] == "measured" and
                [record["round"], record["model_id"], record["precision"], record["threads"]] == [rnd, model, precision, threads],
                "cell identity/status: " + name)
        require(record["script_sha256"] == sha(refs / "openvino_suite.py") and
                record["manifest_sha256"] == sha(refs / "models.json"), "script/manifest binding: " + name)
        require(record["sample_rate"] == 22050 and record["protocol"] == protocol and record["warmup_indices"] == warmup,
                "protocol: " + name)
        prep = record["preparation_sha256"]
        require(re.fullmatch(r"[0-9a-f]{64}", prep) and preparations.setdefault(model, prep) == prep, "preparation changed")
        start, end = (dt.datetime.fromisoformat(record[k]["utc"]) for k in ("environment_start", "environment_end"))
        require(start <= end and (prior_end is None or start >= prior_end), "overlapping/reversed cells")
        prior_end = end
        for key in ("environment_start", "environment_end"):
            env = record[key]
            require(env["cpu_names"] == ["Intel(R) N100"] and env["logical_cpus"] == 4 and
                    env["packages"] == policy["runtime_versions"], "runtime environment drift")
            environment_ids.add(json.dumps({k: env[k] for k in ("platform", "python", "cpu_names", "logical_cpus", "packages", "governors")}, sort_keys=True))
            environments.append(env)
        rows = [json.loads(line) for line in (path / "utterances.jsonl").read_text().splitlines()]
        order = list(range(720))
        random.Random(1234 + rnd).shuffle(order)
        require(len(rows) == 720 and [r["idx"] for r in rows] == order, "missing/duplicate/reordered utterance: " + name)
        for row in rows:
            require(type(row["samples"]) is int and row["samples"] % 256 == 0, "sample/frame count")
            require(all(math.isfinite(row[k]) and row[k] > 0 for k in ("samples", "audio_seconds", "latency_s", "rtf", "peak_abs")),
                    "invalid numeric observation: " + name)
            require(close(row["audio_seconds"], row["samples"] / 22050) and close(row["rtf"], row["latency_s"] / row["audio_seconds"]),
                    "RTF/duration formula: " + name)
            observations.append({"cell": name, **row})
        actual = stats(rows)
        compare(actual, record["summary"], name)
        audit = runtime_audit(read(path / "runtime_nodes.json"), precision, set(policy["models"][model]["allowed_node_names"]))
        require(type(record["peak_process_rss_bytes"]) is int and record["peak_process_rss_bytes"] > 0, "RSS")
        require(math.isfinite(record["load_and_compile_s"]) and record["load_and_compile_s"] > 0, "load/compile time")
        per_cell.append({"cell": name, "model": model, "precision": precision, "round": rnd, **actual,
                         "peak_process_rss_bytes": record["peak_process_rss_bytes"], **audit})
        aggregates[model, precision].extend(rows)
        rss[model, precision].append(record["peak_process_rss_bytes"])
    require(len(environment_ids) == 1, "mixed environments")

    tables, improvements = [], []
    for precision in ("fp32", "int8"):
        stored = previous["tables"][precision]
        with (run / f"summary_{precision}.csv").open(encoding="utf-8", newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        require([r["model"] for r in stored] == list(MODELS) and [r["model"] for r in csv_rows] == list(MODELS), "summary model set")
        for index, model in enumerate(MODELS):
            actual = {"threads": 4, "rounds": 3, **stats(aggregates[model, precision]), "peak_process_rss_bytes": max(rss[model, precision])}
            compare(actual, stored[index], model + " JSON")
            compare(actual, csv_rows[index], model + " CSV")
            tables.append({"model": model, "precision": precision, **actual})
    for model in MODELS:
        fp, quant = [next(r for r in tables if r["model"] == model and r["precision"] == p) for p in ("fp32", "int8")]
        improvements.append({"model": model, "rtf_reduction_percent": 100 * (1 - quant["duration_weighted_rtf"] / fp["duration_weighted_rtf"]),
                             "latency_reduction_percent": 100 * (1 - quant["mean_latency_s"] / fp["mean_latency_s"]),
                             "rss_reduction_percent": 100 * (1 - quant["peak_process_rss_bytes"] / fp["peak_process_rss_bytes"]),
                             "audio_duration_change_percent": 100 * (quant["total_audio_s"] / fp["total_audio_s"] - 1)})
    dates = [dt.datetime.fromisoformat(e["utc"]) for e in environments]
    demo_startup = demo_startup_audit(root, entries, environments[0], min(dates), demo_source_root)
    work_metadata = work_metadata_audit(root, entries, preparations)
    catalog = catalog_audit(root, entries, demo_startup, preparations, catalog_source)
    report = {"status": "received_timing_and_preparations_verified", "received_files": len(received),
              "received_bytes": sum(e["size_bytes"] for e in received),
              "reference_files": sum(e["origin"] == "repository_reference" for e in entries),
              "supplemental_received_files": 2 + demo_startup["received_files"] + work_metadata["received_metadata_files"],
              "preparation_log": preparation_log, "demo_startup": demo_startup, "work_metadata": work_metadata, "catalog": catalog,
              "cells": len(per_cell), "timed_observations": len(observations), "unique_texts": 720,
              "summary_source_hashes_checked": 36, "empty_worker_logs": sum(p.stat().st_size == 0 for p in logs),
              "runtime_audits_passed": 18, "preparation_hashes_verified": preparations,
              "script_sha256": sha(refs / "openvino_suite.py"), "manifest_sha256": sha(refs / "models.json"),
              "environment": {k: environments[0][k] for k in ("platform", "python", "cpu_names", "logical_cpus", "packages", "governors")},
              "start_utc": min(dates).isoformat(), "end_utc": max(dates).isoformat(),
              "start_utc_plus_07": min(dates).astimezone(dt.timezone(dt.timedelta(hours=7))).isoformat(),
              "end_utc_plus_07": max(dates).astimezone(dt.timezone(dt.timedelta(hours=7))).isoformat(),
              "elapsed_wall_seconds": (max(dates) - min(dates)).total_seconds(),
              "max_boundary_temperature_celsius": max(float(v) for e in environments for v in e["thermal_millidegrees_celsius"].values()) / 1000,
              "minimum_audio_seconds": min(r["audio_seconds"] for r in observations),
              "maximum_audio_seconds": max(r["audio_seconds"] for r in observations),
              "audio_over_10_seconds": sum(r["audio_seconds"] > 10 for r in observations),
              "maximum_rtf": max(r["rtf"] for r in observations), "rtf_at_least_1": sum(r["rtf"] >= 1 for r in observations),
              "minimum_peak_abs": min(r["peak_abs"] for r in observations), "maximum_peak_abs": max(r["peak_abs"] for r in observations),
              "peak_above_1": sum(r["peak_abs"] > 1 for r in observations),
              "longest_observations": sorted(observations, key=lambda r: r["audio_seconds"], reverse=True)[:6],
              "limitations": manifest["missing_evidence"]}
    return {"tables": tables, "improvements": improvements, "cells": per_cell}, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--preparation-source", type=Path)
    parser.add_argument("--demo-source-root", type=Path)
    parser.add_argument("--catalog-source", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    try:
        tables, report = calculate(ROOT, args.source, args.preparation_source, args.demo_source_root, args.catalog_source)
        derived = ROOT / "verification"
        outputs = {"recomputed.json": tables, "verification.json": report}
        if args.write:
            derived.mkdir(exist_ok=True)
            for name, data in outputs.items():
                (derived / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            (derived / "recomputed.csv").write_text(csv_text(tables["tables"]), encoding="utf-8", newline="")
        else:
            for name, data in outputs.items():
                require(read(derived / name) == data, "derived output mismatch: " + name)
            require((derived / "recomputed.csv").read_text(encoding="utf-8") == csv_text(tables["tables"]), "derived CSV mismatch")
            listed = set()
            for line in (ROOT / "SHA256SUMS").read_text().splitlines():
                digest, name = line.split("  ", 1)
                require(name not in listed, "duplicate checksum")
                listed.add(name)
                require(sha(safe(ROOT, name)) == digest, "package checksum: " + name)
            actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()
                      and p.name != "SHA256SUMS" and "__pycache__" not in p.parts}
            require(listed == actual, "checksum inventory differs")
        print(json.dumps({"status": report["status"], "received_files": report["received_files"], "cells": report["cells"],
                          "timed_observations": report["timed_observations"], "source_copy_checked": args.source is not None,
                          "preparation_log": report["preparation_log"]["status"],
                          "preparation_source_copy_checked": args.preparation_source is not None,
                          "demo_startup": report["demo_startup"]["status"],
                          "demo_source_copy_checked": args.demo_source_root is not None,
                          "full_package_checksums_checked": not args.write,
                          "work_metadata": report["work_metadata"]["status"],
                          "catalog": report["catalog"]["status"],
                          "catalog_source_copy_checked": args.catalog_source is not None,
                          "external_artifact_recheck": "run verify_work.py --source <work-directory>",
                          "not_verified": "Independent INT8 quality and individual demo synthesis outputs remain pending"}, indent=2))
    except (OSError, ValueError, KeyError, StopIteration, TypeError) as exc:
        parser.exit(1, f"Verification failed: {exc}\n")


if __name__ == "__main__":
    main()
