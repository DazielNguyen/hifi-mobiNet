"""Verify this saved evidence package without loading or executing any model.

Python standard library only. Default mode works in a clean Git checkout;
--with-local-artifacts also checks the original files kept under ignored .local/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import re
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_path(root, name):
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in name, "Unsafe path: " + name)
    path = root.joinpath(*relative.parts).resolve()
    path.relative_to(root.resolve())
    return path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(a, b):
    return math.isfinite(float(a)) and math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-11)


def summary(rows):
    latency = [r["latency_s"] for r in rows]
    duration = [r["audio_seconds"] for r in rows]
    rtfs = [t / d for t, d in zip(latency, duration)]
    p95 = lambda values: sorted(values)[math.ceil(.95 * len(values)) - 1]
    return {"n": len(rows), "mean_rtf": statistics.mean(rtfs), "median_rtf": statistics.median(rtfs),
            "p95_rtf": p95(rtfs), "duration_weighted_rtf": sum(latency) / sum(duration),
            "mean_latency_s": statistics.mean(latency), "p95_latency_s": p95(latency),
            "total_inference_s": sum(latency), "total_audio_s": sum(duration)}


def compare_stats(actual, stored, context):
    for key, value in actual.items():
        require(key in stored and close(value, stored[key]), context + ": changed statistic " + key)


def run(root, with_local):
    checksums = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        require(name not in checksums, "Duplicate checksum path")
        require(sha(safe_path(root, name)) == digest, "Checksum mismatch: " + name)
        checksums[name] = digest
    actual_files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
                    and ".local" not in p.relative_to(root).parts and "__pycache__" not in p.relative_to(root).parts
                    and p != root / "SHA256SUMS"}
    require(actual_files == set(checksums), "SHA256SUMS inventory differs from package files")
    inventory = read(root / "manifest.json")
    entries = inventory["entries"]
    originals = {(e["collection"], e["source_member"]): e for e in entries}
    require(len(originals) == len(entries), "Duplicate source member")
    local_checked = 0
    for entry in entries:
        if "tracked_path" in entry:
            path = safe_path(root, entry["tracked_path"])
            require(path.stat().st_size == entry["tracked_size"] and sha(path) == entry["tracked_sha256"], "Imported file mismatch: " + entry["tracked_path"])
            if entry["disposition"] == "tracked_byte_identical":
                require(entry["source_sha256"] == entry["tracked_sha256"] and not entry["transformations"], "Undeclared transformation")
        if with_local and "local_original" in entry:
            original = safe_path(root, entry["local_original"])
            require(original.stat().st_size == entry["source_size"] and sha(original) == entry["source_sha256"], "Local original mismatch: " + entry["source_member"])
            if entry["disposition"] == "tracked_derivative":
                data = original.read_bytes().replace(b"/home/" + b"capstone", b"/home/<user>")
                data = re.sub(rb"C:[/\\]+Users[/\\]+duyng", b"<WINDOWS_USER>", data)
                require(data == safe_path(root, entry["tracked_path"]).read_bytes(), "Redaction differs from declared policy")
            local_checked += 1
    evidence = root / "evidence"
    data = read(evidence / "models.json")
    models = {m["id"] for m in data["models"]}
    require(len(models) == 6, "Expected six models")
    run_name = "results-n100-t4-20261101T080423Z"
    work = "work-n100-clean-20261101T075458Z"
    run_dir = evidence / run_name
    jobs = read(run_dir / "schedule.json")
    expected = {(r, m, p, 4) for r in range(3) for m in models for p in ("fp32", "int8")}
    require(len(jobs) == 36 and set(map(tuple, jobs)) == expected, "Missing/duplicate cell in schedule")
    previous = read(run_dir / "summary.json")
    sources = {s["cell"]: s for s in previous["sources"]}
    require(len(sources) == 36, "Incomplete summary source list")
    aggregates = {}
    for rnd, model, precision, threads in jobs:
        name = f"r{rnd}_{model}_{precision}_t{threads}"
        directory = run_dir / name
        record = read(directory / "result.json")
        rows = [json.loads(s) for s in (directory / "utterances.jsonl").read_text(encoding="utf-8").splitlines()]
        require(record["status"] == "measured" and [record[k] for k in ("round", "model_id", "precision", "threads")] == [rnd, model, precision, threads], name + ": cell identity/status")
        require(len(rows) == 720 and {r["idx"] for r in rows} == set(range(720)), name + ": sentence coverage")
        order = list(range(720)); random.Random(1234 + rnd).shuffle(order)
        require([r["idx"] for r in rows] == order, name + ": sentence order")
        for file, field in (("result.json", "result_sha256"), ("utterances.jsonl", "utterances_sha256")):
            source = originals["final", run_name + "/" + name + "/" + file]
            require(source["source_sha256"] == sources[name][field], name + ": summary provenance")
        for file, field in (("openvino_suite.py", "script_sha256"), ("models.json", "manifest_sha256"), (work + "/" + model + "/preparation.json", "preparation_sha256")):
            require(record[field] == originals["final", file]["source_sha256"], name + ": execution provenance")
        for row in rows:
            require(all(math.isfinite(row[k]) and row[k] > 0 for k in ("latency_s", "audio_seconds", "rtf", "samples", "peak_abs")), name + ": invalid observation")
            require(close(row["audio_seconds"], row["samples"] / record["sample_rate"]) and close(row["rtf"], row["latency_s"] / row["audio_seconds"]), name + ": RTF or duration formula")
        compare_stats(summary(rows), record["summary"], name)
        aggregates.setdefault((model, precision), []).extend(rows)
    for (model, precision), rows in aggregates.items():
        actual = summary(rows)
        stored = next(t for t in previous["tables"][precision] if t["model"] == model)
        compare_stats(actual, stored, model + " " + precision)
        with (run_dir / f"summary_{precision}.csv").open(encoding="utf-8", newline="") as stream:
            csv_row = next(r for r in csv.DictReader(stream) if r["model"] == model)
        compare_stats(actual, csv_row, model + " CSV")
    links = 0
    validation = evidence / "timing-validation-20261101T080340.996001Z.sha256"
    for line in validation.read_text().splitlines():
        digest, name = line.split("  ", 1)
        require(originals["final", PurePosixPath(name).as_posix()]["source_sha256"] == digest, "Original preflight identity mismatch: " + name)
        links += 1
    warning = re.compile(r"RuntimeWarning|overflow encountered|invalid value encountered|divide by zero|Traceback", re.I)
    for path in [evidence / f"{work}.prepare.log", evidence / f"{run_name}.run.log", *run_dir.glob("*.log")]:
        require(not warning.search(path.read_text(encoding="utf-8")), "Final-session warning: " + path.name)
    return {"status": "passed", "tracked_checksums": len(checksums), "imported_text_files": sum("tracked_path" in e for e in entries),
            "declared_path_derivatives": sum(e["disposition"] == "tracked_derivative" for e in entries),
            "local_originals_checked": local_checked, "local_artifacts_requested": with_local,
            "excluded_cache_files": sum(e["disposition"] == "excluded_cache" for e in entries),
            "original_preflight_identity_links": links, "cells": len(jobs), "timed_observations": sum(map(len, aggregates.values())),
            "interpretation": "Integrity and stored timing verified; retained duration outliers and pending INT8 quality remain documented."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-local-artifacts", action="store_true", help="Require original files under ignored .local/ too")
    args = parser.parse_args()
    try:
        result = run(Path(__file__).resolve().parent, args.with_local_artifacts)
    except (ValueError, OSError, KeyError, StopIteration) as exc:
        parser.exit(1, f"Verification failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
