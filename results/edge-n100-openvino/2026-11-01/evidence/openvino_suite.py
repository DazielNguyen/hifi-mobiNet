"""Portable Intel N100 CPU/OpenVINO suite. No training or checkpoint loading.

Run doctor, prepare, then run on the N100. Each command preserves its records.
INT8 is a new, uniform NNCF experiment for all six models, not historical PTQ.
Heavy dependencies are imported only by the commands that need them.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback

SCALES = [0.667, 1.0, 0.8]
PARITY_IDS = [0, 1, 2, 59, 119, 179, 239, 299, 359, 419, 479, 599, 719]
SCOPES = ("/enc_p/", "/dp/", "/flow/")

# Candidate revision: do not quantize dynamic masked spline operations.
# Conv layers pre/convs/proj inside each duration flow remain eligible.
# This changes activation quantization boundaries; it is a separate NNCF experiment.
NUMERIC_REVISION = "n100-spline-fp32-v1-20261101"
DIAGNOSTIC_GUARD_SHA256 = "28090dbae369f24a22a7c04179191d959265e606c541e07e72b794e5d49ee60c"


def is_masked_duration_spline(name):
    import re
    match = re.match(r"^/dp/flows\.[0-9]+/(.*)$", name)
    if match is None:
        return False
    tail = match.group(1)
    return not any(tail.startswith(prefix) for prefix in ("pre/", "convs/", "proj/"))



def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def verify_file(root, item):
    path = (Path(root) / item["file"]).resolve()
    path.relative_to(Path(root).resolve())
    if not path.is_file() or sha(path) != item["sha256"]:
        raise ValueError(f"missing file or SHA-256 mismatch: {path}")
    return path


def environment(require_target=True):
    cpu = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else platform.processor()
    names = sorted({line.split(":", 1)[1].strip() for line in cpu.splitlines() if line.startswith("model name")})
    if require_target and (platform.system() != "Linux" or not any("N100" in n.split() for n in names)):
        raise RuntimeError(f"this measurement requires Linux on Intel N100; observed {platform.system()} {names}")
    versions = {}
    for name in ("openvino", "nncf", "numpy", "onnx", "onnxruntime"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    sensors = {}
    for p in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            sensors[str(p)] = p.read_text().strip()
        except OSError:
            pass
    governors = {}
    for p in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor"):
        try:
            governors[str(p)] = p.read_text().strip()
        except OSError:
            pass
    return {"utc": utc(), "platform": platform.platform(), "python": platform.python_version(),
            "cpu_names": names, "logical_cpus": os.cpu_count(), "packages": versions,
            "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
            "thermal_millidegrees_celsius": sensors, "governors": governors}


def config(threads):
    return {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": 1, "INFERENCE_NUM_THREADS": threads,
            "INFERENCE_PRECISION_HINT": "f32", "EXECUTION_MODE_HINT": "ACCURACY"}


def feeds(row):
    import numpy as np
    ids = np.asarray(row["phoneme_ids"], dtype=np.int64)[None, :]
    return {"input": ids, "input_lengths": np.asarray([ids.shape[1]], dtype=np.int64),
            "scales": np.asarray(SCALES, dtype=np.float32)}


def audio_info(wave, sample_rate):
    import numpy as np
    wave = np.asarray(wave)
    if wave.ndim != 3 or wave.shape[:2] != (1, 1) or wave.shape[-1] <= 0:
        raise ValueError(f"expected mono [1,1,N] waveform, got {wave.shape}")
    if not np.isfinite(wave).all() or float(np.std(wave)) <= 1e-6:
        raise ValueError("nonfinite or silent waveform")
    return {"samples": int(wave.shape[-1]), "audio_seconds": wave.shape[-1] / sample_rate,
            "peak_abs": float(np.abs(wave).max())}


def summary(rows):
    if not rows:
        raise ValueError("no measured utterances")
    latencies = [float(r["latency_s"]) for r in rows]
    durations = [float(r["audio_seconds"]) for r in rows]
    if any(not math.isfinite(v) or v <= 0 for v in latencies + durations):
        raise ValueError("invalid latency or audio duration")
    rtfs = [t / d for t, d in zip(latencies, durations)]
    def p95(values):
        ordered = sorted(values)
        return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]
    return {"n": len(rows), "mean_rtf": statistics.mean(rtfs), "median_rtf": statistics.median(rtfs),
            "p95_rtf": p95(rtfs), "duration_weighted_rtf": sum(latencies) / sum(durations),
            "mean_latency_s": statistics.mean(latencies), "p95_latency_s": p95(latencies),
            "total_inference_s": sum(latencies), "total_audio_s": sum(durations)}


def runtime_nodes(compiled):
    rows = []
    for node in compiled.get_runtime_model().get_ops():
        info = node.get_rt_info()
        row = {"name": node.get_friendly_name()}
        for key in ("layerType", "runtimePrecision", "execType", "originalLayersNames"):
            if key in info:
                value = info[key]
                row[key] = str(getattr(value, "value", value))
        rows.append(row)
    return rows


def int8_kernels(nodes):
    return [n for n in nodes if n.get("layerType") in ("Convolution", "GroupConvolution", "FullyConnected")
            and any(t in (n.get("runtimePrecision", "") + " " + n.get("execType", "")).lower()
                    for t in ("i8", "u8", "int8"))]


def check_runtime_precision(nodes, precision):
    weighted = [n for n in nodes if n.get("layerType") in ("Convolution", "GroupConvolution", "FullyConnected", "Deconvolution")]
    if not weighted:
        raise ValueError("no weighted execution nodes identified")
    if precision == "fp32":
        if any(n.get("runtimePrecision", "").lower() not in ("f32", "fp32", "float32") for n in weighted):
            raise ValueError("FP32 weighted execution precision could not be verified")
    else:
        kernels = int8_kernels(nodes)
        if not kernels:
            raise ValueError("no executed INT8 weighted kernel identified")
        for n in kernels:
            original = n.get("originalLayersNames", "") + " " + n["name"]
            if "/dec/" in original or not any(scope in original for scope in SCOPES):
                raise ValueError("INT8 weighted kernel outside enc_p/dp/flow scope")


def manifest(bundle):
    value = read(bundle / "models.json")
    if len(value["models"]) != 6 or len({m["id"] for m in value["models"]}) != 6:
        raise ValueError("expected the six distinct, completed internal models")
    for key in ("harvard", "calibration", "split", "calibration_dataset_rows"):
        verify_file(bundle, value[key])
    return value


def harvard_rows(bundle, data):
    rows = read(verify_file(bundle, data["harvard"]))
    if len(rows) != 720 or [r["idx"] for r in rows] != list(range(720)):
        raise ValueError("expected Harvard-720 in canonical idx order")
    return rows


def parity_check(bundle, source, rows, output, ov, core):
    import numpy as np
    import onnx
    import onnx_reference as reference
    original = onnx.load(str(source))
    if any(n.op_type == "ReduceL2" and any(s in n.name for s in SCOPES) for n in original.graph.node):
        raise ValueError("unfolded flow weight norm: do not use a known-defect graph")
    matched, names = reference.matched_noise_graph(original, 192)
    if reference.strip_matched_noise(matched).SerializeToString() != reference.without_random_nodes(original).SerializeToString():
        raise ValueError("matched-noise surgery changed the model")
    path = output / "fp32_matched_validation.onnx"
    onnx.checker.check_model(matched)
    onnx.save(matched, str(path))
    ort = reference.ort_session(path, intra_threads=2)
    compiled = core.compile_model(core.read_model(str(path)), "CPU", config(2))
    request = compiled.create_infer_request()
    results = []
    for idx in PARITY_IDS:
        data = feeds(rows[idx])
        rng = np.random.default_rng(1234 + idx)
        data[names["duration_input"]] = rng.standard_normal((1, 2, len(rows[idx]["phoneme_ids"]))).astype(np.float32)
        data[names["acoustic_input"]] = rng.standard_normal((1, 192, 10000)).astype(np.float32)
        expected = ort.run(None, data)[0]
        request.infer(data)
        observed = request.get_output_tensor(0).data
        result = {"idx": idx, **reference.parity(expected, observed)}
        results.append(result)
        write(output / "fp32_openvino_parity.json", results)
        if not result["passed"]:
            raise ValueError(f"OpenVINO/ORT matched-noise FP32 parity failed: {result}")
    return results


def prepare(bundle, work, selected):
    import numpy as np
    import nncf
    import openvino as ov
    import onnx_reference as reference
    env = environment()
    data = manifest(bundle)
    rows = harvard_rows(bundle, data)
    cal = [json.loads(s) for s in verify_file(bundle, data["calibration"]).read_text().splitlines() if s.strip()]
    subset = [json.loads(s) for s in verify_file(bundle, data["calibration_dataset_rows"]).read_text().splitlines() if s.strip()]
    check = reference.check_calibration(cal, read(verify_file(bundle, data["split"])), subset)
    if not check["passed"] or check["n"] != 60:
        raise ValueError(f"calibration is not the fixed 60 train-only rows: {check}")
    work.mkdir(parents=True, exist_ok=True)
    core = ov.Core()
    for item in data["models"]:
        if selected != "all" and item["id"] != selected:
            continue
        output = work / item["id"]
        output.mkdir()  # never overwrite an earlier attempt
        record = {"status": "running", "started_utc": utc(), "environment": env, "model": item,
                  "experiment": "new NNCF selective INT8, all six models; not historical ORT INT8",
                  "calibration": check, "quality_status": "INT8 listening/WER/UTMOS pending",
                  "script_sha256": sha(__file__), "manifest_sha256": sha(bundle / "models.json"),
                  "rng": "Python/NumPy seed 1234; native OpenVINO graph RNG not forced; no byte reproducibility claim"}
        try:
            random.seed(1234)
            np.random.seed(1234)
            source = verify_file(bundle, item["fp32"])
            record["fp32_parity"] = parity_check(bundle, source, rows, output, ov, core)
            model = core.read_model(str(source))
            ov.save_model(model, str(output / "fp32.xml"), compress_to_fp16=False)
            weighted = {"Convolution", "GroupConvolution", "ConvolutionBackpropData", "GroupConvolutionBackpropData", "MatMul"}
            allowed = [n.get_friendly_name() for n in model.get_ops()
                       if n.get_type_name() in ("Convolution", "GroupConvolution")
                       and any(s in n.get_friendly_name() for s in SCOPES)]
            if not allowed:
                raise ValueError("cannot identify enc_p/dp/flow Conv scope after conversion")
            ignored = [n.get_friendly_name() for n in model.get_ops()
                       if ("/dec/" in n.get_friendly_name() and n.get_type_name() not in ("Constant", "Parameter", "Result"))
                       or (n.get_type_name() in weighted and n.get_friendly_name() not in allowed)]
            spline_ignored = sorted({n.get_friendly_name() for n in model.get_ops()
                                     if is_masked_duration_spline(n.get_friendly_name())
                                     and n.get_type_name() not in ("Constant", "Parameter", "Result")})
            if not spline_ignored:
                raise ValueError("cannot identify masked duration spline scope; review the graph")
            if set(spline_ignored).intersection(allowed):
                raise ValueError("spline policy unexpectedly excludes an eligible convolution")
            ignored = sorted(set(ignored).union(spline_ignored))
            helper_path = Path(__file__).resolve().parent / "diagnose_n100_calibration.py"
            if sha(helper_path) != DIAGNOSTIC_GUARD_SHA256:
                raise ValueError("numeric guard is missing or changed")
            from diagnose_n100_calibration import Recorder
            guard = Recorder(output, np)
            record["numeric_guard"] = {"status": "running", "helper_sha256": sha(helper_path),
                                       "revision": NUMERIC_REVISION}
            record["experiment"] = "NNCF selective INT8 v2 candidate: masked duration spline FP32; not original preparation"
            dataset = nncf.Dataset(cal, feeds)
            record["quantization"] = {"subset_size": 60, "preset": "MIXED", "target_device": "CPU",
                                      "fast_bias_correction": True, "allowed_convolutions": allowed, "ignored_nodes": ignored,
                                      "revision": NUMERIC_REVISION, "spline_fp32_ignored_nodes": spline_ignored,
                                      "scope_change": "exclude masked duration spline activations; eligible Conv scopes unchanged"}
            quantized = guard.guarded_call(nncf.quantize, model, dataset, subset_size=60,
                                          preset=nncf.QuantizationPreset.MIXED,
                                          target_device=nncf.TargetDevice.CPU, fast_bias_correction=True,
                                          ignored_scope=nncf.IgnoredScope(names=ignored))
            record["numeric_guard"]["status"] = "no_numerical_issue_intercepted"
            ov.save_model(quantized, str(output / "int8.xml"), compress_to_fp16=False)
            record["artifacts"] = {}
            for precision in ("fp32", "int8"):
                compiled = core.compile_model(core.read_model(str(output / f"{precision}.xml")), "CPU", config(4))
                request = compiled.create_infer_request()
                samples = []
                for idx in (0, 359, 719):
                    request.infer(feeds(rows[idx]))
                    wave = request.get_output_tensor(0).data
                    samples.append({"idx": idx, **audio_info(wave, item["sample_rate"])})
                    # Listen to these actual native-random outputs; never label smoke as quality validation.
                    import wave as wav
                    pcm = (wave.reshape(-1) * (32767 / max(0.01, float(np.abs(wave).max())))).clip(-32768, 32767).astype("<i2")
                    with wav.open(str(output / f"{precision}_{idx:04d}.wav"), "wb") as f:
                        f.setnchannels(1); f.setsampwidth(2); f.setframerate(item["sample_rate"]); f.writeframes(pcm.tobytes())
                nodes = runtime_nodes(compiled)
                write(output / f"{precision}_runtime_nodes.json", nodes)
                check_runtime_precision(nodes, precision)
                if precision == "int8":
                    kernels = int8_kernels(nodes)
                    if not kernels:
                        raise ValueError("no executed INT8 weighted kernel identified; cannot label this INT8 timing")
                    if any("/dec/" in (n.get("originalLayersNames", "") + n["name"]) for n in kernels):
                        raise ValueError("INT8 decoder kernel found outside the selective scope")
                    record["int8_weighted_kernel_count"] = len(kernels)
                record["artifacts"][precision] = {"files": [{"file": f"{precision}.{ext}", "sha256": sha(output / f"{precision}.{ext}")}
                                                                    for ext in ("xml", "bin")], "smoke": samples}
            record.update(status="prepared", finished_utc=utc())
        except BaseException:
            record.update(status="failed", error=traceback.format_exc(), finished_utc=utc())
            raise
        finally:
            write(output / "preparation.json", record)


def worker(args):
    import numpy as np
    import openvino as ov
    import resource
    env = environment()
    data = manifest(args.bundle)
    item = next(m for m in data["models"] if m["id"] == args.model)
    prepared = args.work / args.model
    prep = read(prepared / "preparation.json")
    if (prep["status"] != "prepared" or prep["model"] != item
            or prep["environment"]["packages"] != env["packages"]
            or prep["script_sha256"] != sha(__file__) or prep["manifest_sha256"] != sha(args.bundle / "models.json")):
        raise ValueError("preparation failed, model changed, or runtime versions changed; prepare in this environment first")
    for f in prep["artifacts"][args.precision]["files"]:
        verify_file(prepared, f)
    args.output.mkdir(parents=True)  # refuse old results
    record = {"status": "running", "model_id": args.model, "precision": args.precision, "threads": args.threads,
              "round": args.round, "environment_start": env, "preparation_sha256": sha(prepared / "preparation.json"),
              "sample_rate": item["sample_rate"],
              "manifest_sha256": sha(args.bundle / "models.json"), "script_sha256": sha(__file__),
              "quality_status": prep["quality_status"], "protocol": {"batch": 1, "device": "CPU", "scales": SCALES,
              "warmup_calls": 10, "timed_region": "synchronous InferRequest.infer only; output copy/check/write excluded",
              "native_random": True, "cross_runtime_RNG_equivalence": False, "configuration": config(args.threads)}}
    try:
        core = ov.Core()
        start = time.perf_counter()
        model = core.read_model(str(prepared / f"{args.precision}.xml"))
        compiled = core.compile_model(model, "CPU", config(args.threads))
        record["load_and_compile_s"] = time.perf_counter() - start
        if {p.get_any_name() for p in compiled.inputs} != {"input", "input_lengths", "scales"} or len(compiled.outputs) != 1:
            raise ValueError("unexpected model I/O")
        request = compiled.create_infer_request()
        rows = harvard_rows(args.bundle, data)
        order = list(range(720))
        random.Random(1234 + args.round).shuffle(order)
        prepared_inputs = [feeds(r) for r in rows]
        by_length = sorted(range(720), key=lambda i: len(rows[i]["phoneme_ids"]))
        warmup = [by_length[round(i * 719 / 9)] for i in range(10)]
        record["warmup_indices"] = warmup
        for idx in warmup:
            request.infer(prepared_inputs[idx])
            audio_info(request.get_output_tensor(0).data, item["sample_rate"])
        nodes = runtime_nodes(compiled)
        write(args.output / "runtime_nodes.json", nodes)
        check_runtime_precision(nodes, args.precision)
        measured = []
        with (args.output / "utterances.jsonl").open("x", encoding="utf-8") as stream:
            for idx in order:
                for name, array in prepared_inputs[idx].items():
                    request.set_tensor(name, ov.Tensor(array))
                start = time.perf_counter_ns()
                request.infer(share_outputs=True)
                elapsed = (time.perf_counter_ns() - start) / 1e9
                wave = request.get_output_tensor(0).data
                row = {"idx": rows[idx]["idx"], "latency_s": elapsed, **audio_info(wave, item["sample_rate"])}
                row["rtf"] = elapsed / row["audio_seconds"]
                measured.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n"); stream.flush()
        record.update(status="measured", summary=summary(measured), environment_end=environment(),
                      peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    except BaseException:
        record.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        write(args.output / "result.json", record)


def schedule(models, threads, rounds):
    cells = [(m["id"], p, t) for t in threads for m in models for p in ("fp32", "int8")]
    random.Random(1234).shuffle(cells)
    return [(r, *cell) for r in range(rounds) for cell in (cells if r % 2 == 0 else list(reversed(cells)))]


def run_matrix(args):
    env = environment()
    data = manifest(args.bundle)
    for item in data["models"]:
        p = read(args.work / item["id"] / "preparation.json")
        if (p["status"] != "prepared" or p["model"] != item
                or p["environment"]["packages"] != env["packages"]
                or p["script_sha256"] != sha(__file__) or p["manifest_sha256"] != sha(args.bundle / "models.json")):
            raise ValueError(f"preparation incomplete or environment/source changed: {item['id']}")
        for precision in ("fp32", "int8"):
            for f in p["artifacts"][precision]["files"]:
                verify_file(args.work / item["id"], f)
    args.output.mkdir(parents=True)
    jobs = schedule(data["models"], args.threads, args.rounds)
    write(args.output / "schedule.json", jobs)
    for rnd, model, precision, threads in jobs:
        name = f"r{rnd}_{model}_{precision}_t{threads}"
        cmd = [sys.executable, str(Path(__file__).resolve()), "worker", "--bundle", str(args.bundle.resolve()),
               "--work", str(args.work.resolve()), "--output", str((args.output / name).resolve()),
               "--model", model, "--precision", precision, "--threads", str(threads), "--round", str(rnd)]
        print(name, flush=True)
        with (args.output / f"{name}.log").open("x") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    summarize(args.output)


def summarize(output):
    jobs = read(output / "schedule.json")
    if not jobs or len({tuple(j) for j in jobs}) != len(jobs):
        raise ValueError("empty or duplicate measurement schedule")
    models = {j[1] for j in jobs}
    threads = {j[3] for j in jobs}
    rounds = {j[0] for j in jobs}
    expected = {(r, m, p, t) for r in range(max(rounds) + 1) for m in models
                for p in ("fp32", "int8") for t in threads}
    if len(models) != 6 or not threads.issubset({1, 2, 4}) or {tuple(j) for j in jobs} != expected:
        raise ValueError("incomplete six-model FP32/INT8 schedule")
    groups = {}
    identities = set()
    preparation_hashes = {}
    sources = []
    for rnd, model, precision, threads in jobs:
        path = output / f"r{rnd}_{model}_{precision}_t{threads}"
        record = read(path / "result.json")
        rows = [json.loads(s) for s in (path / "utterances.jsonl").read_text().splitlines()]
        if record["status"] != "measured" or len(rows) != 720 or {r["idx"] for r in rows} != set(range(720)):
            raise ValueError(f"incomplete cell: {path}")
        if [record["round"], record["model_id"], record["precision"], record["threads"]] != [rnd, model, precision, threads]:
            raise ValueError(f"cell identity mismatch: {path}")
        identities.add(json.dumps({"manifest": record["manifest_sha256"], "script": record["script_sha256"],
                                   "packages": record["environment_start"]["packages"],
                                   "cpu": record["environment_start"]["cpu_names"]}, sort_keys=True))
        if preparation_hashes.setdefault(model, record["preparation_sha256"]) != record["preparation_sha256"]:
            raise ValueError(f"preparation changed between measurement cells: {model}")
        sources.append({"cell": path.name, "result_sha256": sha(path / "result.json"),
                        "utterances_sha256": sha(path / "utterances.jsonl")})
        group = groups.setdefault((model, precision, threads), {"rows": [], "rss": [], "rounds": 0})
        group["rows"].extend(rows); group["rss"].append(record["peak_process_rss_bytes"]); group["rounds"] += 1
    if len(identities) != 1:
        raise ValueError("cannot aggregate different sources, software versions or CPU identities")
    tables = {p: [] for p in ("fp32", "int8")}
    for (model, precision, threads), group in sorted(groups.items()):
        tables[precision].append({"model": model, "threads": threads, "rounds": group["rounds"],
                                  **summary(group["rows"]), "peak_process_rss_bytes": max(group["rss"])})
    for precision, table in tables.items():
        if not table:
            raise ValueError(f"no complete {precision} cells")
        with (output / f"summary_{precision}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    write(output / "summary.json", {"status": "timing_complete", "quality_status": "not certified by this timing suite",
                                    "note": "repeated sentences are paired observations, not independent new utterances",
                                    "sources": sources, "tables": tables})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "prepare", "run", "worker", "summarize"):
        p = sub.add_parser(name)
        p.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
        p.add_argument("--work", type=Path, default=Path("work"))
        if name in ("run", "worker", "summarize"):
            p.add_argument("--output", type=Path, required=True)
        if name == "prepare":
            p.add_argument("--model", default="all")
        if name == "run":
            p.add_argument("--threads", type=int, nargs="+", choices=(1, 2, 4), default=[4])
            p.add_argument("--rounds", type=int, default=3)
        if name == "worker":
            p.add_argument("--model", required=True)
            p.add_argument("--precision", choices=("fp32", "int8"), required=True)
            p.add_argument("--threads", type=int, choices=(1, 2, 4), required=True)
            p.add_argument("--round", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        env = environment(); print(json.dumps(env, indent=2))
        if any(v is None for v in env["packages"].values()):
            raise SystemExit("missing dependencies; use the N100 environment instructions")
        import openvino as ov
        print("OpenVINO devices:", ov.Core().available_devices)
        data = manifest(args.bundle)
        for item in data["models"]:
            verify_file(args.bundle, item["fp32"])
        print("Six FP32 source hashes and evaluation/calibration metadata verified.")
    elif args.command == "prepare":
        if args.model != "all" and args.model not in {m["id"] for m in manifest(args.bundle)["models"]}:
            parser.error("unknown model")
        prepare(args.bundle, args.work, args.model)
    elif args.command == "worker":
        worker(args)
    elif args.command == "run":
        if args.rounds < 1 or len(set(args.threads)) != len(args.threads):
            parser.error("positive rounds and distinct thread settings required")
        run_matrix(args)
    else:
        summarize(args.output)


if __name__ == "__main__":
    main()
