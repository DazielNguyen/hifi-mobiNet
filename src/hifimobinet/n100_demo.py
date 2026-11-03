"""LAN demo using the existing, hash-pinned N100 IRs. No export or PTQ."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata as metadata
import io
import json
import os
import platform
import socket
import threading
import time
import traceback
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "demo/n100_models.json"
PACKAGE = "hifi-mobiNet-n100-openvino-2026-10-31"
SCALES = [0.667, 1.0, 0.8]
CONFIG = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": 1,
          "INFERENCE_NUM_THREADS": 4, "INFERENCE_PRECISION_HINT": "f32",
          "EXECUTION_MODE_HINT": "ACCURACY"}
MAX_TEXT = 200


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, data):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


def bundle_path():
    return Path(os.environ.get("HIFIMOBINET_N100_BUNDLE",
                              str(Path.home() / "benchmarks" / PACKAGE / PACKAGE))).expanduser().resolve()


def runs_path():
    return Path(os.environ.get("HIFIMOBINET_N100_RUNS",
                              str(Path.home() / "benchmarks/n100-demo-runs"))).expanduser().resolve()


def verify_file(base, record):
    base = Path(base).resolve()
    path = (base / record["file"]).resolve()
    if not path.is_relative_to(base):
        raise ValueError("Artifact path escapes its bundle")
    if not path.is_file() or sha(path) != record["sha256"]:
        raise ValueError(f"Missing or changed file: {path}")
    return path


def host_info():
    if platform.system() != "Linux":
        raise RuntimeError("This server must run on the Linux N100; the PC only opens its browser.")
    cpu_names = sorted({line.split(":", 1)[1].strip()
                        for line in Path("/proc/cpuinfo").read_text().splitlines()
                        if line.startswith("model name")})
    if not cpu_names or any("N100" not in name.split() for name in cpu_names):
        raise RuntimeError(f"Expected Intel N100 CPU, got {cpu_names}")
    return {"hostname": socket.gethostname(), "cpu_names": cpu_names,
            "logical_cpus": os.cpu_count(), "platform": platform.platform(),
            "python": platform.python_version(), "pid": os.getpid(),
            "packages": {name: metadata.version(name) for name in
                         ("openvino", "nncf", "numpy", "onnx", "onnxruntime")}}


def check_frontend(rows):
    from hifimobinet.frontend import encode
    if len(rows) != 720 or [row["idx"] for row in rows] != list(range(720)):
        raise ValueError("Expected canonical Harvard-720 reference")
    mismatches = [row["idx"] for row in rows if encode(row["text"]) != row["phoneme_ids"]]
    if mismatches:
        raise ValueError(f"Banhmi frontend ID mismatch: {mismatches[:20]}")
    import banhmi_phonemize
    import banhmi_phonemize_cpp
    return {"matched": 720, "total": 720, "language": "en-us",
            "version": metadata.version("banhmi-phonemize"),
            "python_module_sha256": sha(banhmi_phonemize.__file__),
            "native_module_sha256": sha(banhmi_phonemize_cpp.__file__)}


def preflight(bundle, catalog=None):
    catalog = catalog or read(CATALOG)
    host = host_info()
    if host["packages"] != catalog["expected_packages"]:
        raise ValueError("N100 runtime versions differ from the verified measurement environment. "
                         "Restore the recorded versions; do not prepare new models for this demo.")
    verify_file(bundle, catalog["manifest"])
    rows = read(verify_file(bundle, catalog["harvard"]))
    for item in catalog["models"]:
        verify_file(bundle, item["preparation"])
        for precision in ("fp32", "int8"):
            for artifact in item["artifacts"][precision]:
                verify_file(bundle, artifact)
    return {"status": "ready", "checked_utc": utc(), "host": host,
            "catalog_sha256": sha(CATALOG), "model_count": len(catalog["models"]),
            "frontend": check_frontend(rows), "bundle": str(bundle),
            "inference_performed": False}


def precision_evidence(compiled, precision):
    nodes = []
    for node in compiled.get_runtime_model().get_ops():
        info = node.get_rt_info()
        row = {"name": node.get_friendly_name()}
        for key in ("layerType", "runtimePrecision", "execType", "originalLayersNames"):
            if key in info:
                row[key] = str(getattr(info[key], "value", info[key]))
        nodes.append(row)
    weighted = [n for n in nodes if n.get("layerType") in
                ("Convolution", "GroupConvolution", "FullyConnected", "Deconvolution")]
    kernels = [n for n in weighted if any(token in
               (n.get("runtimePrecision", "") + " " + n.get("execType", "")).lower()
               for token in ("i8", "u8", "int8"))]
    if not weighted:
        raise ValueError("Cannot verify executed weighted kernels")
    if precision == "fp32":
        if any(n.get("runtimePrecision", "").lower() not in ("f32", "fp32", "float32")
               for n in weighted):
            raise ValueError("FP32 execution precision was not verified")
    elif precision == "int8":
        if not kernels:
            raise ValueError("No INT8 weighted execution kernel")
        for node in kernels:
            name = node["name"] + " " + node.get("originalLayersNames", "")
            if "/dec/" in name or not any(scope in name for scope in ("/enc_p/", "/dp/", "/flow/")):
                raise ValueError("INT8 execution outside the validated selective scopes")
    else:
        raise ValueError("Unknown precision")
    return {"weighted_count": len(weighted), "int8_weighted_count": len(kernels), "nodes": nodes}


def to_wav(output, sample_rate):
    import numpy as np
    output = np.asarray(output)
    if output.ndim != 3 or output.shape[:2] != (1, 1) or output.shape[-1] < 1:
        raise ValueError("Expected mono waveform [1, 1, samples]")
    # This is a post-inference playback limit, not an inference memory/time limit.
    if output.shape[-1] > sample_rate * 60:
        raise ValueError("Generated audio exceeds the demo's 60-second playback limit")
    if not np.isfinite(output).all() or float(np.std(output, dtype=np.float64)) <= 1e-6:
        raise ValueError("Generated waveform is nonfinite or silent")
    samples = output.reshape(-1)
    peak = float(np.max(np.abs(samples)))
    pcm = np.clip(samples * (32767 / max(0.01, peak)), -32767, 32767).astype("<i2")
    stream = io.BytesIO()
    with wave.open(stream, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return stream.getvalue(), {"audio_seconds": len(samples) / sample_rate,
                              "samples": len(samples), "raw_peak_abs": peak,
                              "wav_encoding": "mono PCM16, peak scaled"}


class Engine:
    """One compiled model and one request at a time, shared across browser sessions."""

    def __init__(self, bundle, output):
        self.bundle = Path(bundle).resolve()
        self.catalog = read(CATALOG)
        self.preflight = preflight(self.bundle, self.catalog)
        self.models = {m["id"]: m for m in self.catalog["models"]}
        import openvino as ov
        self.ov = ov
        self.core = ov.Core()
        self.preflight["openvino_cpu"] = self.core.get_property("CPU", "FULL_DEVICE_NAME")
        self.session = Path(output) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                      + "-" + uuid.uuid4().hex[:8])
        self.session.mkdir(parents=True, exist_ok=False)
        write_new(self.session / "server.json", {**self.preflight,
                  "source_sha256": sha(__file__), "config": CONFIG, "scales": SCALES})
        self.lock = threading.Lock()
        self.active = None
        self.compiled = None
        self.request = None
        self.execution = None

    def _load(self, model_id, precision):
        if self.active == (model_id, precision):
            return 0.0, False
        self.active = self.request = self.compiled = self.execution = None
        gc.collect()  # Drop the previous model before loading the next one.
        item = self.models[model_id]
        paths = [verify_file(self.bundle, f) for f in item["artifacts"][precision]]
        xml = next(p for p in paths if p.suffix == ".xml")
        weights = next(p for p in paths if p.suffix == ".bin")
        start = time.perf_counter()
        model = self.core.read_model(str(xml), str(weights))
        compiled = self.core.compile_model(model, "CPU", CONFIG)
        elapsed = time.perf_counter() - start
        devices = list(compiled.get_property("EXECUTION_DEVICES"))
        if devices != ["CPU"]:
            raise ValueError(f"Unexpected execution device: {devices}")
        if {p.get_any_name() for p in compiled.inputs} != {"input", "input_lengths", "scales"} or len(compiled.outputs) != 1:
            raise ValueError("Unexpected model I/O contract")
        self.compiled = compiled
        self.request = compiled.create_infer_request()
        self.active = (model_id, precision)
        self.execution = {"devices": devices}
        return elapsed, True

    def synthesize(self, text, model_id, precision):
        if model_id not in self.models or precision not in ("fp32", "int8"):
            raise ValueError("Choose a listed model and precision")
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
            raise ValueError(f"Enter 1–{MAX_TEXT} characters of English text")
        import numpy as np
        from hifimobinet.frontend import encode
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("N100 is processing another request. Try again when it finishes.")
        request_id = uuid.uuid4().hex
        item = self.models[model_id]
        record = {"request_id": request_id, "started_utc": utc(), "status": "running",
                  "text": text, "model_id": model_id, "precision": precision,
                  "sample_rate": item["sample_rate"], "host": self.preflight["host"],
                  "artifacts": item["artifacts"][precision], "checkpoint": item["checkpoint"],
                  "preparation": item["preparation"], "config": CONFIG, "scales": SCALES,
                  "quality_status": item["quality_status"], "benchmark": False,
                  "rng": "native graph RNG; waveform/duration may vary between requests",
                  "timed_region": "synchronous InferRequest.infer only; no warmup; excludes frontend, load/compile, output copy, WAV and network"}
        started = time.perf_counter()
        try:
            front_start = time.perf_counter()
            ids = encode(text)
            record["frontend_s"] = time.perf_counter() - front_start
            record["phoneme_ids"] = ids
            record["load_compile_s"], record["first_call_after_load"] = self._load(model_id, precision)
            feeds = {"input": np.asarray([ids], dtype=np.int64),
                     "input_lengths": np.asarray([len(ids)], dtype=np.int64),
                     "scales": np.asarray(SCALES, dtype=np.float32)}
            for name, value in feeds.items():
                self.request.set_tensor(name, self.ov.Tensor(value))
            infer_start = time.perf_counter_ns()
            self.request.infer(share_outputs=True)
            record["inference_s"] = (time.perf_counter_ns() - infer_start) / 1e9
            output = np.array(self.request.get_output_tensor(0).data, copy=True)
            # Dynamic graphs have actual execution metadata after the first inference.
            if "precision" not in self.execution:
                evidence = precision_evidence(self.compiled, precision)
                runtime_file = f"runtime-{request_id}.json"
                write_new(self.session / runtime_file, evidence)
                self.execution["precision"] = {k: v for k, v in evidence.items() if k != "nodes"}
                self.execution["runtime_record"] = runtime_file
            record["execution"] = self.execution
            wav, audio = to_wav(output, item["sample_rate"])
            record.update(audio)
            record["rtf"] = record["inference_s"] / record["audio_seconds"]
            record["server_generation_s"] = time.perf_counter() - started
            record["wav_sha256"] = hashlib.sha256(wav).hexdigest()
            record["wav_file"] = request_id + ".wav"
            with (self.session / record["wav_file"]).open("xb") as handle:
                handle.write(wav)
            record.update(status="generated", finished_utc=utc())
            write_new(self.session / (request_id + ".json"), record)
            return wav, record
        except Exception:
            record.update(status="failed", finished_utc=utc(), error=traceback.format_exc())
            write_new(self.session / (request_id + "-failed.json"), record)
            self.active = self.request = self.compiled = self.execution = None
            raise
        finally:
            self.lock.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("doctor", "frontend"))
    parser.add_argument("--bundle", type=Path, default=bundle_path())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "frontend":
            result = check_frontend(read(verify_file(args.bundle, read(CATALOG)["harvard"])))
        else:
            result = preflight(args.bundle)
        if args.output:
            write_new(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as error:
        if args.output:
            write_new(args.output, {"status": "failed", "error": traceback.format_exc(), "utc": utc()})
        parser.exit(1, f"CHECK FAILED: {error}\n")


if __name__ == "__main__":
    main()
