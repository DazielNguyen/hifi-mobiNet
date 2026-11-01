#!/usr/bin/env python3
"""Trace N100/NNCF calibration numerics without modifying the original bundle.

Default: prepare ONLY piper-seqir-no-vits2-cpn into a new timestamped directory.
Instrumentation is process-local and limited to nncf.quantize. No casts, clipping,
library-file edits, model-weight edits, uploads, or benchmark runs are performed.
--self-test uses synthetic NumPy arrays only and does not require OpenVINO.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import importlib
import importlib.util
import json
import linecache
from pathlib import Path
import sys
import tempfile
import traceback
import warnings

DEFAULT_MODEL = "piper-seqir-no-vits2-cpn"


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


class NumericDiagnosticStop(RuntimeError):
    pass


def numpy_methods():
    try:
        return importlib.import_module("numpy._core._methods")
    except ModuleNotFoundError:
        return importlib.import_module("numpy.core._methods")


class Recorder:
    def __init__(self, folder, np):
        self.folder = Path(folder)
        self.np = np
        self.event = None
        self.capture_errors = []

    def array_summary(self, value):
        np = self.np
        if not isinstance(value, np.ndarray):
            if type(value).__module__.startswith("nncf.tensor"):
                value = getattr(value, "data", None)
        if not isinstance(value, np.ndarray):
            return None
        result = {"shape": list(value.shape), "dtype": str(value.dtype),
                  "size": int(value.size), "sample_policy": "first at most 4096 elements; not a whole-tensor check"}
        if value.dtype.kind not in "fiu b".replace(" ", ""):
            return result
        with np.errstate(all="ignore"):
            sample = np.asarray(value.flat[:4096], dtype=np.float64)
            finite = np.isfinite(sample)
            result.update(sample_count=int(sample.size), sample_finite_count=int(finite.sum()),
                          sample_nan_count=int(np.isnan(sample).sum()),
                          sample_posinf_count=int(np.isposinf(sample).sum()),
                          sample_neginf_count=int(np.isneginf(sample).sum()))
            if finite.any():
                vals = sample[finite]
                result["sample_finite_min"] = float(vals.min())
                result["sample_finite_max"] = float(vals.max())
        return result

    def frame_summary(self, frame, lineno):
        path = Path(frame.f_code.co_filename)
        result = {"file": str(path), "line": lineno, "function": frame.f_code.co_name,
                  "arrays": {}, "context": {},
                  "source_context": [
                      {"line": n, "text": linecache.getline(str(path), n).rstrip()}
                      for n in range(max(1, lineno - 2), lineno + 3)
                  ]}
        if path.is_file() and path.stat().st_size < 2_000_000:
            result["source_sha256"] = sha(path)
        # Do not serialize arbitrary objects, utterance text, or complete arrays.
        for name, value in list(frame.f_locals.items()):
            if len(result["arrays"]) < 6:
                try:
                    summary = self.array_summary(value)
                    if summary is not None:
                        result["arrays"][name] = summary
                except Exception as exc:
                    result["arrays"][name] = {"inspection_error": type(exc).__name__}
            if name in {"axis", "axes", "dtype", "name", "node_name", "layer_name", "target_node_name"}:
                if value is None or isinstance(value, (str, int, bool)):
                    result["context"][name] = str(value)[:240]
                elif name in {"axis", "axes"} and isinstance(value, (tuple, list)):
                    result["context"][name] = str(value)[:240]
            if name in {"self", "node", "target_node", "collector"}:
                result["context"][name + "_type"] = type(value).__module__ + "." + type(value).__name__
                if hasattr(value, "__dict__"):
                    for key in ("name", "node_name", "layer_name", "target_node_name"):
                        item = vars(value).get(key)
                        if isinstance(item, (str, int)):
                            result["context"][name + "." + key] = str(item)[:240]
        return result

    def capture(self, kind, message, frame, tb=None, warning_location=None, numeric_policy=None):
        if self.event is not None:
            return
        # Set immediately so an inspection failure cannot recurse or erase the trigger.
        self.event = {"kind": kind, "message": str(message), "utc": utc(),
                      "warning_location": warning_location, "numeric_policy": numeric_policy, "frames": []}
        current = frame
        try:
            self.event["call_stack"] = "".join(traceback.format_stack(frame))
            pairs = []
            while current is not None:
                pairs.append((current, current.f_lineno))
                current = current.f_back
            pairs.reverse()
            if tb is not None:
                self.event["exception_traceback"] = "".join(traceback.format_tb(tb))
                pairs.extend(traceback.walk_tb(tb))
            # Include the closest frames plus call-stack text for the complete chain.
            for active, lineno in pairs[-18:]:
                if Path(active.f_code.co_filename).resolve() == Path(__file__).resolve():
                    continue
                try:
                    self.event["frames"].append(self.frame_summary(active, lineno))
                except Exception as exc:
                    self.capture_errors.append(type(exc).__name__ + ": " + str(exc))
            self.event["inspection_errors"] = self.capture_errors
        finally:
            del current
            write_new(self.folder / "numeric-event.json", self.event)

    def guarded_call(self, function, *args, **kwargs):
        np = self.np
        methods = numpy_methods()
        old_showwarning = warnings.showwarning
        originals = {name: getattr(methods, name) for name in ("_mean", "_var", "_std")}

        def strict_method(name, original):
            def wrapped(*values, **options):
                previous_errstate = np.geterr().copy()
                try:
                    # This acts inside an outer library errstate/filter scope.
                    with np.errstate(over="raise", invalid="raise", divide="raise"):
                        return original(*values, **options)
                except FloatingPointError as exc:
                    self.capture("numpy_" + name, exc, sys._getframe(), exc.__traceback__,
                                 numeric_policy={"previous": previous_errstate,
                                                 "diagnostic": {"over": "raise", "invalid": "raise", "divide": "raise"}})
                    raise NumericDiagnosticStop("Captured NumPy numerical issue in " + name) from exc
            return wrapped

        def showwarning(message, category, filename, lineno, file=None, line=None):
            if issubclass(category, RuntimeWarning) and any(
                word in str(message).lower() for word in ("overflow", "invalid value", "divide by zero")
            ):
                self.capture("runtime_warning", message, sys._getframe().f_back,
                             warning_location={"file": filename, "line": lineno})
                raise NumericDiagnosticStop("Captured RuntimeWarning: " + str(message))
            return old_showwarning(message, category, filename, lineno, file=file, line=line)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("always", RuntimeWarning)
                warnings.showwarning = showwarning
                for name, original in originals.items():
                    setattr(methods, name, strict_method(name, original))
                result = function(*args, **kwargs)
                # Even a library that catches the diagnostic exception cannot qualify this run.
                if self.event is not None:
                    raise NumericDiagnosticStop("Numerical issue was captured inside the quantization call")
                return result
        finally:
            warnings.showwarning = old_showwarning
            for name, original in originals.items():
                setattr(methods, name, original)


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()

    def isatty(self):
        return False


def self_test():
    import numpy as np
    methods = numpy_methods()
    old_mean = methods._mean
    old_showwarning = warnings.showwarning
    old_err = np.geterr().copy()
    with tempfile.TemporaryDirectory(prefix="n100-numerics-selftest-") as temporary:
        root = Path(temporary)
        for name in ("mean", "warning", "caught", "clean"):
            (root / name).mkdir()

        def overflow_with_library_overrides():
            with warnings.catch_warnings():
                warnings.simplefilter("always", RuntimeWarning)
                with np.errstate(all="warn"):
                    arr = np.full(4, np.finfo(np.float32).max, dtype=np.float32)
                    return np.mean(arr)

        r = Recorder(root / "mean", np)
        try:
            r.guarded_call(overflow_with_library_overrides)
            raise AssertionError("overflow was not intercepted")
        except NumericDiagnosticStop:
            assert r.event["kind"] == "numpy__mean"
            assert any(f["arrays"] for f in r.event["frames"])
            json.loads((root / "mean" / "numeric-event.json").read_text())

        def invalid_with_filter_reset():
            warnings.simplefilter("always", RuntimeWarning)
            arr = np.array([np.inf], dtype=np.float32)
            with np.errstate(invalid="warn"):
                return arr - arr

        r = Recorder(root / "warning", np)
        try:
            r.guarded_call(invalid_with_filter_reset)
            raise AssertionError("RuntimeWarning was not intercepted")
        except NumericDiagnosticStop:
            assert r.event["kind"] == "runtime_warning"

        def swallowing_library():
            try:
                invalid_with_filter_reset()
            except RuntimeError:
                return "swallowed"

        r = Recorder(root / "caught", np)
        try:
            r.guarded_call(swallowing_library)
            raise AssertionError("caught diagnostic was treated as success")
        except NumericDiagnosticStop:
            assert r.event is not None

        r = Recorder(root / "clean", np)
        assert r.guarded_call(lambda: np.mean(np.array([1, 2, 3], dtype=np.float32))) == 2
        assert r.event is None
        assert not (root / "clean" / "numeric-event.json").exists()

    assert methods._mean is old_mean
    assert warnings.showwarning is old_showwarning
    assert np.geterr() == old_err
    print("SELF-TEST OK: overflow, overridden warning filters, swallowed exceptions, clean path, hook restoration.")
    print("Synthetic NumPy tests only; no OpenVINO model or calibration was run.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()

    bundle = args.bundle.resolve()
    suite_path = bundle / "openvino_suite.py"
    if not suite_path.is_file():
        parser.error("Copy this script next to openvino_suite.py, or pass --bundle <that folder>.")
    sys.path.insert(0, str(bundle))
    spec = importlib.util.spec_from_file_location("_n100_diagnostic_original_suite", suite_path)
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    env = suite.environment()  # Enforce Linux + N100 before OpenVINO imports or inference.
    data = suite.manifest(bundle)
    model = next((m for m in data["models"] if m["id"] == args.model), None)
    if model is None:
        parser.error("model ID is not in models.json")
    suite.verify_file(bundle, model["fp32"])

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = bundle / ("diagnostic-n100-" + stamp)
    output.mkdir(exist_ok=False)
    report = {"started_utc": utc(), "status": "starting", "model": args.model,
              "environment": env, "script_sha256": sha(__file__),
              "original_suite_sha256": sha(suite_path),
              "models_manifest_sha256": sha(bundle / "models.json"),
              "source_onnx_sha256": sha(bundle / model["fp32"]["file"]),
              "interpretation": "diagnostic only; never certifies calibration, quality or RTF",
              "instrumentation": "process-local NumPy _mean/_var/_std errstate and warnings.showwarning only inside nncf.quantize"}
    write_new(output / "session-start.json", report)
    recorder = None
    exit_code = 2
    with (output / "diagnostic.log").open("x", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print("Diagnostic directory:", output, flush=True)
            try:
                import numpy as np
                import nncf
                recorder = Recorder(output, np)
                quantize = nncf.quantize

                def guarded_quantize(*values, **options):
                    print("Entering instrumented NNCF quantization.", flush=True)
                    return recorder.guarded_call(quantize, *values, **options)

                nncf.quantize = guarded_quantize
                try:
                    suite.prepare(bundle, output / "probe-work", args.model)
                finally:
                    nncf.quantize = quantize
                report["status"] = "completed_without_intercepted_numerics"
                print("No numerical issue was intercepted on this attempt; this is NOT a calibration-quality pass.")
            except NumericDiagnosticStop:
                report["status"] = "numeric_issue_captured"
                report["traceback"] = traceback.format_exc()
                traceback.print_exc()
                print("NUMERIC ISSUE CAPTURED. Original work-n100 is unchanged.")
            except BaseException:
                report["status"] = "other_failure"
                report["traceback"] = traceback.format_exc()
                traceback.print_exc()
            finally:
                if recorder and recorder.event is not None:
                    report["event_file"] = "numeric-event.json"
                report["finished_utc"] = utc()
                write_new(output / "diagnostic-report.json", report)
                print("Return diagnostic.log, diagnostic-report.json, and numeric-event.json if present.")
                print("STATUS:", report["status"], flush=True)

    # Return bundle contains diagnostic metadata only, no models, weights or WAVs.
    import zipfile
    archive = output / "diagnostic-report.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name in ("session-start.json", "diagnostic-report.json", "diagnostic.log", "numeric-event.json"):
            file = output / name
            if file.is_file():
                z.write(file, arcname=name)
    (output / "diagnostic-report.zip.sha256").write_text(
        sha(archive) + "  " + archive.name + "\n", encoding="utf-8")
    print("Report ZIP:", archive)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

