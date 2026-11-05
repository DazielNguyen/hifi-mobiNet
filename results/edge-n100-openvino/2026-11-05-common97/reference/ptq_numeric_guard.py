"""Process-local numerical guard used by the validated common97 experiments.

Promoted from the preserved N100 diagnostic Recorder, without changing its
statistics hooks. No clipping/casting, warning suppression or library edits.
Run PTQ in a dedicated process because NumPy/warning hooks are process-wide.
"""
from __future__ import annotations

import datetime
import hashlib
import importlib
import json
import linecache
from pathlib import Path
import sys
import traceback
import warnings



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
