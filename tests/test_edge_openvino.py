"""Offline tests: no runtime install, model execution, or training required."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "scripts/evaluation/edge" if (ROOT / "scripts/evaluation/edge").exists() else ROOT
sys.path.insert(0, str(CODE))
import openvino_suite as suite
import build_bundle as builder


class EdgeProtocolTests(unittest.TestCase):
    def test_rtf_uses_each_actual_audio_duration(self):
        result = suite.summary([{"latency_s": 1, "audio_seconds": 2}, {"latency_s": 9, "audio_seconds": 3}])
        self.assertEqual(result["mean_rtf"], 1.75)
        self.assertEqual(result["duration_weighted_rtf"], 2)
        self.assertEqual(result["p95_rtf"], 3)
        for invalid in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                suite.summary([{"latency_s": invalid, "audio_seconds": 1}])

    def test_no_silent_missing_hash_or_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x").write_bytes(b"original")
            item = {"file": "x", "sha256": suite.sha(root / "x")}
            self.assertEqual(suite.verify_file(root, item), root / "x")
            (root / "x").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                suite.verify_file(root, item)
            with self.assertRaises(ValueError):
                suite.verify_file(root, {"file": "../outside", "sha256": "0"})
            with self.assertRaises(ValueError):
                suite.verify_file(root, {"file": "missing", "sha256": "0"})

    def test_schedule_balances_all_six_precisions_threads_rounds(self):
        models = [{"id": str(i)} for i in range(6)]
        jobs = suite.schedule(models, [1, 2, 4], 3)
        self.assertEqual(len(jobs), 108)
        self.assertEqual(len(set(jobs)), 108)
        self.assertEqual([j[1:] for j in jobs[:36]], list(reversed([j[1:] for j in jobs[36:72]])))

    def test_precision_guard_rejects_fake_int8_and_decoder_int8(self):
        conv = {"layerType": "Convolution", "name": "/enc_p/Conv", "runtimePrecision": "f32"}
        suite.check_runtime_precision([conv], "fp32")
        with self.assertRaises(ValueError):
            suite.check_runtime_precision([conv], "int8")
        quant = {**conv, "runtimePrecision": "u8", "execType": "jit_i8"}
        suite.check_runtime_precision([quant], "int8")
        with self.assertRaises(ValueError):
            suite.check_runtime_precision([quant], "fp32")
        with self.assertRaises(ValueError):
            suite.check_runtime_precision([{**quant, "name": "/dec/Conv"}], "int8")

    def test_n100_guard_rejects_other_machine(self):
        with patch.object(suite.platform, "system", return_value="Windows"):
            with self.assertRaisesRegex(RuntimeError, "N100"):
                suite.environment()

    def test_calibration_cannot_contain_test_rows(self):
        reference = ROOT / "scripts/evaluation/onnx/common.py"
        if not reference.exists():
            reference = ROOT / "reference/scripts/evaluation/onnx/common.py"
        spec = importlib.util.spec_from_file_location("edge_calibration_test", reference)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        row = {"id": "example", "audio_norm_path": "test.pt", "phoneme_ids": [1, 2]}
        result = module.check_calibration([row], {"train": [], "val": [], "test": ["test.pt"]}, [row])
        self.assertFalse(result["passed"])
        self.assertEqual(result["test"], 1)

    def test_summary_requires_full_schedule_and_all_720_rows(self):
        jobs = suite.schedule([{"id": str(i)} for i in range(6)], [4], 1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite.write(root / "schedule.json", jobs[:-1])
            with self.assertRaisesRegex(ValueError, "schedule"):
                suite.summarize(root)
            suite.write(root / "schedule.json", jobs)
            for rnd, model, precision, threads in jobs:
                cell = root / f"r{rnd}_{model}_{precision}_t{threads}"
                cell.mkdir()
                suite.write(cell / "result.json", {"status": "measured", "round": rnd, "model_id": model,
                    "precision": precision, "threads": threads, "manifest_sha256": "same", "script_sha256": "same",
                    "environment_start": {"packages": {"openvino": "fixture"}, "cpu_names": ["Intel N100"]},
                    "preparation_sha256": model, "peak_process_rss_bytes": 123})
                (cell / "utterances.jsonl").write_text("".join(json.dumps({"idx": i, "latency_s": 1,
                    "audio_seconds": 2}) + "\n" for i in range(720)))
            suite.summarize(root)
            report = suite.read(root / "summary.json")
            self.assertEqual(len(report["tables"]["fp32"]), 6)
            self.assertEqual(report["tables"]["int8"][0]["duration_weighted_rtf"], 0.5)
            (cell / "utterances.jsonl").write_text('{"idx":0,"latency_s":1,"audio_seconds":2}\n')
            with self.assertRaisesRegex(ValueError, "incomplete cell"):
                suite.summarize(root)

    def test_copy_hash_and_zip_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.bin"
            source.write_bytes(bytes(range(256)) * 100)
            payload = root / "kit"
            payload.mkdir()
            with self.assertRaises(ValueError):
                builder.copy_checked(source, payload / "bad.bin", "0" * 64)
            builder.copy_checked(source, payload / "model.bin", suite.sha(source))
            result = builder.archive_and_verify(payload, root)
            self.assertEqual(result["status"], "passed")
            self.assertEqual((root / "kit.zip.sha256").read_text().split()[1], "kit.zip")
            self.assertEqual(suite.sha(root / "kit.zip"), result["sha256"])
            self.assertEqual(result["runtime_on_N100"], "not run")


if __name__ == "__main__":
    unittest.main()
