"""Derive FP32 defaults from preserved evidence; never activate legacy INT8.

The common97 migration builds a separate local catalog after fresh validation.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "results/edge-n100-openvino/2026-11-01/evidence"
WORK = "work-n100-clean-20261101T075458Z"


def identity(path, relative):
    return {"file": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def build_catalog():
    manifest = json.loads((EVIDENCE / "models.json").read_text(encoding="utf-8"))
    items = []
    expected = None
    for model in manifest["models"]:
        relative = f"{WORK}/{model['id']}/preparation.json"
        prep = json.loads((EVIDENCE / relative).read_text(encoding="utf-8"))
        assert prep["status"] == "prepared" and prep["model"] == model
        assert prep["script_sha256"] == "fb20086c7907a7730fd3e3446544c6c816809c88b2164e640e1b10882af902cf"
        assert prep["calibration"]["passed"] and all(p["passed"] for p in prep["fp32_parity"])
        packages = prep["environment"]["packages"]
        assert expected is None or packages == expected
        expected = packages
        items.append({"id": model["id"], "label": model["label"], "group": model["group"],
                      "sample_rate": model["sample_rate"], "checkpoint": model["checkpoint"],
                      "quality_status": "Historical FP32; legacy INT8 disabled. Run common97 preparation for the three Piper models.",
                      "preparation": identity(EVIDENCE / relative, relative),
                      "artifacts": {precision: [{"file": f"{WORK}/{model['id']}/{f['file']}",
                                    "sha256": f["sha256"]} for f in prep["artifacts"][precision]["files"]]
                                    for precision in ("fp32",)}})
    assert len(items) == 6
    return {"schema_version": 2, "source": "results/edge-n100-openvino/2026-11-01/evidence",
            "expected_packages": expected,
            "manifest": identity(EVIDENCE / "models.json", "models.json"),
            "harvard": identity(EVIDENCE / "metadata/harvard720.json", "metadata/harvard720.json"),
            "models": items}


if __name__ == "__main__":
    (ROOT / "demo/n100_models.json").write_text(
        json.dumps(build_catalog(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
