"""Manifest-only model access. No checkpoint deserialization."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


class ModelUnavailable(RuntimeError):
    """A selected, known model cannot safely be used in this installation."""


def repository_root() -> Path:
    return Path(os.environ.get("HIFIMOBINET_HOME", Path(__file__).resolve().parents[2])).resolve()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def within(root: Path, relative: str) -> Path:
    """Reject absolute paths, traversal and symlinks that escape an allowed root."""
    candidate = (root / relative).resolve()
    if Path(relative).is_absolute() or not candidate.is_relative_to(root.resolve()):
        raise ValueError("Manifest path escapes its allowed directory")
    return candidate


def manifest(root: Path | None = None) -> dict:
    root = root or repository_root()
    data = json.loads((root / "models/manifest.json").read_text(encoding="utf-8"))
    ids = [m["id"] for m in data["models"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate model IDs in manifest")
    return data


def model_record(model_id: str, root: Path | None = None) -> dict:
    for record in manifest(root)["models"]:
        if record["id"] == model_id:
            return record
    raise ModelUnavailable("Unknown model ID; choose an ID listed by the models command")


def model_path(record: dict, root: Path | None = None, verify: bool = True) -> Path:
    root = root or repository_root()
    storage = Path(os.environ.get("HIFIMOBINET_MODEL_DIR", root / "models/weights")).resolve()
    artifact = record["artifact"]
    if artifact["format"] != "onnx" or artifact["precision"] != "FP32":
        raise ModelUnavailable("Only the identified FP32 ONNX artifacts are supported")
    path = within(storage, artifact["filename"])
    if not path.is_file():
        raise ModelUnavailable("Weights unavailable. See models/README.md; no model is substituted.")
    if path.stat().st_size != artifact["bytes"]:
        raise ModelUnavailable("Model file size differs from the manifest")
    if verify and sha256(path) != artifact["sha256"]:
        raise ModelUnavailable("Model checksum mismatch; inference refused")
    return path


def runtime_status(record: dict, root: Path | None = None) -> tuple[bool, str]:
    if not record.get("local_functional_validation", {}).get("tts_passed", False):
        return False, "TTS not yet functionally validated in this repository"
    try:
        model_path(record, root)
        from .frontend import require_frontend
        require_frontend()
        import onnxruntime  # noqa: F401
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        return False, str(error)
    return True, "Available; functional validation only, not a benchmark"
