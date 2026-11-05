"""Build and verify a source-only portable LAN demo; never copy model weights."""
import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = "hifi-mobiNet-n100-common97-2026-11-05"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def payload_files():
    relative = ["LICENSE", "THIRD_PARTY_NOTICES.md", "demo/n100_models.json",
                "demo/n100_constraints.txt", "demo/N100-README.vi.md",
                "demo/LENH-N100-LAN-DEMO.txt", "demo/SUA-LOI-01-THIEU-GIT.txt", "src/hifimobinet/__init__.py",
                "src/hifimobinet/frontend.py", "src/hifimobinet/n100_demo.py", "src/hifimobinet/n100_server.py",
                "demo/n100_web/index.html", "demo/n100_web/app.js", "demo/n100_web/style.css"]
    relative += ["src/hifimobinet/common97.py", "src/hifimobinet/ptq_numeric_guard.py",
                 "configs/quantization/piper-common97.json", "configs/quantization/piper-common97-regression.json",
                 "scripts/n100_demo/prepare_common97.py", "demo/LENH-COMMON97-N100.txt",
                 "scripts/evaluation/edge/openvino_suite.py", "scripts/evaluation/edge/onnx_reference.py",
                 "scripts/evaluation/onnx/common.py"]
    relative += [p.relative_to(ROOT).as_posix() for p in (ROOT / "scripts/n100_demo").glob("*.sh")]
    relative += [p.relative_to(ROOT).as_posix() for p in (ROOT / "vendor/banhmi-phonemize").rglob("*")
                 if p.is_file() and (p.suffix in (".py", ".toml", ".cmake", ".cpp", ".hpp", ".md", ".txt", ".in")
                                    or p.name == "CMakeLists.txt")
                 and not any(part in ("build", "__pycache__", ".git", ".venv") for part in p.parts)]
    return {name: (ROOT / name).read_bytes() for name in sorted(set(relative))}


def build(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / (NAME + ".zip")
    if archive.exists() or Path(str(archive) + ".sha256").exists():
        raise FileExistsError("Refusing to overwrite an existing transfer package")
    payload = payload_files()
    hashes = {name: digest(data) for name, data in payload.items()}
    payload["SHA256SUMS"] = "".join(f"{h}  {name}\n" for name, h in hashes.items()).encode()
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as handle:
        for name, data in payload.items():
            handle.writestr(NAME + "/" + name, data)
    with zipfile.ZipFile(archive) as handle, tempfile.TemporaryDirectory() as folder:
        assert handle.testzip() is None, "CRC failed"
        assert sorted(handle.namelist()) == sorted(NAME + "/" + n for n in payload)
        handle.extractall(folder)
        for name, data in payload.items():
            assert (Path(folder) / NAME / name).read_bytes() == data, name
        # Re-read sources to detect changes during packaging.
        assert payload_files() == {name: data for name, data in payload.items() if name != "SHA256SUMS"}
    archive_hash = digest(archive.read_bytes())
    Path(str(archive) + ".sha256").write_text(f"{archive_hash}  {archive.name}\n", encoding="ascii")
    (destination / "LENH-N100-LAN-DEMO.txt").write_bytes(payload["demo/LENH-N100-LAN-DEMO.txt"])
    (destination / "LENH-COMMON97-N100.txt").write_bytes(payload["demo/LENH-COMMON97-N100.txt"])
    result = {"zip": archive.name, "bytes": archive.stat().st_size, "sha256": archive_hash,
              "payload_files": len(hashes), "checks": {"crc": True, "file_list": True,
              "source_copy_match": True, "extract_and_internal_hashes": True},
              "model_weights_included": False, "n100_live_inference_tested": False,
              "payload_sha256": hashes}
    (destination / "package-verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return {k: v for k, v in result.items() if k != "payload_sha256"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    print(json.dumps(build(parser.parse_args().output), indent=2))
