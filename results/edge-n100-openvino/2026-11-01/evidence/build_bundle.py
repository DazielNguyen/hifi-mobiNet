"""Assemble a local N100 kit using existing FP32 ONNX files; stdlib only.

Run under WSL to read the actual training clone. Never loads a checkpoint,
executes a model, changes a source file, or contacts a remote service.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

from openvino_suite import read, sha, utc, write

METADATA = {
    "harvard": ("results/manifests/harvard720.json", "bdaf43e33d3a435eebb95d36d350d24d4406fc65affa16ee69c58b623402e272"),
    "split": ("results/manifests/canonical_split.json", "e678cf436075a10bb68c9a1b435f33c4a40ff1f095b7b78ac12f546c5f12ae9d"),
    "calibration": ("docs/evidence/claim-evidence-2026-10-07/source-records/calibration/calibration60.candidate.jsonl",
                    "e4de2ad4a2ed68b917cccb5149d151e004b775e4cf7d35f4d6d9a11bd7c40f53"),
}
DATASET_SHA = "f4a72ae0ed238dfdb3ca995b6678dfe33fe3b59297331d6551a4d6960102295b"
NEW_IDS = ("piper-no-vits2-cpn", "piper-parir-no-vits2-cpn", "piper-seqir-no-vits2-cpn")


def copy_checked(source, target, expected=None):
    before = sha(source)
    if expected and before != expected:
        raise ValueError(f"source hash mismatch: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    shutil.copyfile(source, target)
    if sha(target) != before or sha(source) != before:
        raise ValueError(f"source/copy hash mismatch or changing source: {source}")
    return before


def archive_and_verify(bundle, output):
    paths = sorted(p for p in bundle.rglob("*") if p.is_file())
    if any("__pycache__" in p.parts or p.suffix in (".pyc", ".ckpt", ".pt") for p in paths):
        raise ValueError("cache/checkpoint files must not enter this deployment kit")
    sums = {p.relative_to(bundle).as_posix(): sha(p) for p in paths}
    (bundle / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in sums.items()), encoding="utf-8")
    paths.append(bundle / "SHA256SUMS")
    archive = output / f"{bundle.name}.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for p in paths:
            z.write(p, f"{bundle.name}/{p.relative_to(bundle).as_posix()}")
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP CRC failed")
        expected = {f"{bundle.name}/{p.relative_to(bundle).as_posix()}" for p in paths}
        if set(z.namelist()) != expected or len(z.namelist()) != len(expected):
            raise ValueError("ZIP file list differs or has duplicate entries")
        with tempfile.TemporaryDirectory(prefix="extract-check-", dir=output) as temporary:
            z.extractall(temporary)  # all entries were constructed above from local relative paths
            extracted = Path(temporary) / bundle.name
            for p in paths:
                rel = p.relative_to(bundle)
                if sha(extracted / rel) != sha(p):
                    raise ValueError(f"extracted hash mismatch: {rel}")
            extracted_sums = (extracted / "SHA256SUMS").read_text(encoding="utf-8")
            for line in extracted_sums.splitlines():
                digest, name = line.split("  ", 1)
                if sha(extracted / name) != digest:
                    raise ValueError(f"internal checksum mismatch: {name}")
    digest = sha(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    report = {"status": "passed", "utc": utc(), "archive": archive.name, "bytes": archive.stat().st_size,
              "sha256": digest, "files": len(paths), "crc": "passed", "file_list": "passed",
              "extract_and_all_internal_hashes": "passed", "runtime_on_N100": "not run"}
    write(output / "package-verification.json", report)
    return report


def build(args):
    repo = args.repo.resolve()
    output = args.output.resolve()
    if output == repo or repo in output.parents:
        raise ValueError("put private model bundles outside the repository")
    output.mkdir(parents=True, exist_ok=True)
    bundle = output / "hifi-mobiNet-n100-openvino-2026-10-31"
    bundle.mkdir()  # never overwrite a previous package
    inventory = []

    def include(source, name, association, expected=None):
        digest = copy_checked(source, bundle / name, expected)
        inventory.append({"file": name, "sha256": digest, "bytes": (bundle / name).stat().st_size,
                          "association": association, "source": str(source), "source_copy_hashes": "passed"})
        return {"file": name, "sha256": digest}

    for filename in ("openvino_suite.py", "onnx_reference.py", "build_bundle.py", "requirements.txt", "README.vi.md"):
        include(repo / "scripts/evaluation/edge" / filename, filename, "N100 toolchain")
    include(repo / "scripts/evaluation/onnx/common.py", "reference/scripts/evaluation/onnx/common.py", "unchanged parity/calibration helpers")
    include(repo / "tests/test_edge_openvino.py", "tests/test_edge_openvino.py", "portable tests, no model execution")
    include(repo / "models/manifest.json", "reference/models-manifest.json", "historical FP32 provenance")
    include(repo / "docs/evaluation-onnx-ptq.md", "reference/evaluation-onnx-ptq.md", "historical protocol context, not N100 results")
    include(repo / "THIRD_PARTY_NOTICES.md", "reference/THIRD_PARTY_NOTICES.md", "source notices")
    metadata = {k: include(repo / path, "metadata/" + Path(path).name, "shared " + k, digest)
                for k, (path, digest) in METADATA.items()}
    cal = [json.loads(s) for s in (bundle / metadata["calibration"]["file"]).read_text().splitlines()]
    if sha(args.dataset_jsonl) != DATASET_SHA:
        raise ValueError("training dataset manifest hash changed")
    selected = {r["audio_norm_path"] for r in cal}
    dataset = [json.loads(s) for s in args.dataset_jsonl.read_text().splitlines() if s.strip()]
    subset = [r for r in dataset if r["audio_norm_path"] in selected]
    if len(subset) != 60:
        raise ValueError("expected exactly 60 source dataset rows")
    spec = importlib.util.spec_from_file_location("calibration_reference", repo / "scripts/evaluation/onnx/common.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    check = ref.check_calibration(cal, read(bundle / metadata["split"]["file"]), subset)
    if not check["passed"] or check["n"] != 60:
        raise ValueError(f"calibration membership/phoneme check failed: {check}")
    subset_path = bundle / "metadata/calibration_dataset_rows.jsonl"
    subset_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in subset), encoding="utf-8")
    metadata["calibration_dataset_rows"] = {"file": "metadata/calibration_dataset_rows.jsonl", "sha256": sha(subset_path),
                                           "derived_from_dataset_sha256": DATASET_SHA, "source_rows": len(dataset)}
    if sha(args.dataset_jsonl) != DATASET_SHA:
        raise ValueError("dataset manifest changed during extraction")
    models = []
    historical = read(repo / "models/manifest.json")["models"]
    for item in historical:
        if item["id"] not in ("baseline-resblock2", "parallel-ir", "sequential-ir"):
            continue
        artifact = item["artifact"]
        model = {"id": item["id"], "label": item["label"], "group": "historical Q05 FP32",
                 "sample_rate": item["sample_rate"], "checkpoint": item["checkpoint"],
                 "historical_quality_results_bound": False}
        model["fp32"] = include(repo / "models/weights" / artifact["filename"],
                                f"models/{item['id']}/fp32.onnx", item["id"], artifact["sha256"])
        model["config"] = include(repo / item["config"], f"reference/{item['config']}", item["id"])
        models.append(model)
    for identifier in NEW_IDS:
        relative = f"results/piper-ir-onnx-ptq-v2/{identifier}/export_record.json"
        record = read(repo / relative)
        if record["status"] != "passed" or record["label"] != identifier:
            raise ValueError(f"incomplete/mismatched round-2 FP32 export: {identifier}")
        m = record["model"]
        model = {"id": identifier, "label": m["model_id"], "group": "Config A, round-2 FP32",
                 "sample_rate": m["sample_rate"], "checkpoint": {"filename": m["checkpoint"], "sha256": m["checkpoint_sha256"]}}
        model["fp32"] = include(args.q2_root / identifier / record["graph"]["file"],
                                f"models/{identifier}/fp32.onnx", identifier, record["graph"]["sha256"])
        model["config"] = include(repo / m["config"], f"reference/{m['config']}", identifier, m["config_sha256"])
        model["export_record"] = include(repo / relative, f"reference/{identifier}/export_record.json", identifier)
        models.append(model)
    if len(models) != 6:
        raise ValueError("missing a required model")
    tests = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_edge_openvino.py", "-v"],
                           cwd=bundle, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (bundle / "OFFLINE-TESTS.txt").write_text(tests.stdout, encoding="utf-8")
    if tests.returncode:
        raise RuntimeError("portable offline tests failed; see OFFLINE-TESTS.txt")
    write(bundle / "models.json", {"schema_version": 1, "target": "Intel N100 / Ubuntu Linux / OpenVINO CPU",
                                    "created_utc": utc(), "models": models, **metadata,
                                    "int8": "not prepared; uniform selective NNCF calibration must run on N100",
                                    "calibration_validation": check})
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    generated = ["models.json", "metadata/calibration_dataset_rows.jsonl", "OFFLINE-TESTS.txt"]
    inventory.extend({"file": name, "bytes": (bundle / name).stat().st_size, "sha256": sha(bundle / name),
                      "association": "derived kit metadata", "source_copy_hashes": "generated, not a source copy"} for name in generated)
    write(bundle / "PACKAGE-MANIFEST.json", {"source_git_revision": revision, "created_utc": utc(), "payload": inventory,
          "missing_required_files": [], "pending": ["N100 dependency installation/version lock", "OpenVINO FP32 parity",
          "NNCF INT8 calibration and precision audit", "INT8 quality evaluation", "all N100 timing measurements"],
          "excluded": ["external Piper (seventh model)", "Low running and High not completed", "all historical INT8 graphs",
          "round-1 known-defect PTQ artifacts", "checkpoints", "training audio/cache", "venvs", "credentials", "old timing results"]})
    print(json.dumps(archive_and_verify(bundle, output), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--q2-root", type=Path, required=True)
    p.add_argument("--dataset-jsonl", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    build(p.parse_args())
