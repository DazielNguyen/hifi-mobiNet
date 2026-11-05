"""Migration must never publish a partial or legacy INT8 catalog.

Fake IR bytes exercise artifact binding; actual inference is checked separately
by the three-model CPU integration run recorded in docs/quantization.
"""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from hifimobinet import common97, n100_demo as demo

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def migration(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("activate_common97", ROOT / "scripts/n100_demo/prepare_common97.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    catalog = demo.read(ROOT / "demo/n100_models.json")
    manifest = demo.read(ROOT / "results/edge-n100-openvino/2026-11-01/evidence/models.json")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "models.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo/n100_models.json").write_text(json.dumps(catalog), encoding="utf-8")
    work = bundle / "work-new"
    for item in manifest["models"]:
        if item["id"] not in common97.MODEL_IDS:
            continue
        directory = work / item["id"]
        directory.mkdir(parents=True)
        files = []
        for ext in ("xml", "bin"):
            path = directory / f"int8.{ext}"
            path.write_bytes(f"fixture {item['id']} {ext}".encode())
            files.append({"file": path.name, "sha256": demo.sha(path)})
        regression = common97.policy_path().with_name("piper-common97-regression.json")
        cases = [{"case": row["id"], "length_scale": scale, "min_coverage": 1,
                  "max_coverage": 1, "zero_alignment_frames": 0, "audio_seconds": 15}
                 for row in demo.read(regression) for scale in (1.0, 1.3)]
        prep = {"status": "prepared", "model": item, "policy": common97.identity(),
                "script_sha256": demo.sha(module.suite.__file__),
                "manifest_sha256": demo.sha(bundle / "models.json"),
                "environment": {"packages": catalog["expected_packages"]},
                "regression_sha256": demo.sha(regression),
                "numeric_guard_sha256": demo.sha(ROOT / "src/hifimobinet/ptq_numeric_guard.py"),
                "graph_audit": {"passed": True, "fake_quantize_count": 96},
                "runtime_audit": {"passed": True, "int8_weighted_kernel_count": 97},
                "alignment_validation": {"status": "passed", "cases": cases},
                "quality_status": "fixture", "artifacts": {"int8": {"files": files}}}
        demo.write_new(directory / "preparation.json", prep)
    # Host/frontend verification is covered by its own tests. This test isolates
    # migration's real version, policy, regression and file-hash gates.
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module.suite, "manifest", lambda _: manifest)
    monkeypatch.setattr(demo, "preflight", lambda *_: {"status": "fixture"})
    return module, bundle, work, tmp_path / "active.json"


def test_activate_all_three_only_after_success(migration):
    module, bundle, work, target = migration
    catalog = module.activate(bundle, work, target)
    assert demo.read(target) == catalog
    assert {m["id"] for m in catalog["models"] if "int8" in m["artifacts"]} == set(common97.MODEL_IDS)
    assert all(m["artifacts"]["fp32"] for m in catalog["models"])
    assert all("int8" not in m["artifacts"] for m in demo.read(module.ROOT / "demo/n100_models.json")["models"])
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        module.activate(bundle, work, target)
    assert target.read_bytes() == before


@pytest.mark.parametrize("failure", ["missing-third", "corrupt-third", "old-policy", "duplicate-case", "wrong-checkpoint", "changed-guard"])
def test_no_partial_activation_on_failure(migration, failure):
    module, bundle, work, target = migration
    directory = work / common97.MODEL_IDS[-1]
    path = directory / "preparation.json"
    prep = demo.read(path)
    if failure == "missing-third":
        path.unlink()
    elif failure == "corrupt-third":
        (directory / "int8.bin").write_bytes(b"corrupted")
    else:
        if failure == "old-policy":
            prep.pop("policy")
        elif failure == "duplicate-case":
            prep["alignment_validation"]["cases"][-1] = copy.deepcopy(prep["alignment_validation"]["cases"][0])
        elif failure == "wrong-checkpoint":
            prep["model"]["checkpoint"] = "wrong"
        else:
            prep["numeric_guard_sha256"] = "0" * 64
        path.write_text(json.dumps(prep), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        module.activate(bundle, work, target)
    assert not target.exists()


def test_frontend_failure_does_not_publish_catalog(migration, monkeypatch):
    module, bundle, work, target = migration
    def fail(*_):
        raise ValueError("frontend IDs differ")
    monkeypatch.setattr(demo, "preflight", fail)
    with pytest.raises(ValueError, match="frontend"):
        module.activate(bundle, work, target)
    assert not target.exists()
