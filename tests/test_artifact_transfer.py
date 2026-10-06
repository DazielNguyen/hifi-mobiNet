"""Small synthetic transfer fixtures; never substitute for research artifacts."""
import hashlib
import json
from pathlib import Path
import stat
import zipfile

import pytest

from hifimobinet.artifacts import Package, import_packages, relative_name
from hifimobinet.registry import model_path, sha256
from hifimobinet.comparison import audio_catalog


def record(name, data):
    return {"relative_path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def archive(tmp_path, files=None, payload=None, special=None):
    files = files or {"onnx-q05/fixture.onnx": b"synthetic fixture, not a model"}
    path = tmp_path / "fixture.zip"
    rows = payload or [record(n, d) for n, d in files.items()]
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            if special:
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = (special | 0o777) << 16
                z.writestr(info, data)
            else:
                z.writestr(name, data)
        z.writestr("manifests/packages/fixture.json", json.dumps({"artifacts": rows}))
    (tmp_path / "SHA256SUMS").write_text(sha256(path) + "  fixture.zip\n")
    return path


def test_valid_archive_import_is_repeatable_and_refuses_conflict(tmp_path):
    path = archive(tmp_path)
    p = Package(path)
    try:
        target = tmp_path / "installed"
        assert import_packages([p], target)["copied_files"] == 2
        assert import_packages([p], target)["copied_files"] == 0
        f = target / "onnx-q05/fixture.onnx"
        f.write_bytes(b"another file")
        with pytest.raises(ValueError, match="overwrite"):
            import_packages([p], target)
        assert f.read_bytes() == b"another file"
    finally:
        p.close()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "folder/../../escape", "x\\..\\escape", "X:/escape", "./a", "a//b", "NUL.txt"])
def test_traversal_and_ambiguous_names_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        relative_name(name)
    with pytest.raises(ValueError):
        Package(archive(tmp_path, {name: b"malicious member"}))
    assert not (tmp_path.parent / "escape").exists()


def test_zip_symlink_rejected(tmp_path):
    with pytest.raises(ValueError, match="Nonregular"):
        Package(archive(tmp_path, special=stat.S_IFLNK))


def test_outer_and_inner_checksum_mismatch(tmp_path):
    path = archive(tmp_path)
    (tmp_path / "SHA256SUMS").write_text("0" * 64 + "  fixture.zip\n")
    with pytest.raises(ValueError, match="Archive"):
        Package(path)
    path = archive(tmp_path, {"a": b"actual"}, [record("a", b"wrong!")])
    with pytest.raises(ValueError, match="checksum"):
        Package(path)


def test_unlisted_member_case_collision_and_missing_member(tmp_path):
    for files, rows in [({"a": b"a", "b": b"b"}, [record("a", b"a")]),
                        ({"A": b"a", "a": b"a"}, None),
                        ({"a": b"a"}, [record("a", b"a"), record("missing", b"b")])]:
        with pytest.raises(ValueError):
            Package(archive(tmp_path, files, rows))


def test_directory_import_and_missing_source(tmp_path):
    source = tmp_path / "staging"
    (source / "manifests").mkdir(parents=True)
    (source / "payload").write_bytes(b"fixture")
    mf = source / "manifests/artifact-manifest.json"
    mf.write_text(json.dumps({"artifacts": [record("payload", b"fixture")]}))
    (source / "SHA256SUMS").write_text(sha256(mf)+"  manifests/artifact-manifest.json\n"+sha256(source/"payload")+"  payload\n")
    package = Package(source)
    dest = tmp_path / "dest"
    assert import_packages([package], dest)["copied_files"] == 3
    assert len(Package(dest).records) == 3
    (source / "payload").unlink()
    with pytest.raises(FileNotFoundError, match="Missing artifact"):
        Package(source)


def test_shared_asset_directory_for_models_and_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("HIFIMOBINET_ASSET_DIR", str(tmp_path))
    monkeypatch.delenv("HIFIMOBINET_MODEL_DIR", raising=False)
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "onnx-q05").mkdir()
    data = b"fixture"
    path = tmp_path / "onnx-q05/fixture.onnx"
    path.write_bytes(data)
    model = {"artifact": {"format": "onnx", "precision": "FP32", "filename": path.name, "bytes": len(data), "sha256": sha256(path)}}
    assert model_path(model, root) == path
    (tmp_path / "manifests").mkdir()
    catalog = {"samples": [{"utterance_id": "harvard-0000", "model_id": "baseline-resblock2", "text": "fixture"}]}
    (tmp_path / "manifests/audio-manifest.json").write_text(json.dumps(catalog))
    assert audio_catalog(root) == catalog


def test_destination_parent_file_does_not_partially_import(tmp_path):
    p = Package(archive(tmp_path, {"first": b"a", "blocked/file": b"b"}))
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "blocked").write_bytes(b"keep")
    try:
        with pytest.raises(ValueError):
            import_packages([p], dest)
        assert not (dest / "first").exists()
        assert (dest / "blocked").read_bytes() == b"keep"
    finally:
        p.close()
