"""Deployment checks for pinned identity and explicit network behavior."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("hf_bootstrap",ROOT/"deploy/huggingface/bootstrap.py")
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


@pytest.fixture
def runtime(tmp_path,monkeypatch):
    code=tmp_path/"code"
    (code/"deploy/huggingface").mkdir(parents=True)
    (code/"models").mkdir()
    content=b"identified model bytes"
    identity={"filename":"seq.onnx","bytes":len(content),"sha256":hashlib.sha256(content).hexdigest()}
    (code/"models/manifest.json").write_text(json.dumps({"models":[{"id":"sequential-ir","artifact":identity}]}))
    (code/"deploy/huggingface/targets.json").write_text(json.dumps({"model_repo":"DazielNguyen/hifi-mobiNet","dataset_repo":"DazielNguyen/hifi-mobiNet-harvard"}))
    monkeypatch.setattr(mod,"ROOT",code)
    for name in ["HIFIMOBINET_MODEL_REVISION","HIFIMOBINET_AUDIO_REVISION","HIFIMOBINET_ENABLE_HARVARD_AUDIO"]:
        monkeypatch.delenv(name,raising=False)
    return tmp_path/"assets",content


def test_missing_or_moving_revision_never_downloads(runtime,monkeypatch):
    assets,_=runtime
    calls=[]
    for value in ("","main","v0.1.0","a"*7):
        monkeypatch.setenv("HIFIMOBINET_MODEL_REVISION",value)
        with pytest.raises(ValueError,match="40-character"):
            mod.bootstrap(assets,download=lambda **kwargs:calls.append(kwargs))
    assert calls==[]


def test_download_bound_to_repo_revision_hash_and_no_checkpoints(runtime,monkeypatch):
    assets,content=runtime
    monkeypatch.setenv("HIFIMOBINET_MODEL_REVISION","a"*40)
    calls=[]
    def download(**kwargs):
        calls.append(kwargs)
        path=Path(kwargs["local_dir"])/kwargs["filename"]
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
    report=mod.bootstrap(assets,download=download)
    assert report["models_verified"]==1 and not report["historical_audio_enabled"]
    assert calls==[{"repo_id":"DazielNguyen/hifi-mobiNet","filename":"onnx-q05/seq.onnx","repo_type":"model","revision":"a"*40,"local_dir":str(assets)}]
    assert json.loads((assets/"manifests/audio-manifest.json").read_text())["samples"]==[]


def test_corrupt_existing_model_is_rejected_without_substitution(runtime):
    assets,_=runtime
    path=assets/"onnx-q05/seq.onnx"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"wrong bytes")
    with pytest.raises(ValueError,match="checksum"):
        mod.bootstrap(assets,offline=True)


def test_audio_activation_requires_separate_pinned_revision(runtime,monkeypatch):
    assets,_=runtime
    monkeypatch.setenv("HIFIMOBINET_MODEL_REVISION","a"*40)
    monkeypatch.setenv("HIFIMOBINET_ENABLE_HARVARD_AUDIO","1")
    calls=[]
    with pytest.raises(ValueError,match="AUDIO_REVISION"):
        mod.bootstrap(assets,download=lambda **kwargs:calls.append(kwargs))
    assert calls==[]


def test_offline_verification_makes_no_network_calls(runtime):
    assets,content=runtime
    path=assets/"onnx-q05/seq.onnx"
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    def forbidden(**kwargs):
        raise AssertionError("Offline mode must not download")
    assert mod.bootstrap(assets,offline=True,download=forbidden)["mode"]=="offline_identity_check"


def test_path_escape_and_symlink_are_rejected(runtime,tmp_path):
    assets,content=runtime
    outside=tmp_path/"outside"
    outside.write_bytes(content)
    with pytest.raises(ValueError):
        mod.ensure_file(assets,"../outside",{},"unused",None,"model",True,None)
    path=assets/"onnx-q05/seq.onnx"
    path.parent.mkdir(parents=True)
    path.symlink_to(outside)
    with pytest.raises(ValueError,match="Symlink"):
        mod.bootstrap(assets,offline=True)


def test_release_preparer_refuses_existing_staging_without_touching_it(tmp_path):
    spec=importlib.util.spec_from_file_location("hf_prepare",ROOT/"scripts/huggingface/prepare_release.py")
    prepare=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    stage=tmp_path/"stage"
    stage.mkdir()
    marker=stage/"keep.txt"
    marker.write_text("existing user data")
    with pytest.raises(ValueError,match="Existing staging"):
        prepare.prepare(tmp_path/"absent-assets",stage)
    assert marker.read_text()=="existing user data"


def test_release_preparer_refuses_a_symlink_source(tmp_path):
    spec=importlib.util.spec_from_file_location("hf_prepare",ROOT/"scripts/huggingface/prepare_release.py")
    prepare=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    source=tmp_path/"original"
    source.write_bytes(b"source")
    link=tmp_path/"link"
    link.symlink_to(source)
    with pytest.raises(ValueError,match="regular"):
        prepare.copy_verified(link,tmp_path/"new-copy")
    assert not (tmp_path/"new-copy").exists()


def test_checkpoint_candidate_cannot_target_public_inference_repository():
    spec = importlib.util.spec_from_file_location("hf_prepare_scope", ROOT / "scripts/huggingface/prepare_release.py")
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    public = "DazielNguyen/hifi-mobiNet-inference"
    for targets in ({"model_repo": public}, {"model_repo": public, "private_checkpoint_repo": public}):
        with pytest.raises(ValueError, match="separate private checkpoint"):
            prepare.private_checkpoint_repository(targets)
    assert prepare.private_checkpoint_repository({"model_repo": public, "private_checkpoint_repo": "DazielNguyen/hifi-mobiNet"}) == "DazielNguyen/hifi-mobiNet"
