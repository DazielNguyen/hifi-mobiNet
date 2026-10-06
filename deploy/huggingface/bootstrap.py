"""Fetch only identified runtime files at pinned HF revisions; never load checkpoints."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
from hifimobinet.artifacts import safe_path, check_stream

ROOT=Path(__file__).resolve().parents[2]
COMMIT=re.compile(r"[0-9a-f]{40}\Z")


def revision(name: str, offline: bool):
    value=os.environ.get(name, "")
    if not value and offline:
        return None
    if not COMMIT.fullmatch(value):
        raise ValueError(name+" must be a full 40-character commit from the uploaded HF repository")
    return value


def ensure_file(assets: Path, filename: str, expected: dict, repo: str, commit: str | None, repo_type: str, offline: bool, download):
    path=safe_path(assets,filename)
    if not path.exists():
        if offline:
            raise FileNotFoundError("Required local runtime asset missing: "+filename)
        download(repo_id=repo,filename=filename,repo_type=repo_type,revision=commit,local_dir=str(assets))
    if path.is_symlink() or not path.is_file():
        raise ValueError("Runtime asset must be a regular file")
    with path.open("rb") as stream:
        check_stream(stream,{**expected,"relative_path":filename})
    return path


def bootstrap(assets: Path, offline=False, download=None):
    targets=json.loads((ROOT/"deploy/huggingface/targets.json").read_text())
    model_revision=revision("HIFIMOBINET_MODEL_REVISION",offline)
    enabled=os.environ.get("HIFIMOBINET_ENABLE_HARVARD_AUDIO","0")
    if enabled not in ("0","1"):
        raise ValueError("HIFIMOBINET_ENABLE_HARVARD_AUDIO must be 0 or 1")
    audio_revision=revision("HIFIMOBINET_AUDIO_REVISION",offline) if enabled=="1" else None
    if download is None and not offline:
        from huggingface_hub import hf_hub_download
        download=hf_hub_download
    models=json.loads((ROOT/"models/manifest.json").read_text())["models"]
    for model in models:
        row=model["artifact"]
        ensure_file(assets,"onnx-q05/"+row["filename"],row,targets["model_repo"],model_revision,"model",offline,download)
    catalog_path=safe_path(assets,"manifests/audio-manifest.json")
    if enabled=="1":
        expected=targets["audio_catalog"]
        ensure_file(assets,expected["filename"],expected,targets["dataset_repo"],audio_revision,"dataset",offline,download)
        catalog=json.loads(catalog_path.read_text())
        if len(catalog["samples"])!=2880:
            raise ValueError("Historical catalog must contain 2,880 samples")
        allowed={m["id"] for m in models}
        pairs=set()
        counts={name:0 for name in allowed}
        for sample in catalog["samples"]:
            pair=(sample["utterance_id"],sample["model_id"])
            if sample["model_id"] not in allowed or pair in pairs:
                raise ValueError("Invalid or duplicate historical model/utterance")
            pairs.add(pair)
            counts[sample["model_id"]]+=1
            filename=sample["relative_path"]
            if not filename.startswith("audio/harvard/") or not filename.endswith(".wav"):
                raise ValueError("Historical catalog contains a non-audio path")
            ensure_file(assets,filename,sample,targets["dataset_repo"],audio_revision,"dataset",offline,download)
        if set(counts.values())!={720}:
            raise ValueError("Each model must have 720 historical samples")
    else:
        empty={"schema_version":1,"selection_rule":"Historical comparison is not installed in this deployment.","samples":[]}
        if catalog_path.exists():
            if catalog_path.is_symlink() or json.loads(catalog_path.read_text()).get("samples")!=[]:
                raise ValueError("Use a fresh runtime asset directory for TTS-only mode")
        else:
            catalog_path.parent.mkdir(parents=True,exist_ok=True)
            with catalog_path.open("x") as output:
                output.write(json.dumps(empty,indent=2)+"\n")
    return {"models_verified":len(models),"historical_audio_enabled":enabled=="1","mode":"offline_identity_check" if offline else "pinned_HF_download","model_revision":model_revision,"audio_revision":audio_revision}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline",action="store_true",help="Verify provisioned files without network calls")
    args=parser.parse_args()
    try:
        assets=Path(os.environ["HIFIMOBINET_ASSET_DIR"])
        print(json.dumps(bootstrap(assets,args.offline),indent=2))
    except (OSError,ValueError,KeyError) as error:
        parser.exit(2,"Runtime preparation stopped: "+str(error)+"\n")


if __name__=="__main__":
    main()
