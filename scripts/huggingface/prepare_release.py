"""Prepare local HF candidates without network calls, pickle loads or source changes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from hifimobinet.artifacts import safe_path, check_stream
from hifimobinet.registry import sha256

ROOT = Path(__file__).resolve().parents[2]


def copy_verified(source: Path, destination: Path, expected: dict | None = None):
    if source.is_symlink() or not source.is_file():
        raise ValueError("Only regular source files are accepted")
    if expected:
        with source.open("rb") as stream:
            check_stream(stream, expected)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
    if destination.stat().st_size != source.stat().st_size or sha256(destination) != sha256(source):
        raise ValueError("Post-copy identity mismatch")


def tracked():
    return subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z"]).decode().strip("\0").split("\0")


def inventory(folder: Path):
    rows = []
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise ValueError("Candidate contains a symlink")
        if path.is_file():
            rows.append({"relative_path": path.relative_to(folder).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return rows


def prepare(assets: Path, destination: Path):
    assets = assets.absolute()
    destination = destination.absolute()
    if destination.exists():
        raise ValueError("Use a new destination. Existing staging is never overwritten.")
    for path in [destination, *destination.parents]:
        if path.is_symlink():
            raise ValueError("Destination symlink refused")
    if destination.resolve().is_relative_to(ROOT) or destination.resolve().is_relative_to(assets.resolve()):
        raise ValueError("Staging must be outside Git and the source asset directory")
    source_manifest = safe_path(assets, "manifests/artifact-manifest.json")
    source_rows = json.loads(source_manifest.read_text())["artifacts"]
    lookup = {row["relative_path"]: row for row in source_rows}
    if len(lookup) != len(source_rows):
        raise ValueError("Duplicate source identity")
    models = json.loads((ROOT/"models/manifest.json").read_text())["models"]
    catalog = json.loads(safe_path(assets,"manifests/audio-manifest.json").read_text())
    if len(catalog["samples"]) != 2880:
        raise ValueError("Expected 2,880 catalogued audio files")
    targets = json.loads((ROOT/"deploy/huggingface/targets.json").read_text())
    source_files = []
    for model in models:
        key = next((name for name in lookup if name.startswith("checkpoints/") and name.endswith("/"+model["checkpoint"]["filename"])), None)
        if key is None:
            raise ValueError("Selected checkpoint missing from source manifest")
        for kind, name in [("checkpoint", key), ("artifact", "onnx-q05/"+model["artifact"]["filename"])]:
            expected = model[kind]
            if any(lookup[name][field] != expected[field] for field in ("bytes","sha256")):
                raise ValueError("Model and source manifest identities differ")
            source_files.append(("model",name))
    source_files += [("model",name) for name in lookup if name.startswith(("configs/","logs/public/"))]
    source_files += [("model","manifests/redaction-record.json")]
    source_files += [("dataset",row["relative_path"]) for row in catalog["samples"]]
    source_files += [("dataset","manifests/audio-manifest.json")]
    catalog_identity=targets["audio_catalog"]
    for field in ("bytes","sha256"):
        if lookup["manifests/audio-manifest.json"][field] != catalog_identity[field]:
            raise ValueError("Audio catalog differs from the pinned deployment identity")
    for sample in catalog["samples"]:
        if any(lookup[sample["relative_path"]][field] != sample[field] for field in ("bytes","sha256")):
            raise ValueError("Audio catalog and source manifest differ")
    needed=sum(lookup[name]["bytes"] for _,name in source_files)+16*1024**2
    parent=destination.parent
    while not parent.exists():
        parent=parent.parent
    if shutil.disk_usage(parent).free < needed:
        raise ValueError("Insufficient space for candidate copies")
    # Validate every source before creating any candidate file.
    for _,name in source_files:
        with safe_path(assets,name).open("rb") as stream:
            check_stream(stream,lookup[name])
    destination.mkdir(parents=True)
    for target,name in source_files:
        copy_verified(safe_path(assets,name),safe_path(destination/target,name),lookup[name])
    notices=["LICENSE","THIRD_PARTY_NOTICES.md","docs/artifact-licensing.json","docs/artifact-licensing.md"]
    notices += [name for name in tracked() if name.startswith("licenses/")]
    for target in ("model","dataset"):
        for name in notices:
            if target=="dataset" and name=="LICENSE":
                continue
            copy_verified(safe_path(ROOT,name),safe_path(destination/target,name))
        copy_verified(ROOT/f"deploy/huggingface/{target}-card.md",destination/target/"README.md")
    copy_verified(ROOT/"LICENSE",destination/"dataset/licenses/Team-MIT.txt")
    copy_verified(ROOT/"licenses/Harvard-generated-audio-CC-BY-4.0.md",destination/"dataset/LICENSE.md")
    license_notice=destination/"dataset/LICENSE.md"
    license_notice.write_text(license_notice.read_text().replace("../docs/artifact-licensing.json","docs/artifact-licensing.json"))
    dataset_doc=destination/"dataset/docs/artifact-licensing.md"
    dataset_doc.write_text(dataset_doc.read_text().replace("[MIT](../LICENSE)","[MIT](../licenses/Team-MIT.txt)"))
    copy_verified(ROOT/"models/manifest.json",destination/"model/models/manifest.json")
    for name in tracked():
        if name.startswith(("src/","vendor/","configs/","models/","demo/","tests/","scripts/huggingface/","deploy/huggingface/")) or name in ("pyproject.toml","LICENSE","THIRD_PARTY_NOTICES.md") or name.startswith("licenses/") or name in ("docs/artifact-licensing.json","docs/artifact-licensing.md"):
            copy_verified(safe_path(ROOT,name),safe_path(destination/"space",name))
    for source,target in [("space-card.md","README.md"),("Dockerfile","Dockerfile"),("dockerignore",".dockerignore")]:
        copy_verified(ROOT/"deploy/huggingface"/source,destination/"space"/target)
    reports={}
    for target,repo in [("model",targets["model_repo"]),("dataset",targets["dataset_repo"]),("space",targets["space_repo"])]:
        rows=inventory(destination/target)
        report={"schema_version":1,"repo_id":repo,"status":"local_candidate_not_uploaded","required_initial_visibility":"private","harvard_rights":"pending_confirmation","source_code_basis":"Git-listed working files, including staged changes; HEAD is the preparation base, not a claim of a clean committed snapshot.","source_code_head":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"]).decode().strip(),"files":rows,"scope":"This manifest excludes itself. Original source manifests and asset bytes remain unchanged."}
        (destination/target/"release-manifest.json").write_text(json.dumps(report,indent=2)+"\n")
        reports[target]={"files":len(rows),"bytes":sum(row["bytes"] for row in rows),"repo_id":repo}
    (destination/"PREPARATION.json").write_text(json.dumps({"status":"local_only","targets":reports,"source_assets_changed":False,"network_actions":False},indent=2)+"\n")
    return reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets",required=True,type=Path)
    parser.add_argument("--destination",required=True,type=Path)
    args=parser.parse_args()
    try:
        print(json.dumps(prepare(args.assets,args.destination),indent=2))
    except (OSError,ValueError,KeyError,StopIteration) as error:
        parser.exit(2,str(error)+"\n")


if __name__=="__main__":
    main()
