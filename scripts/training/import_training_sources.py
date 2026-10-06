"""Import the training-only reference sources needed by src/hifimobinet/training.

Run from the hifi-mobiNet checkout inside the FPT-Graduation-Project workspace:

    python scripts/training/import_training_sources.py

Source repositories are read only:
- Piper files are read from Git objects at the pinned commit (exact upstream bytes,
  independent of the dirty local working tree).
- BanhmiTTS files are read from the working tree, matching the convention of the
  first import (docs/source-map.json); each file must be unchanged from HEAD.

Destinations under vendor/ are byte-identical copies. Existing source-map entries
are kept unchanged; new entries are appended. Refuses to overwrite a destination
whose bytes differ from the source.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
PIPER_COMMIT = "73c04d81d5590ecc46e522de3601ce7fb29fc2be"
PIPER_REMOTE = "https://github.com/rhasspy/piper.git"

PIPER_FILES = {
    "src/python/piper_train/vits/monotonic_align/__init__.py": "vendor/piper/vits/monotonic_align/__init__.py",
    "src/python/piper_train/vits/monotonic_align/core.pyx": "vendor/piper/vits/monotonic_align/core.pyx",
    "src/python/piper_train/vits/monotonic_align/setup.py": "vendor/piper/vits/monotonic_align/setup.py",
    "src/python/piper_train/vits/monotonic_align/Makefile": "vendor/piper/vits/monotonic_align/Makefile",
    "src/python/piper_train/__main__.py": "vendor/piper/__main__.py",
    "src/python/build_monotonic_align.sh": "vendor/piper/build_monotonic_align.sh",
    "src/python/requirements.txt": "vendor/piper/requirements.txt",
}
EDGETTS_REMOTE = "https://github.com/MardeusVN/PitchFlowNet.git"
EDGETTS_FILES = {
    **{f"src/python/piper_train/vits/{name}.py": f"vendor/edgetts/vits/{name}.py" for name in (
        "__init__", "attentions", "commons", "config", "dataset", "lightning", "losses",
        "mel_processing", "models", "modules", "transforms", "utils", "wavfile")},
    **{f"src/python/piper_train/vits/monotonic_align/{name}": f"vendor/edgetts/vits/monotonic_align/{name}"
       for name in ("__init__.py", "core.pyx", "setup.py", "Makefile")},
    "src/python/piper_train/__main__.py": "vendor/edgetts/__main__.py",
    "src/python/requirements.txt": "vendor/edgetts/requirements.txt",
    "configs/baseline-vits.yaml": "vendor/edgetts/configs/baseline-vits.yaml",
    "configs/03_Config_C_VITS2.yaml": "vendor/edgetts/configs/03_Config_C_VITS2.yaml",
    "LICENSE.md": "vendor/edgetts/LICENSE.md",
}
BANHMI_FILES = {
    "banhmi_train/train.py": "vendor/banhmi/train.py",
    "banhmi_train/vits/training.py": "vendor/banhmi/vits/training.py",
    "banhmi_train/vits/dataset.py": "vendor/banhmi/vits/dataset.py",
    "banhmi_train/vits/length_bucket_sampler.py": "vendor/banhmi/vits/length_bucket_sampler.py",
    "banhmi_train/warm_length_cache.py": "vendor/banhmi/warm_length_cache.py",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args])


def place(dest_rel: str, data: bytes) -> None:
    dest = ROOT / dest_rel
    if dest.exists():
        if dest.read_bytes() != data:
            raise SystemExit(f"Refusing to overwrite changed destination: {dest_rel}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


def main() -> None:
    piper = WORKSPACE / "piper"
    banhmi = WORKSPACE / "BanhmiTTS"
    remote = git(piper, "remote", "get-url", "origin").decode().strip()
    if remote != PIPER_REMOTE:
        raise SystemExit(f"Unexpected Piper remote: {remote}")
    if git(piper, "cat-file", "-t", PIPER_COMMIT).strip() != b"commit":
        raise SystemExit("Pinned Piper commit is not available locally")
    banhmi_head = git(banhmi, "rev-parse", "HEAD").decode().strip()
    banhmi_dirty = bool(git(banhmi, "status", "--porcelain").strip())

    source_map_path = ROOT / "docs" / "source-map.json"
    source_map = json.loads(source_map_path.read_text(encoding="utf-8"))
    known = {e["destination"] for e in source_map["entries"]}
    new_entries = []

    for src, dest in PIPER_FILES.items():
        data = git(piper, "show", f"{PIPER_COMMIT}:{src}")
        place(dest, data)
        new_entries.append({
            "source": f"piper/{src}",
            "source_repo": "piper",
            "source_commit": PIPER_COMMIT,
            "source_remote": PIPER_REMOTE,
            "source_file_state": "git_object_at_pinned_commit",
            "repository_tracked_dirty": True,
            "source_git_blob": git(piper, "rev-parse", f"{PIPER_COMMIT}:{src}").decode().strip(),
            "source_sha256": sha(data),
            "destination": dest,
            "destination_sha256": sha((ROOT / dest).read_bytes()),
            "bytes": len(data),
            "classification": "upstream_reference_for_training",
            "purpose": "Unmodified upstream Piper training reference (MAS build/import, entry point, requirements)",
            "license_status": "upstream_MIT_retained",
            "release_status": "code_publication_authorized_upstream_terms_review_pending",
        })

    edgetts = WORKSPACE / "EdgeTTS"
    if git(edgetts, "remote", "get-url", "origin").decode().strip() != EDGETTS_REMOTE:
        raise SystemExit("Unexpected EdgeTTS remote")
    edgetts_head = git(edgetts, "rev-parse", "HEAD").decode().strip()
    edgetts_dirty = bool(git(edgetts, "status", "--porcelain").strip())
    for src, dest in EDGETTS_FILES.items():
        data = git(edgetts, "show", f"{edgetts_head}:{src}")
        place(dest, data)
        new_entries.append({
            "source": f"EdgeTTS/{src}",
            "source_repo": "EdgeTTS",
            "source_commit": edgetts_head,
            "source_remote": EDGETTS_REMOTE,
            "source_file_state": "git_object_at_head",
            "repository_tracked_dirty": edgetts_dirty,
            "source_git_blob": git(edgetts, "rev-parse", f"{edgetts_head}:{src}").decode().strip(),
            "source_sha256": sha(data),
            "destination": dest,
            "destination_sha256": sha((ROOT / dest).read_bytes()),
            "bytes": len(data),
            "classification": "project_fork_reference_for_training",
            "purpose": "EdgeTTS (PitchFlowNet) fork of Piper 73c04d8; Config A = all component flags off",
            "license_status": "upstream_MIT_retained_fork_changes_author_owned",
            "release_status": "code_publication_authorized_upstream_terms_review_pending",
        })

    for src, dest in BANHMI_FILES.items():
        path = banhmi / src
        if git(banhmi, "status", "--porcelain", "--", src).strip():
            raise SystemExit(f"BanhmiTTS source differs from HEAD: {src}")
        data = path.read_bytes()
        place(dest, data)
        new_entries.append({
            "source": f"BanhmiTTS/{src}",
            "source_repo": "BanhmiTTS",
            "source_commit": banhmi_head,
            "source_file_state": "tracked_clean",
            "repository_tracked_dirty": banhmi_dirty,
            "source_git_blob": git(banhmi, "rev-parse", f"HEAD:{src}").decode().strip(),
            "source_sha256": sha(data),
            "destination": dest,
            "destination_sha256": sha((ROOT / dest).read_bytes()),
            "bytes": len(data),
            "classification": "current_source",
            "purpose": "Current BanhmiTTS training harness reference; not proven to be the historical training-time source",
            "license_status": "author_review_pending",
            "release_status": "local_only_pending_rights_review",
        })

    added = [e for e in new_entries if e["destination"] not in known]
    source_map["entries"].extend(added)
    # Bytes, not write_text(): text mode would turn LF into CRLF on Windows.
    source_map_path.write_bytes((json.dumps(source_map, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print(f"Imported {len(new_entries)} files; appended {len(added)} source-map entries")


if __name__ == "__main__":
    sys.exit(main())
