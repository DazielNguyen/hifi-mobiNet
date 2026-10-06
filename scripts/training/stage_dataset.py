"""Copy a preprocessed dataset into this repository's ignored data/ directory.

    python scripts/training/stage_dataset.py \\
        --source-data-root SRC_ROOT \\
        --source-dataset-dir SRC_ROOT/training_output/ljspeech/medium \\
        --name ljspeech-medium \\
        --expect-dataset-sha256 f4a72ae0ed238dfdb3ca995b6678dfe33fe3b59297331d6551a4d6960102295b

Result (never tracked by Git):

    data/<name>/                                   <- --data-root for training
      <dataset dir relative to SRC_ROOT>/          <- --dataset-dir for training
        dataset.jsonl, config.json, .spectrogram_lengths_cache.json
      <tensor paths exactly as written in dataset.jsonl>
      STAGING_RECORD.json

Tensor paths in dataset.jsonl are relative to the source root; the same layout
is reproduced under data/<name>/, so dataset.jsonl stays byte-identical and its
SHA-256 identity is kept. Every copied file is verified by SHA-256 against its
source. Re-running skips files already present with matching hashes. The source
is only read. The canonical split is checked against the staged rows.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "results" / "manifests" / "canonical_split.json"
SIDE_FILES = ("dataset.jsonl", "config.json", ".spectrogram_lengths_cache.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_verified(src: Path, dst: Path) -> tuple:
    """Copy src to dst unless an identical file is already there. Returns (copied, sha256)."""
    src_hash = sha256(src)
    if dst.exists() and dst.stat().st_size == src.stat().st_size and sha256(dst) == src_hash:
        return False, src_hash
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    shutil.copyfile(src, tmp)
    if sha256(tmp) != src_hash:
        tmp.unlink()
        raise SystemExit(f"hash mismatch after copying {src}")
    tmp.replace(dst)
    return True, src_hash


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-data-root", required=True, type=Path)
    p.add_argument("--source-dataset-dir", required=True, type=Path)
    p.add_argument("--name", required=True)
    p.add_argument("--expect-dataset-sha256")
    args = p.parse_args()

    src_root = args.source_data_root.resolve()
    src_ds = args.source_dataset_dir.resolve()
    rel_ds = src_ds.relative_to(src_root)
    dest_root = ROOT / "data" / args.name
    if dest_root.resolve() == src_root:
        raise SystemExit("source and destination are the same directory")

    dataset_sha = sha256(src_ds / "dataset.jsonl")
    if args.expect_dataset_sha256 and dataset_sha != args.expect_dataset_sha256:
        raise SystemExit(f"dataset.jsonl SHA-256 {dataset_sha} != expected {args.expect_dataset_sha256}")

    rows = [json.loads(line) for line in (src_ds / "dataset.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    keys = {str(r["audio_norm_path"]) for r in rows}
    missing = {name: sum(1 for k in split[name] if k not in keys) for name in ("train", "val", "test")}
    if any(missing.values()):
        raise SystemExit(f"canonical split entries missing from the dataset: {missing}")

    files = [(src_ds / name, dest_root / rel_ds / name) for name in SIDE_FILES if (src_ds / name).exists()]
    for row in rows:
        for field in ("audio_norm_path", "audio_spec_path"):
            rel = Path(row[field])
            if rel.is_absolute():
                raise SystemExit(f"absolute tensor path in dataset.jsonl: {rel}")
            files.append((src_root / rel, dest_root / rel))

    copied = total_bytes = 0
    for i, (src, dst) in enumerate(files, 1):
        did_copy, _ = copy_verified(src, dst)
        copied += did_copy
        total_bytes += src.stat().st_size
        if i % 2000 == 0:
            print(f"{i}/{len(files)} files verified", flush=True)

    record = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_data_root": str(src_root),
        "dataset_dir_relative": rel_ds.as_posix(),
        "dataset_jsonl_sha256": dataset_sha,
        "config_json_sha256": sha256(src_ds / "config.json"),
        "rows": len(rows),
        "files": len(files),
        "files_copied_this_run": copied,
        "bytes": total_bytes,
        "split_sha256": sha256(SPLIT),
        "verification": "every file SHA-256-compared with its source after copying",
        "f0_files": "not copied (no recipe reads F0)",
    }
    (dest_root / "STAGING_RECORD.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("dataset_dir_relative", "rows", "files", "files_copied_this_run", "bytes")}))
    print(f"--data-root {dest_root.relative_to(ROOT).as_posix()}  --dataset-dir {(dest_root / rel_ds).relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
