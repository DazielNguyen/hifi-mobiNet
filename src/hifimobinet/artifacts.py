"""Verify and import local artifact directories or checksum-listed ZIP packages.

No model deserialization, network access, extraction helpers or source mutation.
Checksums establish byte identity, not publisher authenticity or research validity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import zipfile

from .registry import sha256

MAX_FILE_BYTES = 2 * 1024**3
MAX_TOTAL_BYTES = 8 * 1024**3
MAX_FILES = 10000
HASH = re.compile(r"[0-9a-f]{64}\Z")


def relative_name(name: str) -> str:
    """Use one portable, strict filename grammar on Windows, Linux and macOS."""
    if not isinstance(name, str) or not name or any(c in name for c in "\\:\x00\n\r"):
        raise ValueError("Invalid relative artifact path")
    parts = name.split("/")
    if any(p in ("", ".", "..") or p.endswith((" ", ".")) for p in parts):
        raise ValueError("Traversal or ambiguous artifact path refused")
    if any(p.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]} for p in parts):
        raise ValueError("Reserved artifact path refused")
    return PurePosixPath(*parts).as_posix()


def safe_path(root: Path, name: str) -> Path:
    """Refuse symlinks in every existing component, including inside the root."""
    relative_name(name)
    root = root.absolute()
    target = root.joinpath(*name.split("/"))
    for item in [target, *target.parents]:
        if item.is_symlink():
            raise ValueError("Symlink source/destination refused")
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes destination")
    return target


def entries(data: dict) -> list[dict]:
    if not isinstance(data, dict):
        raise ValueError("Manifest must be a JSON object")
    result = data.get("artifacts")
    if not isinstance(result, list) or not result or len(result) > MAX_FILES:
        raise ValueError("Invalid or empty artifact manifest")
    seen = set()
    total = 0
    for row in result:
        if not isinstance(row, dict):
            raise ValueError("Each artifact must be a JSON object")
        name = relative_name(row["relative_path"])
        folded = name.casefold()
        if folded in seen:
            raise ValueError("Duplicate/case-colliding artifact path")
        seen.add(folded)
        size = row["bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or not 0 <= size < MAX_FILE_BYTES:
            raise ValueError("Artifact size outside supported bounds")
        if not isinstance(row["sha256"], str) or not HASH.fullmatch(row["sha256"]):
            raise ValueError("Invalid SHA-256")
        total += size
    if total > MAX_TOTAL_BYTES:
        raise ValueError("Package exceeds supported total size")
    for name in seen:
        if any(str(p) in seen for p in PurePosixPath(name).parents if str(p) != "."):
            raise ValueError("File/directory artifact collision")
    return result


def checksum_table(file: Path) -> dict[str, str]:
    result = {}
    for line in file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        if not match:
            raise ValueError("Malformed SHA256SUMS line")
        digest, name = match.groups()
        relative_name(name)
        if name in result:
            raise ValueError("Duplicate checksum entry")
        result[name] = digest
    return result


def check_stream(stream, record: dict) -> None:
    h = hashlib.sha256()
    size = 0
    while block := stream.read(1024 * 1024):
        size += len(block)
        if size > record["bytes"]:
            raise ValueError("Artifact exceeds declared size")
        h.update(block)
    if size != record["bytes"] or h.hexdigest() != record["sha256"]:
        raise ValueError("Artifact checksum/size mismatch: " + record["relative_path"])


def describe_file(path: Path, relative: str) -> dict:
    return {"relative_path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}


class Package:
    """A fully verified source. All ZIP data is checked before any destination write."""

    def __init__(self, source: Path, checksums: Path | None = None):
        self.source = source.absolute()
        self.archive = None
        if source.is_symlink():
            raise ValueError("Symlink package refused")
        if source.is_dir():
            mf = safe_path(source, "manifests/artifact-manifest.json")
            sum_file = safe_path(source, "SHA256SUMS")
            sums = checksum_table(sum_file)
            if sums.get("manifests/artifact-manifest.json") != sha256(mf):
                raise ValueError("Artifact manifest checksum mismatch")
            self.records = entries(json.loads(mf.read_text(encoding="utf-8")))
            for row in self.records:
                if sums.get(row["relative_path"]) != row["sha256"]:
                    raise ValueError("Manifest and SHA256SUMS disagree")
                path = safe_path(source, row["relative_path"])
                if not path.is_file():
                    raise FileNotFoundError("Missing artifact: " + row["relative_path"])
                with path.open("rb") as f:
                    check_stream(f, row)
            self.records = [*self.records, describe_file(mf, "manifests/artifact-manifest.json"), describe_file(sum_file, "SHA256SUMS")]
        elif source.is_file() and source.suffix.lower() == ".zip":
            sums = checksum_table(checksums or source.parent / "SHA256SUMS")
            if sums.get(source.name) != sha256(source):
                raise ValueError("Archive missing from SHA256SUMS or checksum differs")
            self.archive = zipfile.ZipFile(source)
            try:
                infos = self.archive.infolist()
                names = [relative_name(i.filename) for i in infos]
                if len(names) > MAX_FILES or len({n.casefold() for n in names}) != len(names):
                    raise ValueError("Duplicate or excessive ZIP members")
                for info in infos:
                    mode = info.external_attr >> 16
                    if info.is_dir() or stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                        raise ValueError("Nonregular ZIP member refused (including links)")
                    if info.flag_bits & 1 or info.file_size >= MAX_FILE_BYTES:
                        raise ValueError("Encrypted or oversized ZIP member refused")
                metadata = [n for n in names if n.startswith("manifests/packages/") and n.endswith(".json")]
                if len(metadata) != 1 or self.archive.getinfo(metadata[0]).file_size > 10 * 1024**2:
                    raise ValueError("ZIP must contain one bounded package manifest")
                raw = self.archive.read(metadata[0])
                self.records = entries(json.loads(raw))
                if set(names) != {r["relative_path"] for r in self.records} | {metadata[0]}:
                    raise ValueError("ZIP members do not exactly match the manifest")
                for row in self.records:
                    if self.archive.getinfo(row["relative_path"]).file_size != row["bytes"]:
                        raise ValueError("ZIP size differs from manifest")
                    with self.archive.open(row["relative_path"]) as stream:
                        check_stream(stream, row)
                self.records = [*self.records, {"relative_path": metadata[0], "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}]
            except BaseException:
                self.archive.close()
                raise
        else:
            raise FileNotFoundError("Provide an existing staging directory or .zip package")

    def open(self, row):
        if self.archive:
            return self.archive.open(row["relative_path"])
        return safe_path(self.source, row["relative_path"]).open("rb")

    def close(self):
        if self.archive:
            self.archive.close()


def import_packages(packages: list[Package], destination: Path) -> dict:
    plan = {}
    for package in packages:
        for row in package.records:
            name = row["relative_path"]
            if name in plan and (plan[name][1]["sha256"], plan[name][1]["bytes"]) != (row["sha256"], row["bytes"]):
                raise ValueError("Packages disagree about a shared file")
            plan[name] = (package, row)
    entries({"artifacts": [row for _, row in plan.values()]})
    needed = 0
    for name, (_, row) in plan.items():
        path = safe_path(destination, name)
        if path.exists():
            if not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
                raise ValueError("Refusing to overwrite a different file: " + name)
        else:
            needed += row["bytes"]
        for parent in path.parents:
            if parent.exists() and not parent.is_dir():
                raise ValueError("Destination parent is a file")
    existing_parent = destination.absolute()
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    if shutil.disk_usage(existing_parent).free < needed + 64 * 1024**2:
        raise ValueError("Insufficient free space for selected packages")
    copied = 0
    for name, (package, row) in plan.items():
        path = safe_path(destination, name)
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also refuses a file introduced after preflight.
        created = False
        try:
            with path.open("xb") as output:
                created = True
                with package.open(row) as stream:
                    shutil.copyfileobj(stream, output, 1024 * 1024)
            if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
                raise ValueError("Post-copy checksum failure: " + name)
        except BaseException:
            if created:
                path.unlink(missing_ok=True)  # Only this invocation's incomplete new file.
            raise
        copied += 1
    return {"verified_files": len(plan), "copied_files": copied, "existing_identical_files": len(plan)-copied}


def runtime_assets_status(destination: Path) -> dict:
    """Report missing runtime packages; never enable another model as a substitute."""
    from .registry import manifest
    missing = ["onnx-q05/" + m["artifact"]["filename"] for m in manifest()["models"]
               if not safe_path(destination, "onnx-q05/"+m["artifact"]["filename"]).is_file()]
    catalog = safe_path(destination, "manifests/audio-manifest.json")
    if not catalog.is_file():
        missing.append("manifests/audio-manifest.json")
    else:
        data = json.loads(catalog.read_text(encoding="utf-8"))
        missing.extend(s["relative_path"] for s in data["samples"] if not safe_path(destination, s["relative_path"]).is_file())
    return {"missing_runtime_asset_count": len(missing), "missing_runtime_assets_first_10": missing[:10],
            "limits": "Presence only here; CLI/demo verify hashes again. Native frontend and Mac inference remain separate checks."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["verify", "import"])
    parser.add_argument("sources", type=Path, nargs="+")
    parser.add_argument("--checksums", type=Path, help="Archive SHA256SUMS (defaults to each archive's sibling)")
    parser.add_argument("--destination", type=Path, help="External artifact store for import")
    args = parser.parse_args(argv)
    if args.action == "import" and not args.destination:
        parser.error("--destination is required for import")
    packages = []
    try:
        for source in args.sources:
            packages.append(Package(source, args.checksums))
        if args.action == "verify":
            print(json.dumps({"status": "verified", "packages": len(packages), "files_including_shared_metadata": sum(len(p.records) for p in packages)}))
        else:
            report = import_packages(packages, args.destination)
            report.update(runtime_assets_status(args.destination))
            print(json.dumps(report, indent=2))
            print("Set HIFIMOBINET_ASSET_DIR to the destination for both CLI and Streamlit.")
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        print("Artifact operation stopped: " + str(error), file=sys.stderr)
        return 2
    finally:
        for package in packages:
            package.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
