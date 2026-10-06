"""Explicit manifest URL downloads; validates size and SHA-256 before promotion."""
from __future__ import annotations
import argparse
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from .registry import model_record, repository_root, sha256, within


def download(model_id: str, kind: str, output_dir: Path) -> Path:
    item = model_record(model_id)[kind]
    url = item.get("url")
    if not url:
        raise ValueError("Download URL pending. No public artifact is substituted.")
    if urllib.parse.urlparse(url).scheme != "https":
        raise ValueError("Only the explicit HTTPS URL recorded in the manifest is allowed")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = within(output_dir.resolve(), item["filename"])
    if target.exists():
        if sha256(target) == item["sha256"] and target.stat().st_size == item["bytes"]:
            return target
        raise ValueError("Destination already exists with different contents; refusing overwrite")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
            with urllib.request.urlopen(url, timeout=60) as response:
                if urllib.parse.urlparse(response.geturl()).scheme != "https":
                    raise ValueError("Download redirected away from HTTPS")
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > item["bytes"]:
                        raise ValueError("Download exceeded the manifest size")
                    stream.write(chunk)
        if temporary.stat().st_size != item["bytes"] or sha256(temporary) != item["sha256"]:
            raise ValueError("Download checksum or size mismatch; file was not installed")
        # A concurrently created destination must not be silently replaced.
        os.link(temporary, target)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--kind", choices=["checkpoint", "artifact"], default="artifact")
    parser.add_argument("--output-dir", type=Path, default=repository_root()/"models/weights")
    parser.add_argument("--confirm-large-download", action="store_true", help="Explicitly accept downloads over 100 MiB")
    args = parser.parse_args(argv)
    try:
        item = model_record(args.model)[args.kind]
        if item["bytes"] > 100 * 1024**2 and not args.confirm_large_download:
            parser.error("This file exceeds 100 MiB; pass --confirm-large-download to proceed")
        path = download(args.model, args.kind, args.output_dir)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Download failed: {error}\n")
    print(f"Verified: {path.name}. Checkpoint files are not deserialized by this utility.")


if __name__ == "__main__":
    main()
