"""Prepare three Piper models on N100, then publish a separate demo catalog.

Existing model graphs, reports and the default catalog are never overwritten.
Run in a child process; failure leaves diagnostics and no active catalog.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/evaluation/edge"))

from hifimobinet import common97, n100_demo as demo
import openvino_suite as suite


def activate(bundle, work, catalog_path):
    """Only validated, hash-checked preparations can become active INT8 entries."""
    bundle, work = Path(bundle).resolve(), Path(work).resolve()
    work.relative_to(bundle)
    catalog = demo.read(ROOT / "demo/n100_models.json")
    manifest = suite.manifest(bundle)
    if {m["id"] for m in catalog["models"]} != {m["id"] for m in manifest["models"]}:
        raise ValueError("catalog/manifest model mismatch")
    for item in catalog["models"]:
        if item["id"] not in common97.MODEL_IDS:
            continue
        directory = work / item["id"]
        prep = suite.read(directory / "preparation.json")
        original = next(m for m in manifest["models"] if m["id"] == item["id"])
        if (prep["model"] != original or prep["script_sha256"] != suite.sha(suite.__file__)
                or prep["manifest_sha256"] != suite.sha(bundle / "models.json")
                or prep["environment"]["packages"] != catalog["expected_packages"]):
            raise ValueError("preparation does not match current code/source/runtime")
        item["artifacts"]["int8"] = []
        for record in prep["artifacts"]["int8"]["files"]:
            path = suite.verify_file(directory, record)
            item["artifacts"]["int8"].append({"file": path.relative_to(bundle).as_posix(), "sha256": record["sha256"]})
        item["int8_preparation"] = {"file": (directory / "preparation.json").relative_to(bundle).as_posix(),
                                    "sha256": suite.sha(directory / "preparation.json")}
        item["int8_recipe"] = common97.RECIPE
        item["quality_status"] = prep["quality_status"]
        demo.check_common97_preparation(prep, item)
    # Validate all retained FP32 files and the frontend before activation.
    demo.preflight(bundle, catalog)
    catalog["common97"] = {**common97.identity(), "work": work.relative_to(bundle).as_posix(),
                           "activated_utc": suite.utc(), "n100_benchmark": "not run by this command"}
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    demo.write_new(catalog_path, catalog)
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=demo.bundle_path())
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=ROOT / ".local/n100-common97-models.json")
    args = parser.parse_args()
    if args.catalog.exists():
        parser.error("active catalog already exists; use a new --catalog path for another experiment")
    if args.work.exists():
        parser.error("work directory already exists; use a new path to preserve earlier evidence")
    args.work.resolve().relative_to(args.bundle.resolve())
    demo.preflight(args.bundle, demo.read(ROOT / "demo/n100_models.json"))
    suite.prepare(args.bundle, args.work, "piper")
    activate(args.bundle, args.work, args.catalog)
    print("COMMON97 READY: three Piper models prepared and local catalog activated.", flush=True)
    print(f"Catalog: {args.catalog}\nEvidence/WAVs: {args.work}", flush=True)


if __name__ == "__main__":
    main()
