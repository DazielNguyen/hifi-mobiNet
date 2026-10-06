"""Write docs/training/source-provenance.json for the training additions.

    python scripts/training/write_source_provenance.py

Lists each new training file with its SHA-256 and what it derives from.
Vendored reference files themselves are inventoried in docs/source-map.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIPER = "rhasspy/piper 73c04d81d5590ecc46e522de3601ce7fb29fc2be"

DERIVATION = {
    "src/hifimobinet/training/baseline_module.py": (
        "adapted", ["vendor/banhmi/vits/training.py"],
        "Loss/optimizer/scheduler/skip/health-gate logic kept; Vocos/F0/MRD removed; data injected"),
    "src/hifimobinet/training/piper_module.py": (
        "adapter_subclass", ["vendor/piper/vits/lightning.py"],
        f"Subclasses upstream VitsModel ({PIPER}); training_step_g body copied with added logging"),
    "src/hifimobinet/training/harness.py": (
        "adapted", ["vendor/banhmi/vits/training.py"],
        "Non-finite skip and health check factored out of BanhmiTTS VitsModel; RNG save/restore and "
        "Lightning-1.7 scheduler hook are new"),
    "src/hifimobinet/training/train.py": (
        "adapted", ["vendor/banhmi/train.py", "vendor/piper/__main__.py"],
        "Checkpoint rule and seeding follow both entry points; explicit model ID/config/split are new"),
    "src/hifimobinet/training/data.py": (
        "new", ["vendor/banhmi/vits/training.py"],
        "Split/length-cache semantics follow BanhmiTTS; loaders call unmodified vendor Dataset/Collate"),
    "configs/training/baseline-resblock2-vits2.yaml": (
        "derived_values", ["configs/baseline-resblock2.json", "configs/original/default.yaml"], "Values only"),
    "configs/training/piper-no-vits2-cpn.yaml": (
        "derived_values", ["configs/piper-original.json", "vendor/piper/vits/lightning.py"], "Values only"),
    "configs/training/piper-no-vits2-cpn-upstream-harness.yaml": (
        "derived_values", ["configs/piper-original.json", "vendor/piper/vits/lightning.py"], "Values only"),
    "scripts/training/build_mas.sh": (
        "new", ["vendor/piper/build_monotonic_align.sh", "vendor/banhmi/vits/utils/monotonic_align/setup.py"],
        "Same output layout as upstream; compiled in a temporary directory"),
    "requirements/training-linux-cu130.txt": (
        "new", [], "Versions observed in the existing workstation training interpreter"),
}
GLOBS = ("src/hifimobinet/training/*.py", "configs/training/*.yaml", "scripts/training/*",
         "requirements/*.txt", "tests/test_training.py")


def main() -> None:
    files = sorted({p for g in GLOBS for p in ROOT.glob(g) if p.is_file()})
    entries = []
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        kind, sources, note = DERIVATION.get(rel, ("new", [], ""))
        entries.append({"file": rel, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "classification": kind, "derived_from": sources, "note": note})
    out = {"schema_version": 1,
           "scope": "Training additions of 2026-10-06; vendored references are in docs/source-map.json",
           "license": "Original code MIT (repository LICENSE); adapted parts retain upstream terms "
                      "(Piper MIT; BanhmiTTS author review pending per source-map)",
           "files": entries}
    (ROOT / "docs/training/source-provenance.json").write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
    print(f"{len(entries)} files")


if __name__ == "__main__":
    main()
