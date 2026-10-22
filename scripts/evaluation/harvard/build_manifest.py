"""Build a Harvard-720 manifest with another espeak voice (frontend-mismatch test).

Same sentences, ids and order as ``results/manifests/harvard720.json``; only ``phonemes`` and
``phoneme_ids`` are rebuilt with ``banhmi_phonemize`` (the frontend that built the en-us
manifest) for ``--voice``. Before writing, the script rebuilds the en-us manifest the same
way and stops unless it equals the stored one, so that the only difference is the voice.

    python scripts/evaluation/harvard/build_manifest.py --voice en \
        --output results/manifests/harvard720_espeak_en.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build(rows, voice):
    from banhmi_phonemize import phoneme_ids_espeak, phonemize_espeak

    out = []
    for r in rows:
        ph = [p for sentence in phonemize_espeak(r["text"], voice) for p in sentence]
        out.append({**r, "phonemes": ph, "phoneme_ids": phoneme_ids_espeak(ph)})
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default=Path("results/manifests/harvard720.json"), type=Path)
    p.add_argument("--base-voice", default="en-us")
    p.add_argument("--voice", required=True)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args(argv)
    if a.output.exists():
        p.error("output exists; choose a new path")
    base = json.loads(a.base.read_text(encoding="utf-8"))
    rebuilt = build(base, a.base_voice)
    if rebuilt != base:
        bad = [r["idx"] for r, b in zip(rebuilt, base) if r != b]
        raise SystemExit(f"rebuilding the base manifest with {a.base_voice!r} does not reproduce it: {bad[:10]}")
    new = build(base, a.voice)
    changed = sum(n["phoneme_ids"] != b["phoneme_ids"] for n, b in zip(new, base))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(new, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{a.base} rebuilt with {a.base_voice!r}: identical; {a.voice!r} changes the phoneme IDs of "
          f"{changed}/{len(new)} sentences -> {a.output} ({hashlib.sha256(a.output.read_bytes()).hexdigest()[:16]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
