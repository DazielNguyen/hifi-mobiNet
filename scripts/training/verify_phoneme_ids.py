"""Re-phonemize dataset.jsonl texts with the original frontend and compare IDs.

    python scripts/training/verify_phoneme_ids.py --dataset path/to/dataset.jsonl [--limit N]

Mirrors banhmi_train.preprocess.worker.process_utterance: phonemize_espeak on
the (cased) text with voice en-us, flatten sentences, phoneme_ids_espeak.
The casing used by the original preprocessing run was not recorded, so the
default ("ignore") is tested first and the other choices are reported for any
mismatching row. Writes a JSON report to stdout. Needs banhmi_phonemize.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CASINGS = {"ignore": lambda s: s, "lower": str.lower, "upper": str.upper, "casefold": str.casefold}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    from banhmi_phonemize import phoneme_ids_espeak, phonemize_espeak

    def encode(text: str, casing: str):
        sentences = phonemize_espeak(CASINGS[casing](text), "en-us")
        return phoneme_ids_espeak([ph for sentence in sentences for ph in sentence])

    raw = args.dataset.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    matched, mismatches = 0, []
    for row in rows:
        if encode(row["text"], "ignore") == row["phoneme_ids"]:
            matched += 1
            continue
        alternatives = [c for c in CASINGS if c != "ignore" and encode(row["text"], c) == row["phoneme_ids"]]
        mismatches.append({"audio_norm_path": row["audio_norm_path"], "matching_casings": alternatives})
    print(json.dumps({
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "rows_checked": len(rows),
        "matched_with_default_casing": matched,
        "mismatches": mismatches,
        "status": "pass" if not mismatches else "fail",
    }, indent=2))


if __name__ == "__main__":
    main()
