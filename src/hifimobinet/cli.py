"""Local CLI. Model IDs come only from the checked-in manifest."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from .registry import ModelUnavailable, manifest, model_record, model_path, runtime_status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models", help="List artifact and local TTS status")
    verify = sub.add_parser("verify", help="Check a manifest model's local SHA-256")
    verify.add_argument("--model", required=True)
    tts = sub.add_parser("tts", help="Functional FP32 inference; not a benchmark")
    tts.add_argument("--model", required=True)
    tts.add_argument("--text", required=True)
    tts.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "models":
            for model in manifest()["models"]:
                ok, reason = runtime_status(model)
                print(json.dumps({"id":model["id"],"label":model["label"],"tts_available":ok,"reason":reason}))
        elif args.command == "verify":
            record = model_record(args.model)
            model_path(record)
            print(f"{record['id']}: SHA-256 verified")
        else:
            record = model_record(args.model)
            ok, reason = runtime_status(record)
            if not ok:
                raise ModelUnavailable(reason)
            from .inference import Voice
            audio = Voice(args.model).synthesize(args.text)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("xb") as stream:
                stream.write(audio)
            print("WAV written. New functional synthesis, not a historical research sample.")
    except (ModelUnavailable, ImportError, OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
