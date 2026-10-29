"""Print the Low launch command by default; training needs explicit opt-in.

Only use --start --confirm-training AFTER the user has approved training.
This opt-in is not a replacement for the user's confirmation in conversation.
Each plan uses the existing controller's preflight, evidence and recovery rules.
Run Low on both GPUs, then complete its evaluation/ONNX work before launching
High explicitly. This launcher never advances from Low to High automatically.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=("low", "high"), default="low")
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--confirm-training", action="store_true")
    args = parser.parse_args(argv)
    if args.start and not args.confirm_training:
        parser.error("training requires the user's approval, then --start --confirm-training")
    if args.start and os.name != "posix":
        parser.error("launch from the Linux/WSL training checkout, not Windows")
    tiers = (args.run,)
    commands = []
    for tier in tiers:
        plan_path = Path(f"configs/training/sequences/piper-{tier}-2026-10-30.json")
        plan = json.loads((ROOT / plan_path).read_text(encoding="utf-8"))
        run = plan["runs"][0]
        command = [sys.executable, "scripts/training/run_sequence.py", "start", "--plan", plan_path.as_posix()]
        print(f"{tier}: {run['model_id']} -> {run['run_dir']}", flush=True)
        print(shlex.join(command), flush=True)
        commands.append(command)
    if not args.start:
        print("PREPARED / AWAITING USER CONFIRMATION. No training process started.")
        return 0
    for command in commands:
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
