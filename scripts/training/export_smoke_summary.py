"""Write a commit-safe summary of smoke reports (no host paths, no tracebacks).

    python scripts/training/export_smoke_summary.py RUN_DIR [RUN_DIR ...] --output docs/training/training-smoke-results.json

Raw ``smoke_report.json`` files stay in local run storage.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_PATHLIKE = re.compile(r"(/[^\s'\"]+)+")


def _scrub(value):
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    if isinstance(value, str):
        return _PATHLIKE.sub(lambda m: "<path>/" + Path(m.group(0)).name, value)
    return value


def summarize(run_dir: Path) -> dict:
    report = json.loads((run_dir / "smoke_report.json").read_text(encoding="utf-8"))
    env = report.get("environment", {})
    out = {
        "run": run_dir.name,
        "model_id": report["model_id"],
        "config": Path(report["config"]).name,
        "config_sha256": report["config_sha256"],
        "smoke_overrides": report["smoke_overrides"],
        "data": report.get("data"),
        "summary": report["summary"],
        "checks": _scrub(report["checks"]),
        "optimizer_updates": {
            "phase_a": report.get("phase_a", {}).get("updates"),
            "phase_b": report.get("phase_b", {}).get("updates"),
            "total": report.get("total_optimizer_updates"),
        },
        "phase_a_final_metrics": report.get("phase_a", {}).get("metrics"),
        "phase_a_parameter_changes": report.get("phase_a", {}).get("parameter_changes"),
        "repository_commit": env.get("repository_commit"),
        "repository_dirty": env.get("repository_dirty"),
        "classification": report.get("classification"),
    }
    if "error" in report:
        out["error"] = _scrub({"type": report["error"]["type"], "message": report["error"]["message"][:600]})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    data = {"classification": "functional smoke tests; outputs are not research results",
            "runs": [summarize(r) for r in args.runs]}
    args.output.write_bytes((json.dumps(data, indent=2) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
