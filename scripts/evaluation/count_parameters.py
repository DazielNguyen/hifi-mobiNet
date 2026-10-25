"""Parameter counts and sizes of the Harvard-720 models, in one record.

New checkpoints are read with ``torch.load(weights_only=True)``; nothing is built or run.
Counts follow the Q05 definition (``docs/evidence/claim-evidence-2026-10-07``,
``scripts/q05/run_q05_model.py``) so that new and historical rows are comparable:
generator (``model_g``) parameters without the posterior encoder ``enc_q``, in
weight-norm form (``weight_g`` and ``weight_v`` both counted), buffers excluded. That
count still contains the stochastic duration predictor's training-only path, so each new
row also gives the subset that inference uses:
- ``dp.post_*`` (posterior path of the SDP) is used only in training;
- ``dp.flows.1`` (the first ConvFlow) is dropped in reverse mode ("remove a useless vflow").

Historical rows start from the stored Q05 reports and ``models/manifest.json``. When a
historical checkpoint is given with ``--historical-checkpoint``, it must match the manifest's
SHA-256 and its recount must equal the Q05 report; the row then also gets the inference and
decoder counts. Historical checkpoints pickle ``pathlib.PosixPath`` in their hyperparameters,
so they are read with that one class allowlisted.

    python scripts/evaluation/count_parameters.py \
        --checkpoint Piper_no_VITS2_cpn=<run>/checkpoints/best-epoch=...ckpt [--checkpoint ...] \
        [--historical-checkpoint baseline-resblock2=<path> ...] \
        --output results/model_parameters.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
Q05 = REPO / "docs/evidence/claim-evidence-2026-10-07/source-records/q05"
HISTORICAL = {  # models/manifest.json id -> Q05 report label
    "baseline-resblock2": "baseline",
    "parallel-ir": "mrf",
    "sequential-ir": "seq",
    "piper-original": "piper",
}
BUFFER_SUFFIXES = ("._n_channels", ".hann_window")
TRAINING_ONLY_PREFIXES = ("model_g.dp.post_pre.", "model_g.dp.post_proj.", "model_g.dp.post_convs.",
                          "model_g.dp.post_flows.")
DROPPED_IN_REVERSE_PREFIXES = ("model_g.dp.flows.1.",)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def count_checkpoint(path: Path, allow_posix_path: bool = False) -> dict:
    import pathlib

    import torch

    with torch.serialization.safe_globals([pathlib.PosixPath] if allow_posix_path else []):
        ck = torch.load(path, map_location="cpu", weights_only=True)
    state = ck["state_dict"]
    by_module: dict[str, int] = defaultdict(int)
    buffers = {}
    unused = {"sdp_training_only_path": 0, "sdp_flow_dropped_in_reverse": 0}
    for key, tensor in state.items():
        if key.endswith(BUFFER_SUFFIXES):
            buffers[key] = tensor.numel()
            continue
        if tensor.dtype != torch.float32:
            raise SystemExit(f"unexpected non-float32 entry {key} ({tensor.dtype}); classify it first")
        parts = key.split(".")
        by_module[".".join(parts[:2]) if parts[0] == "model_g" else parts[0]] += tensor.numel()
        if key.startswith(TRAINING_ONLY_PREFIXES):
            unused["sdp_training_only_path"] += tensor.numel()
        elif key.startswith(DROPPED_IN_REVERSE_PREFIXES):
            unused["sdp_flow_dropped_in_reverse"] += tensor.numel()
    if not unused["sdp_training_only_path"] or not unused["sdp_flow_dropped_in_reverse"]:
        raise SystemExit(f"{path}: expected stochastic-duration-predictor keys not found")
    generator = sum(n for m, n in by_module.items() if m.startswith("model_g."))
    q05 = generator - by_module.get("model_g.enc_q", 0)
    inference = q05 - sum(unused.values())
    return {
        "checkpoint": path.name,
        "checkpoint_sha256": sha256(path),
        "checkpoint_bytes": os.path.getsize(path),
        "checkpoint_contents": "full Lightning training state (generator, discriminators, optimizer states)"
        if "optimizer_states" in ck else "state dict without optimizer states",
        "epoch": int(ck["epoch"]),
        "parameters_by_module": dict(sorted(by_module.items())),
        "buffers_excluded": buffers,
        "generator_parameters": generator,
        "generator_parameters_excluding_posterior": q05,
        "not_used_at_inference": unused,
        "inference_parameters": inference,
        "decoder_parameters": by_module.get("model_g.dec", 0),
        "discriminator_parameters": {m: n for m, n in by_module.items() if m.startswith("model_d")},
        "fp32_bytes_generator_excluding_posterior": 4 * q05,
        "fp32_bytes_inference_parameters": 4 * inference,
        "onnx": None,
    }


def historical_rows(recount: dict[str, Path]) -> dict:
    manifest = json.loads((REPO / "models/manifest.json").read_text(encoding="utf-8"))
    unknown = set(recount) - {m["id"] for m in manifest["models"]}
    if unknown:
        raise SystemExit(f"--historical-checkpoint ids not in models/manifest.json: {sorted(unknown)}")
    rows = {}
    for model in manifest["models"]:
        label = HISTORICAL[model["id"]]
        report_path = Q05 / f"REPORT.{label}.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        q05 = int(report["generator_parameters_excluding_posterior"])
        rows[model["id"]] = {
            "source": {"q05_report": report_path.relative_to(REPO).as_posix(), "q05_report_sha256": sha256(report_path),
                       "models_manifest": "models/manifest.json"},
            "checkpoint": model["checkpoint"]["filename"],
            "checkpoint_sha256": model["checkpoint"]["sha256"],
            "checkpoint_bytes": model["checkpoint"]["bytes"],
            "generator_parameters_excluding_posterior": q05,
            "q05_note": report.get("metadata_note"),
            "inference_parameters": None,
            "fp32_bytes_generator_excluding_posterior": 4 * q05,
            "onnx": {k: model["artifact"][k] for k in ("filename", "sha256", "bytes", "precision", "opset")},
        }
        if model["id"] in recount:
            counted = count_checkpoint(recount[model["id"]], allow_posix_path=True)
            if counted["checkpoint_sha256"] != model["checkpoint"]["sha256"]:
                raise SystemExit(f"{model['id']}: checkpoint SHA-256 does not match models/manifest.json")
            if counted["generator_parameters_excluding_posterior"] != q05:
                raise SystemExit(f"{model['id']}: recount {counted['generator_parameters_excluding_posterior']:,} "
                                 f"differs from the Q05 report {q05:,}")
            for key in ("epoch", "parameters_by_module", "buffers_excluded", "generator_parameters",
                        "not_used_at_inference", "inference_parameters", "decoder_parameters",
                        "discriminator_parameters", "fp32_bytes_inference_parameters"):
                rows[model["id"]][key] = counted[key]
            rows[model["id"]]["source"]["recounted_from_checkpoint"] = True
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    p.add_argument("--historical-checkpoint", action="append", default=[], metavar="ID=PATH",
                   help="recount a historical model (id from models/manifest.json) from its checkpoint")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args(argv)
    if a.output.exists():
        p.error("output exists; choose a new path")
    new = {}
    for item in a.checkpoint:
        label, _, path = item.partition("=")
        new[label] = count_checkpoint(Path(path))
    recount = {}
    for item in a.historical_checkpoint:
        model_id, _, path = item.partition("=")
        recount[model_id] = Path(path)
    historical = historical_rows(recount)
    all_recounted = len(recount) == len(historical)
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "new_static_count_of_stored_checkpoints_and_records",
        "definition": {
            "generator_parameters_excluding_posterior": "model_g parameters without enc_q, weight-norm form, buffers excluded (Q05 definition); includes the SDP training-only path",
            "inference_parameters": "generator_parameters_excluding_posterior minus dp.post_* (training only) and dp.flows.1 (dropped in reverse mode); weight-norm form",
            "fp32_bytes": "4 bytes per parameter; not a file size",
            "checkpoint_bytes": "file size of the full training checkpoint, not a deployable model size",
        },
        "new_models": new,
        "historical_models": historical,
        "limits": [
            "Counts are in weight-norm form; removing weight norm drops the weight_g entries and changes counts slightly.",
            "FP32 bytes are computed (4 bytes per parameter), not measured file sizes.",
            "Historical rows were recounted from their checkpoints (SHA-256 checked against models/manifest.json) "
            "and the recount equals the Q05 report." if all_recounted else
            "Historical counts without --historical-checkpoint are taken from the Q05 reports, not recounted here.",
        ],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    for label, row in new.items():
        print(f"{label}: {row['generator_parameters_excluding_posterior']:,} (Q05 definition), "
              f"{row['inference_parameters']:,} used at inference, decoder {row['decoder_parameters']:,}")
    for model_id, row in historical.items():
        if row.get("decoder_parameters") is not None:
            print(f"{model_id}: {row['generator_parameters_excluding_posterior']:,} (Q05 definition, equals the report), "
                  f"{row['inference_parameters']:,} used at inference, decoder {row['decoder_parameters']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
