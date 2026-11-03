"""Sequential Piper Low FP32 evaluation in the existing WSL environments.

Creates a new evidence directory; refuses to overwrite or silently resume a run.
Every subprocess has a separate log and exit status. Raw WAVs/graphs stay in WSL;
--publish-dir receives small records and logs after each stage. Never starts training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import wave
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABEL = "piper-low-no-vits2-cpn"
CONFIG = Path("configs/training/piper-low-no-vits2-cpn.yaml")
MANIFEST = Path("results/manifests/harvard720.json")
TRAIN_RUN = Path("training_output/piper-low-no-vits2-cpn-2026-10-30")
CONTROLLER = Path("training_output/piper-low-sequence-2026-10-30")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=1, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def check_synth(path, manifest, sample_rate, matched=False):
    record = read(path)
    rows = record["rows"]
    if len(rows) != 720 or [(r["idx"], r["text"]) for r in rows] != [(r["idx"], r["text"]) for r in manifest]:
        raise ValueError(f"{path}: incomplete or mismatched Harvard manifest")
    if record["protocol"]["sample_rate"] != sample_rate:
        raise ValueError(f"{path}: incorrect sample rate")
    for row in rows:
        wav = Path(record["wav_dir"]) / row["wav"]
        if not row["finite"] or sha256(wav) != row["wav_sha256"]:
            raise ValueError(f"{wav}: non-finite output or changed bytes")
        with wave.open(str(wav), "rb") as audio:
            if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (sample_rate, 1, 2):
                raise ValueError(f"{wav}: incorrect PCM format")
            if audio.getnframes() == 0 or abs(audio.getnframes() / sample_rate - row["audio_duration_s"]) > 1e-9:
                raise ValueError(f"{wav}: incorrect duration")
        if matched and (not row["parity"]["passed"] or not row["pytorch_wav_matches_reference"]):
            raise ValueError(f"{wav}: ONNX parity / reference determinism failed")
    return {"status": "passed", "n": len(rows), "sample_rate": sample_rate,
            "synth_sha256": sha256(path), "matched_noise_parity_required": matched}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--publish-dir", type=Path, required=True)
    parser.add_argument("--train-python", default="/home/capstone/hifimobinet-train/venv/bin/python")
    parser.add_argument("--score-python", default="/home/capstone/env/bin/python")
    a = parser.parse_args()
    os.chdir(ROOT)
    if a.out_dir.exists():
        parser.error("output directory exists; choose a new session directory")
    if a.publish_dir.exists() and any(a.publish_dir.iterdir()):
        parser.error("publication directory exists and is not empty")
    a.out_dir.mkdir(parents=True)
    a.publish_dir.mkdir(parents=True, exist_ok=True)
    state = {"started_utc": utc(), "status": "running", "pid": os.getpid(), "cwd": str(ROOT),
             "raw_artifacts": str(a.out_dir.resolve()), "stages": [],
             "scope": "Low PyTorch FP32 + ONNX FP32, Harvard720, WER/CER/UTMOSv2, PC/WSL timing; Medium control",
             "not_run": ["INT8/PTQ", "N100 benchmark", "human MOS", "Piper High training"]}
    env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HIFIMOBINET_HOME=str(ROOT), OMP_NUM_THREADS="12", MKL_NUM_THREADS="12")

    def publish():
        write(a.out_dir / "status.json", state)
        lines = ["PIPER LOW - HARVARD720 / ONNX FP32", f"Status: {state['status']}",
                 f"Current stage: {state.get('current_stage')}", f"Raw artifacts (WSL): {a.out_dir.resolve()}",
                 "Native Low WAV: 16000 Hz; Medium control WAV: 22050 Hz.",
                 "RTF measurements: workstation CPU/WSL, not Intel N100.",
                 "No INT8/PTQ or High training in this session. Human listening is still needed.", ""]
        lines += [f"{s['name']}: {s['status']}" for s in state["stages"]]
        for label in ("pytorch", "onnx-fp32", "medium-control"):
            score = a.out_dir / label / "score_record.json"
            if score.exists():
                lines += ["", label + ":", json.dumps(read(score)["summary"], indent=2)]
        if state.get("error"):
            lines += ["", state["error"]]
        (a.out_dir / "RESULTS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        for f in a.out_dir.rglob("*"):
            if f.is_file() and f.suffix in {".json", ".jsonl", ".log", ".txt", ".py", ".yaml"}:
                target = a.publish_dir / f.relative_to(a.out_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, target)

    def stage(name, command, threads=12):
        command = list(map(str, command))
        row = {"name": name, "command": command, "started_utc": utc(), "status": "running",
               "threads_env": threads, "log": f"logs/{name}.log", "load_average_start": list(os.getloadavg())}
        state["stages"].append(row)
        state["current_stage"] = name
        publish()
        logfile = a.out_dir / row["log"]
        logfile.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{utc()}] START {name}", flush=True)
        started = time.monotonic()
        with logfile.open("xb") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     env=dict(env, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads)))
            row["pid"] = child.pid
            publish()
            code = child.wait()
        row.update(exit_code=code, finished_utc=utc(), wall_time_s=time.monotonic() - started,
                   status="completed" if code == 0 else "failed", log_sha256=sha256(logfile))
        publish()
        if code:
            raise RuntimeError(f"{name} exited {code}; see {logfile}")
        print(f"[{utc()}] FINISHED {name}", flush=True)

    try:
        validation = read(CONTROLLER / "validation_low.json")
        controller = read(CONTROLLER / "status.json")
        selected = validation["selected_checkpoint"]
        if not validation["valid"] or controller["runs"]["low"]["state"] != "completed":
            raise ValueError("Low training has not completed validated selection")
        checkpoint = TRAIN_RUN / "checkpoints" / selected["file"]
        if sha256(checkpoint) != selected["sha256"]:
            raise ValueError("selected checkpoint hash changed")
        state["selected_checkpoint"] = selected
        manifest = read(MANIFEST)
        if len(manifest) != 720:
            raise ValueError("expected Harvard720")
        for f in [CONFIG, MANIFEST, Path("results/manifests/canonical_split.json"),
                  CONTROLLER / "validation_low.json", CONTROLLER / "training_report.json", CONTROLLER / "status.json",
                  TRAIN_RUN / "run_record.json"]:
            target = a.out_dir / "inputs" / f
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, target)
        files = [Path(__file__).relative_to(ROOT)]
        for directory in ["scripts/evaluation/harvard", "scripts/evaluation/onnx"]:
            files += list(Path(directory).glob("*.py"))
        files += [Path("scripts/evaluation/count_parameters.py")]
        for f in files:
            target = a.out_dir / "source" / f
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, target)
        write(a.out_dir / "provenance.json", {"utc": utc(), "checkpoint": str(checkpoint),
              "checkpoint_sha256": sha256(checkpoint), "scripts": {str(f): sha256(f) for f in files},
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
              "environment_overrides": {k: env[k] for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"]}})
        (a.out_dir / "source" / "working-tree-diff.txt").write_text(
            subprocess.check_output(["git", "diff", "--", "scripts/evaluation"], text=True), encoding="utf-8")
        h = Path("scripts/evaluation/harvard")
        o = Path("scripts/evaluation/onnx")
        native = a.out_dir / "pytorch"
        control = a.out_dir / "medium-control"
        onnx_quality = a.out_dir / "onnx-fp32"
        graphs = a.out_dir / "graphs"

        def synth(name, config, ckpt, directory):
            stage(name, [a.train_python, h / "synthesize.py", "--config", config, "--checkpoint", ckpt,
                         "--manifest", MANIFEST, "--wav-dir", directory / "wavs", "--output", directory / "synth.json"])

        synth("01_low_pytorch", CONFIG, checkpoint, native)
        write(native / "validation.json", check_synth(native / "synth.json", manifest, 16000))
        stage("02_parameter_count", [a.train_python, "scripts/evaluation/count_parameters.py", "--checkpoint",
                                    f"Piper_Low_no_VITS2_cpn={checkpoint}", "--output", a.out_dir / "parameters.json"])
        medium_ckpt = Path("training_output/piper-no-vits2-cpn/checkpoints/best-epoch=1403-val_loss_mel=19.7719.ckpt")
        synth("03_medium_pytorch_control", "configs/training/piper-no-vits2-cpn.yaml", medium_ckpt, control)
        write(control / "validation.json", check_synth(control / "synth.json", manifest, 22050))
        stage("04_onnx_export_parity", [a.score_python, o / "export.py", "--label", LABEL, "--config", CONFIG,
                                      "--checkpoint", checkpoint, "--out-dir", graphs])
        if read(graphs / "export_record.json")["status"] != "passed":
            raise ValueError("ONNX export parity gate did not pass")
        stage("05_onnx_matched_noise_720", [a.score_python, o / "synthesize.py", "--precision", "fp32",
              "--config", CONFIG, "--checkpoint", checkpoint, "--graph", graphs / f"{LABEL}_fp32_matched_noise.onnx",
              "--reference-synth", native / "synth.json", "--wav-dir", onnx_quality / "wavs",
              "--output", onnx_quality / "synth.json"])
        write(onnx_quality / "validation.json", check_synth(onnx_quality / "synth.json", manifest, 16000, matched=True))
        medium_graph = Path("training_output/onnx-ptq-v2/piper-no-vits2-cpn/piper-no-vits2-cpn_fp32.onnx")
        medium_export = read(medium_graph.parent / "export_record.json")
        if medium_export["status"] != "passed" or sha256(medium_graph) != medium_export["graph"]["sha256"]:
            raise ValueError("Medium round2 graph differs from passed export record")
        write(a.out_dir / "medium-onnx-source.json", medium_export)
        for threads in (2, 12):
            for repeat in ("A", "B"):
                order = [("low", graphs / f"{LABEL}_fp32.onnx"), ("medium", medium_graph)]
                if repeat == "B":
                    order.reverse()
                for label, graph in order:
                    name = f"06_timing_t{threads}_{repeat}_{label}"
                    output = a.out_dir / "timing" / f"t{threads}_{repeat}_{label}.json"
                    stage(name, [a.score_python, o / "time_graphs.py", "--graph", graph, "--threads", threads,
                                 "--round", repeat, "--output", output], threads=threads)
                    timed = read(output)
                    if timed["summary"]["finite"] != 720 or timed["summary"]["n"] != 720:
                        raise ValueError(f"{name}: incomplete or non-finite timing run")
        for label, directory in [("low-pytorch", native), ("low-onnx", onnx_quality), ("medium-control", control)]:
            stage("07_score_" + label, [a.score_python, h / "score.py", "--synth", directory / "synth.json",
                  "--output", directory / "results.json", "--record", directory / "score_record.json", "--cer",
                  "--progress", directory / "score_progress.jsonl"])
            if len(read(directory / "results.json")) != 720:
                raise ValueError(f"{label}: incomplete scores")
        for label, candidate, reference in [("low-vs-medium", native, control), ("onnx-vs-pytorch", onnx_quality, native)]:
            stage("08_compare_" + label, [a.score_python, h / "compare.py", "--candidate", candidate / "results.json",
                  "--label", "candidate", "--no-historical", "--extra-reference", f"reference={reference / 'results.json'}",
                  "--metrics", "utmosv2,wer", "--output", a.out_dir / f"{label}.json"])
        if sha256(checkpoint) != selected["sha256"]:
            raise ValueError("checkpoint changed during evaluation")
        state.update(status="completed", finished_utc=utc(), current_stage=None,
                     note="Measured metrics complete; human listening and N100 evaluation remain separate. High is not started.")
    except BaseException:
        state.update(status="failed", finished_utc=utc(), error=traceback.format_exc())
        raise
    finally:
        publish()
        files = sorted(f for f in a.publish_dir.rglob("*") if f.is_file() and f.name != "SHA256SUMS")
        (a.publish_dir / "SHA256SUMS").write_text("".join(f"{sha256(f)}  {f.relative_to(a.publish_dir).as_posix()}\n" for f in files), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
