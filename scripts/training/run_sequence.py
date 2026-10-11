"""Run a sequence of long trainings one after the other, unattended (Linux/WSL).

    python scripts/training/run_sequence.py start  --plan configs/training/sequences/piper-ir-2026-10.json
    python scripts/training/run_sequence.py status --plan ...
    python scripts/training/run_sequence.py resume --plan ... --run parir --note "why"   # operator only
    python scripts/training/run_sequence.py validate --plan ... --run parir           # re-check a finished run

Run from the training checkout root with the training venv's Python, inside tmux
(docs/training/piper-ir-runbook.md). For each run in plan order the controller:

1. skips it if already ``completed``; stops if any run is ``failed`` or
   ``needs_recovery`` (a failed run never lets a later run start);
2. runs the preflight (Git clean, data/split/config hashes against the plan,
   environment versions, no other training process, GPU memory, free disk, empty
   run directory) and records it, including the compiled MAS extension's hash;
3. starts ``python -m hifimobinet.training.train`` from scratch and waits for it;
4. validates the finished run (exit code 0, budget reached, records consistent,
   every logged scalar finite, no traceback, checkpoint chosen by the validation
   rule) and writes the selected-checkpoint manifest; only a valid run is
   ``completed``.

A ``running`` entry whose process no longer exists becomes ``needs_recovery``:
the controller never starts a duplicate, never resets outputs and never resumes
by itself. ``resume`` continues an interrupted run from its ``last.ckpt`` (full
training state, the repository's ``--resume`` path) and records it.

Status is written atomically to ``<state_dir>/status.json``; ``controller.log``
holds the controller's own log. A lock file prevents two controllers.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path.cwd()
TERMINAL_BAD = ("failed", "needs_recovery")


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Controller:
    def __init__(self, plan_path: Path):
        self.plan_path = plan_path
        self.plan = json.loads(plan_path.read_text(encoding="utf-8"))
        self.state_dir = ROOT / self.plan["state_dir"]
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.state_dir / "status.json"
        self.log_path = self.state_dir / "controller.log"
        self.lock_handle = None

    # ------------------------------------------------------------- plumbing
    def log(self, message: str) -> None:
        line = f"{now()} {message}"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)

    def lock(self) -> None:
        import fcntl  # POSIX only; the controller runs under Linux/WSL
        self.lock_handle = open(self.state_dir / "controller.lock", "a+")
        try:
            fcntl.flock(self.lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another controller holds the lock; see `status`")
        self.lock_handle.seek(0)
        self.lock_handle.truncate()
        self.lock_handle.write(f"{os.getpid()} {now()}\n")
        self.lock_handle.flush()

    def load_status(self) -> Dict[str, Any]:
        if self.status_path.exists():
            return json.loads(self.status_path.read_text(encoding="utf-8"))
        return {"sequence_id": self.plan["sequence_id"], "plan": str(self.plan_path), "plan_sha256": sha256(self.plan_path),
                "created_utc": now(), "runs": {r["name"]: {"state": "pending", "attempts": []} for r in self.plan["runs"]}}

    def save_status(self, status: Dict[str, Any]) -> None:
        status["updated_utc"] = now()
        write_json_atomic(self.status_path, status)

    # ------------------------------------------------------------- preflight
    def git(self, *args: str) -> str:
        return subprocess.check_output(["git", "-c", "safe.directory=*", *args], cwd=ROOT).decode().strip()

    def preflight(self, run: Dict[str, Any], resume: bool) -> Dict[str, Any]:
        import torch
        import numpy
        import pytorch_lightning
        exp, pre = self.plan["expected"], self.plan["preflight"]
        args = dict(zip(self.plan["train_args"][::2], self.plan["train_args"][1::2]))
        dataset_dir = ROOT / args["--dataset-dir"]
        checks: Dict[str, Any] = {}

        def check(name, ok, **detail):
            checks[name] = {"passed": bool(ok), **detail}

        check("git_clean", self.git("status", "--porcelain") == "", commit=self.git("rev-parse", "HEAD"))
        check("dataset_jsonl_sha256", sha256(dataset_dir / "dataset.jsonl") == exp["dataset_jsonl_sha256"])
        check("dataset_config_sha256", sha256(dataset_dir / "config.json") == exp["dataset_config_sha256"])
        check("split_sha256", sha256(ROOT / args["--split"]) == exp["split_sha256"])
        from hifimobinet.training.config import load_config
        cfg = load_config(ROOT / run["config"])
        check("config_model_id", cfg.model_id == run["model_id"], config_sha256=cfg.sha256)
        env = {"python": sys.version.split()[0], "torch": torch.__version__,
               "pytorch_lightning": pytorch_lightning.__version__, "numpy": numpy.__version__}
        check("environment_versions", env == exp["environment"], observed=env)
        mas = sorted((ROOT / "vendor/edgetts/vits/monotonic_align/monotonic_align").glob("core*.so"))
        check("mas_extension_present", bool(mas), files=[{"file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in mas])
        others = [l for l in subprocess.run(["pgrep", "-af", "hifimobinet.training"], capture_output=True, text=True).stdout.splitlines()
                  if "run_sequence.py" not in l]
        check("no_other_training_process", not others, processes=others)
        try:
            smi = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total",
                                           "--format=csv,noheader,nounits"]).decode().strip().splitlines()
            gpus = [dict(zip(("index", "name", "used_mib", "total_mib"), [x.strip() for x in s.split(",")])) for s in smi]
            check("gpus_available", len(gpus) >= pre["gpus"] and all(int(g["used_mib"]) <= pre["max_gpu_memory_used_mib"] for g in gpus),
                  gpus=gpus)
        except (OSError, subprocess.CalledProcessError) as error:
            check("gpus_available", False, error=repr(error))
        free = shutil.disk_usage(ROOT).free / 2**30
        check("free_disk", free >= pre["min_free_disk_gib"], free_gib=round(free, 1))
        run_dir = ROOT / run["run_dir"]
        if resume:
            check("resume_checkpoint_present", (run_dir / "checkpoints/last.ckpt").is_file())
        else:
            check("run_dir_new", not run_dir.exists() or not any(run_dir.iterdir()))
            check("log_new", not (ROOT / run["log"]).exists())
        return {"utc": now(), "run": run["name"], "resume": resume, "passed": all(c["passed"] for c in checks.values()),
                "checks": checks}

    # ------------------------------------------------------------- run
    def launch(self, run: Dict[str, Any], status: Dict[str, Any], resume: bool, note: Optional[str] = None) -> int:
        entry = status["runs"][run["name"]]
        pre = self.preflight(run, resume)
        tag = f"{run['name']}_{'resume' if resume else 'start'}_{datetime.datetime.now():%Y%m%dT%H%M%S}"
        write_json_atomic(self.state_dir / f"preflight_{tag}.json", pre)
        if not pre["passed"]:
            failed = [k for k, v in pre["checks"].items() if not v["passed"]]
            self.log(f"[{run['name']}] preflight FAILED: {failed}; nothing started")
            entry.update(state="pending" if not resume else entry["state"], last_preflight_failure=failed)
            self.save_status(status)
            return 2
        cmd = [sys.executable, "-m", "hifimobinet.training.train", "--model-id", run["model_id"], "--config", run["config"],
               "--run-dir", run["run_dir"], *self.plan["train_args"]]
        if resume:
            cmd += ["--resume", str(Path(run["run_dir"]) / "checkpoints/last.ckpt")]
        log_path = ROOT / run["log"]
        attempt = {"kind": "resume" if resume else "start", "note": note, "command": cmd, "commit": pre["checks"]["git_clean"]["commit"],
                   "preflight": f"preflight_{tag}.json", "started_utc": now(),
                   "log_offset": log_path.stat().st_size if log_path.exists() else 0}
        with open(log_path, "a", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            attempt["pid"] = proc.pid
            entry["attempts"].append(attempt)
            entry["state"] = "running"
            self.save_status(status)
            self.log(f"[{run['name']}] started pid {proc.pid}: {' '.join(cmd)}")
            code = proc.wait()
        attempt.update(ended_utc=now(), exit_code=code)
        self.log(f"[{run['name']}] process exited with code {code}")
        self.finish(run, status, code)
        return code

    def finish(self, run: Dict[str, Any], status: Dict[str, Any], code: Optional[int]) -> None:
        entry = status["runs"][run["name"]]
        offset = entry["attempts"][-1].get("log_offset", 0) if entry["attempts"] else 0
        try:
            report = validate_run(ROOT, run, self.plan, code, log_offset=offset)
        except Exception as error:  # e.g. an unreadable checkpoint: recorded as a failure, nothing is skipped
            report = {"utc": now(), "run": run["name"], "valid": False, "failed_checks": ["validation_error"],
                      "error": repr(error), "selected_checkpoint": None}
        write_json_atomic(self.state_dir / f"validation_{run['name']}.json", report)
        if report["valid"]:
            write_json_atomic(ROOT / run["run_dir"] / "selected_checkpoint.json", report["selected_checkpoint"])
            entry.update(state="completed", completed_utc=now(), selected_checkpoint=report["selected_checkpoint"])
            self.log(f"[{run['name']}] COMPLETED; selected {report['selected_checkpoint']['file']}")
        else:
            entry.update(state="failed", failed_utc=now(), failure=report["failed_checks"])
            self.log(f"[{run['name']}] FAILED validation: {report['failed_checks']}")
        self.save_status(status)

    def reconcile(self, status: Dict[str, Any]) -> None:
        """A 'running' run without its process is an interruption, never restarted silently."""
        for name, entry in status["runs"].items():
            if entry["state"] != "running":
                continue
            pid = entry["attempts"][-1].get("pid") if entry["attempts"] else None
            if pid and pid_alive(pid):
                raise SystemExit(f"[{name}] is still running as pid {pid} without a controller; inspect before acting")
            entry.update(state="needs_recovery", interrupted_detected_utc=now())
            self.log(f"[{name}] was running but its process is gone: needs_recovery (use `resume` after inspection)")
        self.save_status(status)

    # ------------------------------------------------------------- commands
    def start(self) -> int:
        self.lock()
        status = self.load_status()
        self.save_status(status)
        self.log(f"controller start pid {os.getpid()} plan {self.plan_path} ({status['plan_sha256'][:12]})")
        self.reconcile(status)
        return self.run_remaining(status)

    def run_remaining(self, status: Dict[str, Any]) -> int:
        for run in self.plan["runs"]:
            state = status["runs"][run["name"]]["state"]
            if state == "completed":
                continue
            if state in TERMINAL_BAD:
                self.log(f"[{run['name']}] is {state}; the sequence stops here (later runs are not started)")
                return 1
            code = self.launch(run, status, resume=False)
            if status["runs"][run["name"]]["state"] != "completed":
                self.log(f"[{run['name']}] did not complete (exit {code}); later runs are not started")
                return 1
        self.final_report(status)
        self.log("sequence COMPLETED")
        return 0

    def resume(self, name: str, note: str) -> int:
        self.lock()
        status = self.load_status()
        self.reconcile(status)
        run = next(r for r in self.plan["runs"] if r["name"] == name)
        entry = status["runs"][name]
        if entry["state"] not in TERMINAL_BAD:
            raise SystemExit(f"[{name}] is {entry['state']}; only needs_recovery or failed runs can be resumed")
        earlier = [r["name"] for r in self.plan["runs"][: self.plan["runs"].index(run)] if status["runs"][r["name"]]["state"] != "completed"]
        if earlier:
            raise SystemExit(f"earlier runs are not completed: {earlier}")
        self.log(f"[{name}] operator resume from last.ckpt: {note}")
        self.launch(run, status, resume=True, note=note)
        if status["runs"][name]["state"] != "completed":
            return 1
        return self.run_remaining(status)

    def revalidate(self, name: str) -> int:
        self.lock()
        status = self.load_status()
        run = next(r for r in self.plan["runs"] if r["name"] == name)
        if status["runs"][name]["state"] == "running":
            raise SystemExit("run is marked running; use status/resume")
        code = status["runs"][name]["attempts"][-1].get("exit_code") if status["runs"][name]["attempts"] else None
        self.finish(run, status, code)
        return 0 if status["runs"][name]["state"] == "completed" else 1

    def final_report(self, status: Dict[str, Any]) -> None:
        runs = {}
        for run in self.plan["runs"]:
            v = json.loads((self.state_dir / f"validation_{run['name']}.json").read_text(encoding="utf-8"))
            runs[run["name"]] = {"model_id": run["model_id"], "run_dir": run["run_dir"], "attempts": status["runs"][run["name"]]["attempts"],
                                 "selected_checkpoint": v["selected_checkpoint"], "checkpoints": v["checkpoints"],
                                 "val_loss_mel": v["val_loss_mel_summary"], "run_record": v["run_record"]}
        report = {"sequence_id": self.plan["sequence_id"], "completed_utc": now(), "plan_sha256": status["plan_sha256"], "runs": runs,
                  "not_done": "Harvard, MOS, ONNX/PTQ and edge benchmarks are separate, later requests"}
        write_json_atomic(self.state_dir / "training_report.json", report)
        lines = [f"# Training report: {self.plan['sequence_id']}", "", f"Completed {report['completed_utc']}.", "",
                 "| Run | Model | Selected checkpoint | Epoch | val_loss_mel | SHA-256 |", "|---|---|---|---:|---:|---|"]
        for name, r in runs.items():
            s = r["selected_checkpoint"]
            lines.append(f"| {name} | {r['model_id']} | `{s['file']}` | {s['epoch']} | {s['val_loss_mel']:.4f} | `{s['sha256'][:16]}…` |")
        lines += ["", "Selection: lowest validation val_loss_mel (Config A rule). No held-out set was used.",
                  "Harvard, MOS, ONNX/PTQ and edge benchmarks are not part of this report."]
        (self.state_dir / "training_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.log("final training report written")

    def status_command(self) -> int:
        status = self.load_status()
        out = {"updated_utc": status.get("updated_utc"), "runs": {}}
        for run in self.plan["runs"]:
            e = status["runs"][run["name"]]
            last = e["attempts"][-1] if e["attempts"] else {}
            pid = last.get("pid")
            out["runs"][run["name"]] = {"state": e["state"], "pid": pid, "pid_alive": bool(pid and pid_alive(pid)),
                                        "started_utc": last.get("started_utc"), "ended_utc": last.get("ended_utc"),
                                        "exit_code": last.get("exit_code"), "progress": progress(ROOT / run["log"]),
                                        "selected": (e.get("selected_checkpoint") or {}).get("file"), "failure": e.get("failure")}
        print(json.dumps(out, indent=1))
        return 0


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def progress(log: Path) -> Optional[str]:
    if not log.exists():
        return None
    with open(log, "rb") as f:
        f.seek(max(0, log.stat().st_size - 20000))
        tail = f.read().decode("utf-8", "ignore").replace("\r", "\n")
    epochs = re.findall(r"Epoch (\d+):\s+(\d+)%[^\n]*?\|\s*(\d+)/(\d+)", tail)
    return f"epoch {epochs[-1][0]} ({epochs[-1][2]}/{epochs[-1][3]})" if epochs else None


# ----------------------------------------------------------------- validation
def check_log_text(text: str, max_epochs: int) -> Dict[str, Any]:
    return {"budget_reached": f"`max_epochs={max_epochs}` reached" in text, "traceback": "Traceback (most recent call last)" in text}


def select_checkpoint(values: List[float], files: List[str]) -> Dict[str, Any]:
    """Lowest val_loss_mel epoch (first on ties, as ModelCheckpoint keeps the earlier one) and its file."""
    best_epoch = min(range(len(values)), key=lambda i: (values[i], i))
    matches = [f for f in files if f.startswith(f"best-epoch={best_epoch}-")]
    return {"epoch": best_epoch, "val_loss_mel": values[best_epoch], "file": matches[0] if len(matches) == 1 else None}


def read_scalars(run_dir: Path) -> Dict[str, List[tuple]]:
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    import numpy as np
    series: Dict[str, Dict[int, float]] = {}
    for f in sorted((run_dir / "lightning_logs").rglob("events.out.tfevents*")):
        for event in EventFileLoader(str(f)).Load():
            for v in event.summary.value:
                if v.HasField("simple_value"):
                    x = v.simple_value
                elif v.HasField("tensor") and v.tensor.dtype == 1 and not v.tensor.tensor_shape.dim:
                    x = float(np.frombuffer(v.tensor.tensor_content, np.float32)[0]) if v.tensor.tensor_content else float(v.tensor.float_val[0])
                else:
                    continue
                series.setdefault(v.tag, {})[event.step] = x  # a resumed run's later point replaces the same step
    return {tag: sorted(points.items()) for tag, points in series.items()}


def validate_run(root: Path, run: Dict[str, Any], plan: Dict[str, Any], exit_code: Optional[int],
                 log_offset: int = 0) -> Dict[str, Any]:
    """Valid only if every check passes. Log checks read the last attempt's part of the
    log (from ``log_offset``), so an earlier interrupted attempt cannot hide or fake the result."""
    import torch
    budget = plan["budget"]
    run_dir = root / run["run_dir"]
    checks: Dict[str, Any] = {}

    def check(name, ok, **detail):
        checks[name] = {"passed": bool(ok), **detail}

    check("exit_code_zero", exit_code == 0, exit_code=exit_code)
    text = ""
    if (root / run["log"]).exists():
        with open(root / run["log"], "rb") as f:
            f.seek(log_offset)
            text = f.read().decode("utf-8", "ignore")
    lg = check_log_text(text, budget["max_epochs"])
    check("log_budget_reached", lg["budget_reached"])
    check("log_no_traceback", not lg["traceback"])
    records = [run_dir / "run_record.json", *sorted(run_dir.glob("resume_record_*.json"))]
    rr = json.loads(records[0].read_text(encoding="utf-8")) if records[0].exists() else {}
    exp = plan["expected"]
    check("run_record_consistent", rr.get("model_id") == run["model_id"] and rr.get("split_sha256") == exp["split_sha256"]
          and rr.get("config_sha256") == sha256(root / run["config"])
          and rr.get("dataset_jsonl_sha256") == exp["dataset_jsonl_sha256"] and rr.get("resume") is None
          and (rr.get("environment") or {}).get("repository_dirty") is False,
          resume_records=[p.name for p in records[1:]])
    scalars = read_scalars(run_dir)
    val = [v for _, v in scalars.get("val_loss_mel", [])]
    nonfinite = {t: sum(1 for _, x in pts if not math.isfinite(x)) for t, pts in scalars.items()}
    check("validation_every_epoch", len(val) == budget["max_epochs"], points=len(val))
    check("all_scalars_finite", bool(scalars) and not any(nonfinite.values()), nonfinite={t: n for t, n in nonfinite.items() if n})
    ck_dir = run_dir / "checkpoints"
    files = sorted(p.name for p in ck_dir.glob("*.ckpt")) if ck_dir.exists() else []
    last_ok = False
    if (ck_dir / "last.ckpt").exists():
        last = torch.load(ck_dir / "last.ckpt", map_location="cpu", weights_only=True)
        expected_steps = budget["batches_per_epoch_per_rank"] * budget["max_epochs"] * budget["optimizer_steps_per_batch"]
        last_ok = last["epoch"] == budget["max_epochs"] - 1 and last["global_step"] == expected_steps
        check("last_checkpoint_at_budget", last_ok, epoch=last["epoch"], global_step=last["global_step"], expected_global_step=expected_steps)
        del last
    else:
        check("last_checkpoint_at_budget", False, missing="last.ckpt")
    selected = None
    if val and all(math.isfinite(v) for v in val):
        sel = select_checkpoint(val, files)
        if sel["file"]:
            c = torch.load(ck_dir / sel["file"], map_location="cpu", weights_only=True)
            ok = c["epoch"] == sel["epoch"] and f"val_loss_mel={sel['val_loss_mel']:.4f}" in sel["file"]
            path = ck_dir / sel["file"]
            selected = {"file": sel["file"], "bytes": path.stat().st_size, "sha256": sha256(path), "epoch": c["epoch"],
                        "global_step": c["global_step"], "val_loss_mel": sel["val_loss_mel"],
                        "selection_rule": "lowest validation val_loss_mel over all epochs (Config A rule; top-3 + last kept)",
                        "contents": "full Lightning training state; load with torch.load(weights_only=True)"}
            del c
            check("selected_checkpoint_matches_minimum", ok, epoch=sel["epoch"], val_loss_mel=sel["val_loss_mel"])
        else:
            check("selected_checkpoint_matches_minimum", False, epoch=sel["epoch"], files=files)
    else:
        check("selected_checkpoint_matches_minimum", False, reason="no finite val_loss_mel series")
    summary = None
    if val:
        best = min(val)
        summary = {"epochs": len(val), "best": best, "best_epoch": val.index(best), "last": val[-1],
                   "mean_last_100": sum(val[-100:]) / len(val[-100:])}
    checkpoints = [{"file": f, "bytes": (ck_dir / f).stat().st_size} for f in files]
    failed = [k for k, v in checks.items() if not v["passed"]]
    return {"utc": now(), "run": run["name"], "valid": not failed, "failed_checks": failed, "checks": checks,
            "selected_checkpoint": selected, "checkpoints": checkpoints, "val_loss_mel_summary": summary,
            "run_record": {k: rr.get(k) for k in ("created_utc", "config_sha256", "split_sha256", "dataset_jsonl_sha256")}
            | {"repository_commit": (rr.get("environment") or {}).get("repository_commit")}}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("start", "status", "resume", "validate"))
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--run")
    p.add_argument("--note")
    a = p.parse_args(argv)
    c = Controller(a.plan)
    if a.command == "start":
        return c.start()
    if a.command == "status":
        return c.status_command()
    if a.command == "resume":
        if not (a.run and a.note):
            p.error("resume needs --run and --note")
        return c.resume(a.run, a.note)
    if not a.run:
        p.error("validate needs --run")
    return c.revalidate(a.run)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    raise SystemExit(main())
