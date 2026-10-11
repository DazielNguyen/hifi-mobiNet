"""Sequential training controller: state machine, recovery rules and selection (no training is run)."""
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_sequence", ROOT / "scripts/training/run_sequence.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)


def _controller(tmp_path, monkeypatch, states=None):
    plan = {"sequence_id": "t", "state_dir": "state", "runs": [
        {"name": "a", "model_id": "A", "config": "a.yaml", "run_dir": "ra", "log": "ra.log"},
        {"name": "b", "model_id": "B", "config": "b.yaml", "run_dir": "rb", "log": "rb.log"}],
        "train_args": [], "budget": {}, "expected": {}, "preflight": {}}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(rs, "ROOT", tmp_path)
    c = rs.Controller(tmp_path / "plan.json")
    monkeypatch.setattr(c, "lock", lambda: None)
    monkeypatch.setattr(c, "final_report", lambda status: None)
    status = c.load_status()
    for name, state in (states or {}).items():
        status["runs"][name]["state"] = state
    c.save_status(status)
    launched = []

    def fake_launch(run, status, resume, note=None):
        launched.append((run["name"], resume))
        status["runs"][run["name"]]["state"] = fake_launch.outcome.get(run["name"], "completed")
        return 0 if status["runs"][run["name"]]["state"] == "completed" else 1

    fake_launch.outcome = {}
    monkeypatch.setattr(c, "launch", fake_launch)
    return c, launched, fake_launch


def test_runs_in_order_and_skips_completed(tmp_path, monkeypatch):
    c, launched, _ = _controller(tmp_path, monkeypatch, {"a": "completed"})
    assert c.run_remaining(c.load_status()) == 0
    assert launched == [("b", False)]


def test_failed_first_run_stops_the_sequence(tmp_path, monkeypatch):
    c, launched, fake = _controller(tmp_path, monkeypatch)
    fake.outcome = {"a": "failed"}
    assert c.run_remaining(c.load_status()) == 1
    assert launched == [("a", False)]


@pytest.mark.parametrize("state", ["failed", "needs_recovery"])
def test_bad_state_is_never_restarted_automatically(tmp_path, monkeypatch, state):
    c, launched, _ = _controller(tmp_path, monkeypatch, {"a": state})
    assert c.run_remaining(c.load_status()) == 1
    assert launched == []


def test_lost_process_becomes_needs_recovery(tmp_path, monkeypatch):
    c, _, _ = _controller(tmp_path, monkeypatch)
    status = c.load_status()
    status["runs"]["a"].update(state="running", attempts=[{"pid": 2 ** 22 + 12345}])
    monkeypatch.setattr(rs, "pid_alive", lambda pid: False)
    c.reconcile(status)
    assert json.loads(c.status_path.read_text())["runs"]["a"]["state"] == "needs_recovery"


@pytest.mark.skipif(os.name == "nt", reason="the controller is POSIX-only; os.kill(pid, 0) terminates on Windows")
def test_live_orphan_process_blocks_the_controller(tmp_path, monkeypatch):
    c, _, _ = _controller(tmp_path, monkeypatch)
    status = c.load_status()
    status["runs"]["a"].update(state="running", attempts=[{"pid": os.getpid()}])
    with pytest.raises(SystemExit, match="still running"):
        c.reconcile(status)


@pytest.mark.parametrize("state", ["pending", "completed"])
def test_resume_only_for_interrupted_or_failed_runs(tmp_path, monkeypatch, state):
    c, launched, _ = _controller(tmp_path, monkeypatch, {"a": state})
    with pytest.raises(SystemExit, match="only needs_recovery or failed"):
        c.resume("a", "note")
    assert launched == []


def test_resume_requires_earlier_runs_completed(tmp_path, monkeypatch):
    c, launched, _ = _controller(tmp_path, monkeypatch, {"a": "needs_recovery", "b": "needs_recovery"})
    with pytest.raises(SystemExit, match="earlier runs are not completed"):
        c.resume("b", "note")
    assert launched == []


def test_resume_continues_with_later_runs(tmp_path, monkeypatch):
    c, launched, _ = _controller(tmp_path, monkeypatch, {"a": "needs_recovery"})
    assert c.resume("a", "power loss") == 0
    assert launched == [("a", True), ("b", False)]


def test_log_checks_and_checkpoint_selection():
    assert rs.check_log_text("... `Trainer.fit` stopped: `max_epochs=1500` reached.", 1500) == {"budget_reached": True, "traceback": False}
    assert rs.check_log_text("Traceback (most recent call last):", 1500)["traceback"]
    files = ["best-epoch=3-val_loss_mel=1.5000.ckpt", "best-epoch=1-val_loss_mel=1.5000.ckpt", "last.ckpt"]
    assert rs.select_checkpoint([2.0, 1.5, 1.7, 1.5], files) == {"epoch": 1, "val_loss_mel": 1.5, "file": files[1]}
    assert rs.select_checkpoint([1.0, 2.0], files) == {"epoch": 0, "val_loss_mel": 1.0, "file": None}  # minimum not on disk


def test_status_written_atomically(tmp_path):
    path = tmp_path / "s.json"
    rs.write_json_atomic(path, {"x": 1})
    assert json.loads(path.read_text()) == {"x": 1} and not (tmp_path / "s.json.tmp").exists()
