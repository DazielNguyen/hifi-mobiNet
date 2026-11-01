"""N100-only audit + isolated SeqIR candidate preparation. No RTF benchmark."""
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback
import zipfile

sys.dont_write_bytecode = True
ORIGINAL = "06dd0c87cd287554b52fd264d3d6450eb60b3bfe0bb09c2bdd7cd04adefc7c0e"
GUARD = "28090dbae369f24a22a7c04179191d959265e606c541e07e72b794e5d49ee60c"

def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()

def write(path, data):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")

def child(cmd, log_path, cwd):
    print("STAGE:", log_path.name, flush=True)
    with log_path.open("x", encoding="utf-8") as log:
        with subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1) as proc:
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            rc = proc.wait()
    if rc:
        raise RuntimeError(f"{log_path.name}: exit {rc}; see preserved log")

def main():
    package = Path(__file__).resolve().parent
    bundle = package.parent
    manifest = json.loads((package / "payload-sha256.json").read_text())
    for name, digest in manifest.items():
        p = (package / name).resolve()
        p.relative_to(package)
        if sha(p) != digest:
            raise RuntimeError("patch payload hash mismatch: " + name)
    suite_path = bundle / "openvino_suite.py"
    candidate_hash = sha(package / "openvino_suite.py")
    if sha(suite_path) not in (ORIGINAL, candidate_hash):
        raise RuntimeError("unknown current openvino_suite.py; refusing to overwrite")
    if sha(bundle / "diagnose_n100_calibration.py") != GUARD:
        raise RuntimeError("missing/changed diagnostic guard")
    sys.path.insert(0, str(bundle))
    spec = importlib.util.spec_from_file_location("_current_suite", suite_path)
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    env = suite.environment()  # Linux/N100 check before any inference or installation.
    import fcntl
    lock = (bundle / ".n100-spline-probe.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("another spline probe is still running; no second run started")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    session = bundle / ("probe-n100-empty-spline-" + stamp)
    session.mkdir()
    report = {"status": "running", "environment": env, "started_utc": suite.utc(),
              "session": str(session), "model": "piper-seqir-no-vits2-cpn",
              "original_suite_sha256": ORIGINAL, "candidate_suite_sha256": candidate_hash,
              "patch_payload_sha256": manifest, "scope": "masked duration spline FP32, same eligible Conv scopes",
              "not_run": ["six-model preparation", "RTF benchmark", "quality scoring"],
              "original_evidence": "work-n100 and diagnostic-n100 directories are preserved"}
    write(session / "session-start.json", report)
    success = False
    try:
        child([sys.executable, "-B", str(package / "audit_empty_branch.py"),
               "--bundle", str(bundle), "--output", str(session / "empty-branch-audit.json")],
              session / "audit.log", bundle)
        audit = json.loads((session / "empty-branch-audit.json").read_text())
        if audit["status"] != "completed" or len(audit["rows"]) != 60 or audit["empty_count"] == 0:
            raise RuntimeError("empty branch was not confirmed; do not install candidate or continue")
        backup_dir = bundle / "code-history"
        backup_dir.mkdir(exist_ok=True)
        backup = backup_dir / (ORIGINAL + ".openvino_suite.py")
        current_hash = sha(suite_path)
        if current_hash == ORIGINAL:
            if backup.exists():
                if sha(backup) != ORIGINAL:
                    raise RuntimeError("existing original-code backup hash mismatch")
            else:
                with backup.open("xb") as out:
                    out.write(suite_path.read_bytes())
            if sha(backup) != ORIGINAL:
                raise RuntimeError("original-code backup copy did not verify")
            pending = session / "openvino_suite.pending"
            with pending.open("xb") as out:
                out.write((package / "openvino_suite.py").read_bytes())
            if sha(pending) != candidate_hash:
                raise RuntimeError("candidate copy did not verify")
            pending.replace(suite_path)
        elif not backup.is_file() or sha(backup) != ORIGINAL:
            raise RuntimeError("candidate installed but original-code backup is missing")
        if sha(suite_path) != candidate_hash:
            raise RuntimeError("installed suite hash mismatch")
        write(session / "code-install.json", {"original_backup": str(backup),
              "original_sha256": ORIGINAL, "candidate_sha256": candidate_hash,
              "notice": "original SHA256SUMS now intentionally differs for openvino_suite.py; it is not rewritten"})
        child([sys.executable, "-B", str(suite_path), "prepare", "--bundle", str(bundle),
               "--work", str(session / "probe-work"), "--model", report["model"]],
              session / "prepare.log", bundle)
        model_dir = session / "probe-work" / report["model"]
        prep = json.loads((model_dir / "preparation.json").read_text())
        if prep["status"] != "prepared" or prep["script_sha256"] != candidate_hash:
            raise RuntimeError("candidate preparation not complete")
        if prep["numeric_guard"]["status"] != "no_numerical_issue_intercepted":
            raise RuntimeError("numeric guard did not complete")
        if prep["int8_weighted_kernel_count"] <= 0:
            raise RuntimeError("no actual INT8 weighted kernels")
        log = (session / "prepare.log").read_text(errors="replace")
        if re.search(r"RuntimeWarning|overflow encountered|invalid value encountered|divide by zero|Traceback", log, re.I):
            raise RuntimeError("warnings/errors remain; candidate is not accepted")
        for precision in ("fp32", "int8"):
            for item in prep["artifacts"][precision]["files"]:
                suite.verify_file(model_dir, item)
        report.update(status="seqir_candidate_prepared_without_intercepted_numerics",
                      empty_count=audit["empty_count"],
                      int8_weighted_kernel_count=prep["int8_weighted_kernel_count"],
                      quality_status="not certified; only one model tested")
        success = True
    except BaseException:
        report.update(status="probe_failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["finished_utc"] = suite.utc()
        write(session / "probe-report.json", report)
        destination = Path.home() / "Downloads" / ("N100-SPLINE-PROBE-" + stamp)
        destination.mkdir(parents=True, exist_ok=False)
        archive = destination / "spline-probe-report.zip"
        # Only reports/logs and code; no models, WAVs, venv, credentials or data cache.
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(session.rglob("*")):
                if p.is_file() and p.suffix in (".json", ".log"):
                    z.write(p, p.relative_to(session).as_posix())
            for name in ("run_probe.py", "audit_empty_branch.py", "openvino_suite.py",
                         "suite-change.diff", "payload-sha256.json"):
                z.write(package / name, "candidate-code/" + name)
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                raise RuntimeError("report archive CRC failed")
        (destination / (archive.name + ".sha256")).write_text(
            sha(archive) + "  " + archive.name + "\n", encoding="utf-8")
        print("\nSTATUS:", report["status"])
        print("REPORT ZIP:", archive)
        print("Copy report ZIP + .sha256 back to the PC. Do not run steps 05-10 yet.")
        try:
            subprocess.Popen(["xdg-open", str(destination)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass
    return 0 if success else 1

if __name__ == "__main__":
    raise SystemExit(main())

