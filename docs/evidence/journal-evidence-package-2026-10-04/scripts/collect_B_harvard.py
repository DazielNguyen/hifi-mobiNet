"""Part B: copy the original Harvard-720 artifacts byte-for-byte (shutil.copy2),
hash every WAV, and check that the four per-sentence result files share the same
IDs, texts and order. Read-only on the sources. Run with <WSL_PROJECT_HOME>/env/bin/python.
Usage: collect_B_harvard.py <package_root>
"""
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import wave

PKG = sys.argv[1]
RAW = os.path.join(PKG, "evidence", "B_harvard", "raw")
os.makedirs(RAW, exist_ok=True)
MODELS = ["baseline", "mrf", "seq", "piper"]
SOURCES = (["/tmp/harvard.html", "/tmp/harvard_testset_720.json", "/tmp/harvard_run_all.log"]
           + [f"/tmp/harvard_{m}_{k}.json" for m in MODELS for k in ("synth", "results")])


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


copied = []
for s in SOURCES:
    d = os.path.join(RAW, os.path.basename(s))
    shutil.copy2(s, d)
    assert sha(s) == sha(d)
    copied.append({"source": s, "copy": os.path.relpath(d, PKG), "sha256": sha(d), "bytes": os.path.getsize(d),
                   "source_mtime_utc": __import__("datetime").datetime.utcfromtimestamp(os.stat(s).st_mtime).isoformat() + "Z"})
sd = os.path.join(RAW, "scripts_original")
os.makedirs(sd, exist_ok=True)
for f in sorted(os.listdir("<WSL_PROJECT_HOME>/scripts/harvard")):
    s = os.path.join("<WSL_PROJECT_HOME>/scripts/harvard", f)
    shutil.copy2(s, os.path.join(sd, f))
    copied.append({"source": s, "copy": os.path.relpath(os.path.join(sd, f), PKG), "sha256": sha(s),
                   "bytes": os.path.getsize(s),
                   "source_mtime_utc": __import__("datetime").datetime.utcfromtimestamp(os.stat(s).st_mtime).isoformat() + "Z"})

test = json.load(open("/tmp/harvard_testset_720.json"))
res = {m: json.load(open(f"/tmp/harvard_{m}_results.json")) for m in MODELS}
syn = {m: json.load(open(f"/tmp/harvard_{m}_synth.json")) for m in MODELS}

checks = {"n_testset": len(test)}
for m in MODELS:
    r, s = res[m], syn[m]
    checks[m] = {
        "n_results": len(r), "n_synth": len(s),
        "idx_is_0..719_in_order": [x["idx"] for x in r] == list(range(720)),
        "text_matches_testset_in_order": [x["text"] for x in r] == [t["text"] for t in test],
        "synth_text_matches_results": [x["text"] for x in s] == [x["text"] for x in r],
        "synth_timing_copied_unchanged": all(a["rtf"] == b["rtf"] and a["synth_time_s"] == b["synth_time_s"]
                                             and a["audio_duration_s"] == b["audio_duration_s"] for a, b in zip(r, s)),
        "nonfinite_utmosv2": sum(not math.isfinite(x["utmosv2"]) for x in r),
        "nonfinite_wer": sum(not math.isfinite(x["wer"]) for x in r),
        "nonfinite_or_nonpositive_rtf": sum(not (math.isfinite(x["rtf"]) and x["rtf"] > 0) for x in r),
        "audio_shorter_than_0.2s": sum(x["audio_duration_s"] < 0.2 for x in r),
        "empty_asr_text": sum(not x["asr_text"].strip() for x in r),
        "fields": sorted(r[0].keys()),
    }
checks["all_four_same_text_order"] = all([x["text"] for x in res[m]] == [x["text"] for x in res["baseline"]] for m in MODELS)
checks["testset_fields"] = sorted(test[0].keys())

rows, dur_mismatch = [], {m: 0 for m in MODELS}
for m in MODELS:
    for x in syn[m]:
        p = x["wav_path"]
        with wave.open(p, "rb") as w:
            sr, ch, nf, sw = w.getframerate(), w.getnchannels(), w.getnframes(), w.getsampwidth()
        dur = nf / sr
        if abs(dur - x["audio_duration_s"]) > 1.0 / sr + 1e-9:
            dur_mismatch[m] += 1
        rows.append({"model": m, "idx": x["idx"], "wav_path": p, "sha256": sha(p), "bytes": os.path.getsize(p),
                     "sample_rate": sr, "channels": ch, "sample_width_bytes": sw, "frames": nf,
                     "duration_s": round(dur, 6), "json_audio_duration_s": x["audio_duration_s"]})
checks["wav_duration_vs_json_mismatch"] = dur_mismatch
checks["wav_formats"] = sorted({(r["sample_rate"], r["channels"], r["sample_width_bytes"]) for r in rows})
with open(os.path.join(PKG, "evidence", "B_harvard", "wav_inventory.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

log = open("/tmp/harvard_run_all.log", encoding="utf-8", errors="replace").read().splitlines()
checks["run_all_log"] = {
    "lines": len(log),
    "traceback_or_error_lines": [l for l in log if "Traceback" in l or l.startswith("Error") or "error:" in l.lower()][:20],
    "summary_lines": [l for l in log if l.startswith("===") or "synth done" in l or l.strip() == "ALL_DONE"],
    "header_lines_with_ckpt": [l for l in log if "ckpt=" in l or "resblock=" in l],
}
json.dump({"copied": copied, "checks": checks}, open(os.path.join(PKG, "evidence", "B_harvard", "harvard_integrity_checks.json"), "w"),
          indent=1, ensure_ascii=False)
print(json.dumps(checks, indent=1, ensure_ascii=False)[:4000])
