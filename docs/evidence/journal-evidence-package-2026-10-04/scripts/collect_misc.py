"""Copies the remaining sources byte-for-byte (shutil.copy2) and records stat data.
Groups: historical LJSpeech-500 PyTorch results + manifests, prior-round agent evidence,
author/agent notes (Markdown reports), recovered source diffs, exploratory benchmark
scripts, ORT profile files, and checkpoint ctime/mtime (to show content unchanged
since before the Harvard run). Usage (WSL python): collect_misc.py <package_root>
"""
import datetime
import glob
import hashlib
import json
import os
import shutil
import sys

PKG = sys.argv[1]
W = "<WORKSPACE>/"
D = f"{W}/BanhmiTTS/docs"
GROUPS = {
    "evidence/C_data_split/lj500_results_historical": [
        "/tmp/eval_true_testset_500_baseline.json", "/tmp/eval_true_testset_500_mrf.json",
        "/tmp/eval_baseline_fp32_seeded_500_results.json", "/tmp/eval_mrf_final1500_500_results.json",
        "/tmp/eval_mrf_fp32_seeded_500_results.json", "/tmp/eval_seq111_fp32_seeded_500_results.json",
        "/tmp/eval_piper_original_500_full_results.json", "/tmp/piper_infer_batch.py", "/tmp/score_piper_original.py",
        "<WSL_PROJECT_HOME>/scripts/eval_baseline_fp32_seeded_500.py", "<WSL_PROJECT_HOME>/scripts/eval_mrf_final1500_500.py",
        "<WSL_PROJECT_HOME>/scripts/eval_mrf_fp32_seeded_500.py", "<WSL_PROJECT_HOME>/scripts/eval_seq111_fp32_seeded_500.py",
        "<WSL_PROJECT_HOME>/scripts/eval_mrf_final1500_500.log", "/tmp/eval_seq111_fp32_seeded_500.log",
        "/tmp/score_piper_original.log", "/tmp/piper_infer_batch.log", "/tmp/build_mrf_testset.py"],
    "evidence/C_data_split/prior_round_q03": glob.glob(f"{D}/evidence/q03_bosung_2026-10-04/*")
        + glob.glob(f"{D}/evidence/q03_2026-10-04/Q03_EVIDENCE.json") + glob.glob(f"{D}/evidence/q03_review_2026-10-04/REVIEW_CHECKS.json")
        + glob.glob(f"{D}/evidence/q03_q04_completion_2026-10-04/source_diffs/*")
        + glob.glob(f"{D}/evidence/q03_q04_completion_2026-10-04/SOURCE_RECOVERY_SUMMARY.json")
        + glob.glob(f"{D}/evidence/q03_q04_completion_2026-10-04/WSL_CHECKPOINT_SEARCH.json")
        + glob.glob(f"{D}/evidence/q03_q04_completion_2026-10-04/WINDOWS_CHECKPOINT_SEARCH.json")
        + glob.glob(f"{D}/evidence/q03_q04_completion_2026-10-04/CANDIDATE_AUDIT_v0.2.json"),
    "evidence/E_onnx_ptq/prior_round_q05": [f"{D}/evidence/q05_2026-10-04/{f}" for f in
        ("FP32_HANDOFF.json", "FINAL_VERIFICATION.json", "RUN_CONFIG.json", "EXPORT_CHECK_INPUTS.json", "SOURCE_MANIFEST.json",
         "LIVE_EXPORTER_FIX.json", "run_q05_model.executed.py")]
        + [f"{D}/evidence/q05_2026-10-04/{m}/REPORT.json" for m in ("baseline", "mrf", "seq", "piper")],
    "evidence/E_onnx_ptq/prior_round_audits": [f"{W}/paper_revisions/thesis_ptq/{f}" for f in
        ("round7_evaluation_code.md", "round7_evaluation_code.json", "round7_calibration_code.md", "round7_calibration_code.json",
         "round6_mrf_export_code.md", "round5_calibration_overlap.md")],
    "evidence/F_device_listening/exploratory_desktop_profiling": [
        "/tmp/bench_dec_vary_T.py", "/tmp/bench_sequential_vary_T.py", "/tmp/export_and_bench_dec_only.py",
        f"{W}/onnxruntime_profile__2026-09-19_23-36-08.json", f"{W}/onnxruntime_profile__2026-09-19_23-36-10.json"],
    "evidence/author_notes": [f"{D}/{f}" for f in (
        "Q03_CAU_HINH_VA_LICH_SU_HUAN_LUYEN.md", "Q04_PROTOCOL_DU_LIEU_VA_DANH_GIA.md", "Q05_EXPORT_ONNX_VA_KIEM_CHUNG.md",
        "CHECKLIST_UU_TIEN_PAPER_Q2.md", "MRF_vs_SEQ_comparison_Q2.md", "Piper_VITS2_BanhmiTTS_differences.md")]
        + [f"{W}/paper_revisions/thesis_ptq/CHECKLIST_SUA_LUAN_VAN.md"],
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


iso = lambda t: datetime.datetime.utcfromtimestamp(t).isoformat() + "Z"
log = []
for dest, srcs in GROUPS.items():
    os.makedirs(os.path.join(PKG, dest), exist_ok=True)
    for s in srcs:
        if not os.path.isfile(s):
            log.append({"source": s, "status": "MISSING"})
            continue
        d = os.path.join(PKG, dest, os.path.basename(s))
        shutil.copy2(s, d)
        st = os.stat(s)
        log.append({"source": s, "copy": os.path.relpath(d, PKG), "sha256": sha(d), "bytes": st.st_size,
                    "source_mtime_utc": iso(st.st_mtime), "source_ctime_utc": iso(st.st_ctime)})
ck = {}
for p in ["<WSL_PROJECT_HOME>/baseline_v2_noclip/checkpoints/best-epoch=1489-val_loss_mel=19.7882.ckpt",
          "<WSL_PROJECT_HOME>/mrf_111_run_500ep/checkpoints/best-epoch=1386-val_loss_mel=18.7681.ckpt",
          "<WSL_PROJECT_HOME>/seq111_final/checkpoints/best-epoch=1442-val_loss_mel=19.0136.ckpt",
          "<WSL_PROJECT_HOME>/piper_original/lj-med_1000.ckpt"]:
    st = os.stat(p)
    ck[p] = {"mtime_utc": iso(st.st_mtime), "ctime_utc": iso(st.st_ctime), "bytes": st.st_size}
q01 = json.load(open(f"{D}/evidence/q01_2026-10-04/CHECKPOINT_METADATA.json"))
json.dump({"copies": log, "checkpoint_stat_now": ck, "q01_checkpoint_metadata_excerpt": json.dumps(q01)[:4000]},
          open(os.path.join(PKG, "evidence", "A_inventory", "misc_copies_and_checkpoint_stat.json"), "w"), indent=1)
print("copied", sum("copy" in x for x in log), "missing", [x["source"] for x in log if x.get("status") == "MISSING"])
print(json.dumps(ck, indent=1))
