"""Builds claim-ledger.json (evidence-audit schema v1), the claim table Markdown
fragment, and evidence-manifest.csv (every file in the package with SHA-256, origin,
source type and supported claim IDs). Run from the package root with Windows Python:
    python scripts/build_ledger_and_manifest.py
"""
import csv
import datetime
import hashlib
import json
import os
import re

PKG = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
TODAY = "2026-10-04"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def src(path, locator, kind):
    full = os.path.join(PKG, path)
    return {"path": path, "sha256": sha(full) if os.path.isfile(full) else None, "locator": locator, "kind": kind}


H = "evidence/B_harvard/"
HS = H + "recomputed/harvard_stats_recomputed.json"
C = "evidence/C_data_split/"
D = "evidence/D_architecture/"
E = "evidence/E_onnx_ptq/"
CLAIMS = [
 ("J-E001", "Harvard-720: the four per-sentence result files (baseline, MRF, SEQ, Piper) contain 720 rows each with identical idx 0..719, text and order as harvard_testset_720.json; 0 non-finite UTMOSv2/WER/RTF, 0 audio <0.2 s, 0 empty ASR text; WAV durations match JSON.",
  "result", "SUPPORTED", [src(H + "harvard_integrity_checks.json", "checks.*", "result"), src(H + "raw/harvard_testset_720.json", "whole file", "result")],
  "Checked: IDs/text/order, finiteness, WAV headers vs JSON durations, log has no Traceback. Not checked: audio content quality beyond scores.", "PyTorch CPU synthesis, Harvard-720, single draw per sentence."),
 ("J-E002", "Harvard-720 mean per-sentence RTF (PyTorch CPU): HiFi-GAN ResBlock2 0.04177, Parallel-IR 0.03388, Sequential-IR 0.03081, Piper 0.03916.",
  "result", "SUPPORTED", [src(HS, "per_model.*.mean_rtf_per_sentence", "result"), src(H + "raw/scripts_original/synth_banhmi.py", "timer around model_g.infer only", "protocol"), src(H + "raw/scripts_original/synth_piper.py", "timer around model(...) only", "protocol")],
  "Recomputed from per-sentence JSON. Timer = infer call only (excludes model load, phonemization, WAV write); RTF = time/duration per sentence then mean. Not controlled: no warm-up, one timing pass, torch threads not recorded at run time (default 12 observed now), Piper ran under torch 2.14.1 vs 2.13.0, an idle TensorBoard server was running.", "i7-14700, WSL Ubuntu 24.04, PyTorch CPU FP32; single run."),
 ("J-E003", "Sequential-IR is 9% faster than Parallel-IR on Harvard-720 (p=7e-37).",
  "result", "PARTIAL", [src(HS, "paired_tests.seq_vs_mrf_rtf", "result")],
  "9.08% = RTF reduction 1-mean(SEQ)/mean(MRF); throughput gain 9.98%; ratio-of-sums reduction 9.34%. Wilcoxon (scipy default, two-sided, zero_method=wilcox) p=7.0e-37 replicates. Variation measured is between sentences only; benchmark repetition and training-run variance unmeasured; no warm-up.", "PyTorch CPU only; not ONNX/edge."),
 ("J-E004", "On Harvard-720, Sequential-IR UTMOSv2 is lower than Parallel-IR by about 0.02 (p about 0.05).",
  "result", "SUPPORTED", [src(HS, "paired_tests.seq_vs_mrf_utmosv2", "result")],
  "Mean diff -0.0237; Wilcoxon p=0.0471; MWU p=0.039; Holm over 15 tests p=0.094 (exploratory); paired bootstrap 95% [-0.0458, -0.0010] (exploratory). UTMOSv2 is a predicted score with unseeded random crops, one pass.", "UTMOSv2 predicted quality, not human MOS."),
 ("J-E005", "On Harvard-720, the WER difference between Sequential-IR and Parallel-IR is not statistically significant.",
  "result", "PARTIAL", [src(HS, "paired_tests.seq_vs_mrf_wer", "result")],
  "Mean diff +0.0149 (WER fraction; SEQ worse). Wilcoxon p=0.0523 after dropping 323/720 zero differences; MWU p=0.080. Exploratory paired bootstrap CI [+0.0019, +0.0277] excludes 0. 'Not significant' is literally true for the reported test but does not imply equivalence.", "Whisper-small macro WER."),
 ("J-E006", "Sequential-IR and Parallel-IR have comparable/equivalent speech quality.",
  "inference", "UNSUPPORTED", [src(HS, "paired_tests.seq_vs_mrf_*", "result")],
  "No equivalence/non-inferiority margin was pre-specified; both point estimates favour Parallel-IR; WER bootstrap CI excludes 0.", "—"),
 ("J-E007", "On Harvard-720, Sequential-IR vs Piper: 21% faster, UTMOSv2 +0.09, WER -0.05, all p<1e-10.",
  "result", "SUPPORTED", [src(HS, "paired_tests.seq_vs_piper_*", "result")],
  "21.33% RTF reduction (27.1% throughput gain); UTMOSv2 +0.0936; WER -0.0525 as macro WER fraction = 5.25 percentage points absolute; Wilcoxon p = 8.1e-98 / 2.9e-13 / 4.7e-11. System-level comparison: different backbone, decoder, precision, data split, torch build.", "PyTorch CPU; Piper published checkpoint."),
 ("J-E008", "Harvard-720 was not used to select checkpoints or the historical PTQ scope.",
  "inference", "PARTIAL", [src("evidence/A_inventory/checkpoint_metadata_now.json", "*.stored_in_checkpoint.callbacks", "result"), src(H + "harvard_integrity_checks.json", "copied[].source_mtime_utc", "result")],
  "Earliest 'harvard' mention in the inspected transcripts 2026-10-03T18:50:53Z; all Harvard files dated 2026-10-03/04; checkpoint callbacks finished 2026-09-02/09-17/09-29; PTQ graphs dated 2026-09-07..09-30. Not covered: activity outside the inspected sessions. Harvard results DID inform the Q02 claim wording proposed in this conversation.", "Inspected records only."),
 ("J-E009", "Harvard-720 sentences do not overlap the LJSpeech training transcripts.",
  "result", "PARTIAL", [src(C + "data_split_checks_recomputed.json", "exact_sentence_overlap_*, harvard_sentences_sharing_any_*gram, exact_phoneme_id_sequence_overlap", "result")],
  "0 exact matches under two normalizations; 0 shared 6-grams; 0 identical phoneme-ID sequences; 37 sentences share a 4-gram and 7 a 5-gram (common phrases). Only against the 13,100 transcripts in dataset.jsonl; Piper's own training transcripts, audio similarity and scorer training data unchecked.", "Text-level only."),
 ("J-E010", "HiFi-GAN ResBlock2 and Sequential-IR were trained on the canonical 12,500/100/500 partition; Parallel-IR on a different random partition sharing 11,930/12,500 (95.44%) training utterances.",
  "result", "SUPPORTED", [src(C + "data_split_checks_recomputed.json", "split_overlap", "result"), src(C + "prior_round_q03/TENSORBOARD_VAL_HEADS.json", "distinct_tag_sequences", "result"), src(C + "MRF_SPLIT_RECONSTRUCTED_VALIDATED.json", "checks", "result")],
  "Canonical corroborated by TensorBoard validation heads (9 phases) and kernel timestamps; MRF split reconstructed and corroborated against logged validation heads and the 2026-09-15 test manifest. Not a contemporaneous full-manifest hash.", "Training data membership."),
 ("J-E011", "Selected checkpoints are epoch 1489 (baseline), 1386 (MRF), 1442 (SEQ), chosen by ModelCheckpoint on val_loss_mel (min); Piper lj-med_1000 is epoch 999 with monitor=None.",
  "result", "SUPPORTED", [src("evidence/A_inventory/checkpoint_metadata_now.json", "*.stored_in_checkpoint.epoch / callbacks", "result")],
  "Epochs are zero-based metadata. Selection by validation mel L1 (x45), not by UTMOS/WER.", "—"),
 ("J-E012", "HiFi-GAN ResBlock2 baseline used gradient clipping 1.0 in its final phases; Parallel-IR and Sequential-IR used none.",
  "result", "SUPPORTED", [src("evidence/A_inventory/checkpoint_metadata_now.json", "baseline.hyper_parameters.gradient_clip_val", "result"), src(C + "prior_round_q03/Q03_EVIDENCE.json", "phases", "result")],
  "Baseline v0 null, v1/v2 1.0 (hparams per session, Q03).", "—"),
 ("J-E013", "The baseline's SDP Log-flow exp clamp was changed from 30 to 10 (and an inference logw clamp added) before resuming from epoch 870.",
  "method", "SUPPORTED", [src(C + "prior_round_q03/baseline_v0_to_baseline_v2_flows.py.diff", "hunk @@ -16,10", "protocol"), src(C + "prior_round_q03/TRANSCRIPT_TIMELINE_TRAINING_WINDOWS.txt", "2026-08-31T21:07-21:08", "protocol")],
  "Recovered editor source + transcript timeline + resume log line; no runtime source hash.", "Baseline only."),
 ("J-E014", "Sequential-IR training code differs from Parallel-IR's: non-finite-gradient update skipping, SDP inference-health gating of best-checkpoint eligibility, duration-predictor edits, canonical split loader.",
  "method", "SUPPORTED", [src(C + "prior_round_q03/mrf_v0_to_seq_v0_training.py.diff", "whole diff", "protocol")],
  "Recovered source diff; SEQ logs contain skip messages (Q03). Confounder for any MRF-vs-SEQ causal attribution.", "—"),
 ("J-E015", "Decoder parameters: Sequential-IR 1.11M, Parallel-IR 1.15M, HiFi-GAN ResBlock2 1.66M.",
  "result", "SUPPORTED", [src(D + "counts_banhmi_recomputed.json", "*.decoder_params_folded_state_dict", "result")],
  "Exact 1,107,968 / 1,154,560 / 1,662,976; weight-norm folded; equals live count after remove_weight_norm.", "model_g.dec only."),
 ("J-E016", "Decoder compute: Sequential-IR 0.98 and Parallel-IR 1.23 GFLOPs per second of audio.",
  "result", "SUPPORTED", [src(D + "counts_banhmi_recomputed.json", "*.decoder_gflops_T86 / per_exact_second", "result")],
  "Convention: Conv1d/ConvTranspose1d MACs x2 only (no activations/bias); T=86 frames (0.998 s): 0.9846 / 1.2306; per exact second 0.9861 / 1.2325. Baseline and Piper 3.90.", "Decoder only, batch 1."),
 ("J-E017", "The Sequential-IR decoder is 33% smaller than Piper's; total inference parameters are 3.9% larger.",
  "result", "SUPPORTED", [src(D + "counts_banhmi_recomputed.json", "SEQ", "result"), src(D + "counts_piper_recomputed.json", "Piper", "result")],
  "1-1,107,968/1,662,976 = 33.37%; 17,052,464/16,412,464-1 = 3.90%; boundary enc_p+dp+flow+dec, enc_q and discriminators excluded.", "—"),
 ("J-E018", "The HiFi-GAN ResBlock2 baseline decoder is identical to Piper's decoder.",
  "result", "CONTRADICTED", [src(D + "code_excerpts.md", "normalization.py L25-38; piper models.py L364", "protocol"), src(D + "counts_piper_recomputed.json", "decoder", "result")],
  "Same parameter count/shapes and MACs, but the activation before conv_post differs: LeakyReLU slope 0.1 (BanhmiTTS) vs F.leaky_relu default 0.01 (Piper).", "Forward semantics."),
 ("J-E019", "Parallel-IR uses three parallel inverted-residual branches (depthwise kernels 3/5/7) averaged per stage; Sequential-IR uses two serial blocks (kernel 3, expansion 1) per stage; upsampling strides 8/8/4, kernels 16/16/8, channels 128/64/32.",
  "method", "SUPPORTED", [src(D + "code_excerpts.md", "generator.py L74-169", "protocol"), src("evidence/A_inventory/checkpoint_metadata_now.json", "hyper_parameters", "result")],
  "Current code + checkpoint hparams + state shapes (Q03 §3). Current code is not a runtime hash of training source.", "—"),
 ("J-E020", "The normalizing flow has 4 TransformerCouplingLayer+Flip pairs, each with a 1-layer 2-head self-attention (residual) before a 4-layer WaveNet (kernel 5), mean-only shift, run in reverse at inference; 8.29M params vs Piper 7.09M (Piper flow has no attention).",
  "method", "SUPPORTED", [src(D + "code_excerpts.md", "flow_block.py L14-122; synthesizer.py flow construction/infer", "protocol"), src(D + "counts_piper_recomputed.json", "params_folded_by_module.model_g.flow", "result")],
  "Backbone description; not a novel contribution and not flow matching.", "—"),
 ("J-E021", "Historical ONNX Runtime FP32 RTF: Parallel-IR 0.0416 vs HiFi-GAN ResBlock2 0.0388 (Parallel-IR slower).",
  "result", "PARTIAL", [src(E + "historical_results/eval_mrf_fp32_onnx_500_results.json", "mean rtf", "result"), src(E + "scripts_and_logs/eval_mrf_fp32_onnx_500.log", "line 2", "result"), src(E + "prior_round_q05/FP32_HANDOFF.json", "historical graph check", "result")],
  "Numbers reproduce, but mrf_fp32.onnx does not match checkpoint 1386 (Q05: 0/478 tensors) and MRF used its own 500-sentence manifest (21/500 shared). Cannot be attributed to MRF-1386.", "Historical; 2 threads."),
 ("J-E022", "Sequential-IR's speed advantage holds across PyTorch, ONNX FP32 and INT8.",
  "inference", "UNSUPPORTED", [src(E + "onnx_inventory_inspected_now.json", "seq_final_int8 vs *_flow_enc_p_dp", "result")],
  "No final runs on the Q05 graphs; historical MRF graph source unknown; historical INT8 scopes differ (SEQ adds 40 QLinearMatMul).", "—"),
 ("J-E023", "In the inspected INT8 graphs the decoder remains FP32.",
  "result", "SUPPORTED", [src(E + "onnx_inventory_inspected_now.json", "any_quantized_op_under_dec", "result")],
  "No quantized op under /dec/ in baseline/MRF flow_enc_p_dp and SEQ int8 graphs. Does not show INT8 robustness of the decoder.", "Graph structure only."),
 ("J-E024", "The PTQ scope quantizes Conv layers of enc_p, dp and flow.",
  "result", "PARTIAL", [src(E + "onnx_inventory_inspected_now.json", "quantized_ops_by_prefix", "result"), src(E + "scripts_and_logs/ptq_quantize.py", "L64-118", "protocol")],
  "Baseline/MRF historical: 133 QLinearConv (enc_p 37, dp 32, flow 64), no QLinearMatMul. SEQ final INT8 and the 09-21 rtf_compare graphs also have 40 QLinearMatMul (op-types Conv,ConvTranspose,MatMul). Static QOperator, int8 per-channel weights, uint8 activations.", "Historical graphs."),
 ("J-E025", "The 60 PTQ calibration sentences were drawn from training data only.",
  "result", "CONTRADICTED", [src(E + "historical_calibration60_recomputed.json", "membership", "result")],
  "Recomputed from ptq_quantize.py's own rule (Random(42).sample over all dataset.jsonl rows): 3 in canonical/LJ-500-baseline test, 1 in MRF test manifest, 2 in MRF validation, 0 Harvard.", "Historical PTQ."),
 ("J-E026", "The historical PTQ scope was selected with a validated evaluator.",
  "result", "CONTRADICTED", [src(E + "prior_round_audits/round7_evaluation_code.md", "Main findings 1-2", "author-note"), src(E + "scripts_and_logs/eval_wer_cheap.py", "line 71", "protocol")],
  "Cheap selector passed 22.05 kHz arrays to Whisper (expects 16 kHz), used base.en and different normalization; no rerun of the selection exists.", "—"),
 ("J-E027", "Q05 FP32 ONNX graphs match PyTorch within NRMSE<=1e-3 and max-abs<=5e-3 on 10 cases per model.",
  "result", "PARTIAL", [src(E + "prior_round_q05/FINAL_VERIFICATION.json", "checks", "result"), src(E + "prior_round_q05/RUN_CONFIG.json", "thresholds", "protocol")],
  "Evidence produced by a prior round with saved NPZ in WSL; copied here but not re-executed in this round.", "ORT 1.23.2 CPU, 2 threads, given inputs."),
 ("J-E028", "RTF/latency measurements on Intel N150 and Raspberry Pi 5.",
  "result", "UNSUPPORTED", [], "NOT AVAILABLE: only planning documents and desktop i7-14700 measurements exist.", "—"),
 ("J-E029", "Human listening test results.",
  "result", "UNSUPPORTED", [], "NOT AVAILABLE: no protocol, responses or approvals found. UTMOSv2 is not a substitute.", "—"),
 ("J-E030", "MRF_vs_SEQ_comparison_Q2.md §2 'PyTorch FP32' row for baseline and MRF.",
  "author-note", "CONTRADICTED", [src("evidence/author_notes/MRF_vs_SEQ_comparison_Q2.md", "§2 table", "author-note"), src(C + "lj500_results_historical/eval_baseline_fp32_seeded_500_results.json", "means", "result"), src(C + "lj500_results_historical/eval_mrf_final1500_500_results.json", "means", "result")],
  "Baseline RTF 0.0434 is PyTorch but UTMOSv2 3.132/WER 0.1230 are the ONNX values (PyTorch: 3.125/0.1230); MRF RTF 0.0370 is PyTorch-1386 but 3.537/0.1043 are from the unverified ONNX graph (PyTorch-1386: 3.651/0.0983) on the MRF manifest.", "Historical LJ-500."),
 ("J-E031", "Sequential-IR has higher UTMOSv2 than Parallel-IR (LJSpeech-500 tables).",
  "result", "CONTRADICTED", [src(C + "lj500_results_historical/eval_seq111_fp32_seeded_500_results.json", "means", "result"), src(C + "lj500_results_historical/eval_mrf_final1500_500_results.json", "means", "result"), src(HS, "seq_vs_mrf_utmosv2", "result")],
  "LJ-500 rows use different manifests for MRF vs SEQ (21/500 shared); on the shared Harvard-720 set SEQ is lower (-0.024).", "—"),
 ("J-E032", "Piper has seen most of the canonical LJ-500 test sentences during training.",
  "inference", "NOT_CHECKED", [src(C + "data_split_checks_recomputed.json", "hparams_from_checkpoints.Piper (validation_split, num_test_examples)", "result")],
  "Plausible from validation_split=0.05 and num_test_examples=0, but Piper's split is unknown; not measurable.", "—"),
 ("J-E033", "Scorer identity: Whisper small and UTMOSv2 fusion_stage3 fold0 s42 with the stated hashes were used for Harvard scoring.",
  "result", "PARTIAL", [src(H + "raw/scripts_original/score.py", "whisper.load_model('small'); utmosv2.create_model(pretrained=True)", "protocol"), src("evidence/A_inventory/environment_now.json", "main_env.packages", "result")],
  "Script names the models; weights hashes (Whisper 9ecf7799..., UTMOSv2 c8149d98...) and package versions are current observations (Q04), not recorded at run time.", "—"),
]

ledger = {"schema_version": 1, "project_id": "hifi-mobinet-journal-evidence-2026-10-04", "claims": []}
for cid, stmt, kind, status, sources, rationale, scope in CLAIMS:
    ledger["claims"].append({"id": cid, "statement": stmt, "kind": kind, "status": status, "manuscript_locations": [],
                             "sources": sources, "verification": {"by": "ai", "checked_at": TODAY, "rationale": rationale},
                             "limitations": scope})
json.dump(ledger, open(os.path.join(PKG, "claim-ledger.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)

rows = ["| ID | Assertion | Loại | Nguồn (path · locator) | Trạng thái | Đã kiểm / chưa kiểm | Phạm vi |", "|---|---|---|---|---|---|---|"]
for c in ledger["claims"]:
    s = "<br>".join(f"`{x['path']}` · {x['locator']} · `{(x['sha256'] or 'null')[:12]}`" for x in c["sources"]) or "—"
    rows.append(f"| {c['id']} | {c['statement']} | {c['kind']} | {s} | **{c['status']}** | {c['verification']['rationale']} | {c['limitations']} |")
open(os.path.join(PKG, "evidence", "claim_table.md"), "w", encoding="utf-8").write("\n".join(rows) + "\n")

# Manifest: claims supported per file = claims whose sources reference that file
ref = {}
for c in ledger["claims"]:
    for x in c["sources"]:
        ref.setdefault(x["path"], set()).add(c["id"])
origins = {}
for f in ["evidence/B_harvard/harvard_integrity_checks.json", "evidence/C_data_split/data_split_checks_recomputed.json",
          "evidence/E_onnx_ptq/copied_files.json", "evidence/A_inventory/misc_copies_and_checkpoint_stat.json"]:
    d = json.load(open(os.path.join(PKG, f), encoding="utf-8"))
    items = d if isinstance(d, list) else (d.get("copied") or d.get("copies") or [])
    for it in items:
        if isinstance(it, dict) and it.get("copy"):
            origins[it["copy"].replace("\\", "/")] = it["source"]


def stype(rel):
    if rel.startswith("scripts/") or "/recomputed/" in rel or rel.endswith(("_recomputed.json", "_now.json")) or rel in (
            "claim-ledger.json", "evidence/claim_table.md") or "code_excerpts" in rel or "inspected_now" in rel \
            or rel.endswith(("integrity_checks.json", "wav_inventory.csv", "copied_files.json", "misc_copies_and_checkpoint_stat.json")):
        return "recomputed_this_round"
    if "author_notes/" in rel or "prior_round_audits/" in rel and rel.endswith(".md"):
        return "author_or_agent_note"
    if "prior_round_" in rel:
        return "prior_round_agent_output"
    if "exploratory_desktop_profiling" in rel:
        return "original_exploratory"
    return "original_raw"


out = []
for root, _, files in os.walk(PKG):
    for fn in sorted(files):
        p = os.path.join(root, fn)
        rel = os.path.relpath(p, PKG).replace(os.sep, "/")
        if rel in ("evidence-manifest.csv",) or rel.endswith(".zip"):
            continue
        out.append({"file": rel, "sha256": sha(p), "bytes": os.path.getsize(p), "original_path": origins.get(rel, ""),
                    "source_type": stype(rel), "claims": ";".join(sorted(ref.get(rel, [])))})
with open(os.path.join(PKG, "evidence-manifest.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(sorted(out, key=lambda r: r["file"]))
print(len(ledger["claims"]), "claims;", len(out), "files in manifest;",
      {s: sum(c["status"] == s for c in ledger["claims"]) for s in ("SUPPORTED", "PARTIAL", "UNSUPPORTED", "CONTRADICTED", "NOT_CHECKED")})
print("sources missing:", [x["path"] for c in ledger["claims"] for x in c["sources"] if x["sha256"] is None])
