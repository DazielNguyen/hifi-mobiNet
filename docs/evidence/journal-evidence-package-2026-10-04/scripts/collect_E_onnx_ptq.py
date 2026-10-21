"""Part E: ONNX artifact inventory and graph inspection (read-only; onnx.load only,
no inference, no quantization, no export). Also copies the PTQ/eval scripts and logs,
and RECOMPUTES the historical 60-sentence calibration set from ptq_quantize.py's own
sampling rule (random.Random(42).sample(all dataset.jsonl rows, 60)) to check membership.
Usage: collect_E_onnx_ptq.py <package_root>
"""
import collections
import datetime
import glob
import hashlib
import json
import os
import random
import shutil
import sys

import onnx

PKG = sys.argv[1]
OUT = os.path.join(PKG, "evidence", "E_onnx_ptq")
os.makedirs(os.path.join(OUT, "scripts_and_logs"), exist_ok=True)
os.makedirs(os.path.join(OUT, "historical_results"), exist_ok=True)

ARTIFACTS = (sorted(glob.glob("<WSL_PROJECT_HOME>/paper_evidence/q05_2026-10-04T000830Z/*/*.onnx"))
             + sorted(glob.glob("<WSL_PROJECT_HOME>/quantized_models/ptq_sweep/baseline_*.onnx"))
             + sorted(glob.glob("<WSL_PROJECT_HOME>/quantized_models/ptq_sweep/mrf_*.onnx"))
             + sorted(glob.glob("/tmp/seq_final_onnx/*.onnx")) + sorted(glob.glob("/tmp/rtf_compare/*.onnx"))
             + sorted(glob.glob("/tmp/dec_only_*.onnx")))
QOPS = {"QLinearConv", "ConvInteger", "QLinearMatMul", "MatMulInteger", "QuantizeLinear", "DequantizeLinear",
        "DynamicQuantizeLinear", "QLinearAdd", "QLinearMul"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def prefix(name):
    parts = name.split("/")
    return parts[1] if len(parts) > 2 else "(top-level)"


inv = []
for p in ARTIFACTS:
    st = os.stat(p)
    m = onnx.load(p, load_external_data=False)
    ops = collections.Counter(n.op_type for n in m.graph.node)
    q_by_prefix = collections.defaultdict(collections.Counter)
    fp_conv_by_prefix = collections.defaultdict(collections.Counter)
    for n in m.graph.node:
        if n.op_type in QOPS:
            q_by_prefix[prefix(n.name)][n.op_type] += 1
        if n.op_type in ("Conv", "ConvTranspose", "MatMul"):
            fp_conv_by_prefix[prefix(n.name)][n.op_type] += 1
    dims = lambda vi: [d.dim_param or d.dim_value for d in vi.type.tensor_type.shape.dim]
    inv.append({
        "path": p, "sha256": sha(p), "bytes": st.st_size,
        "mtime_utc": datetime.datetime.utcfromtimestamp(st.st_mtime).isoformat() + "Z",
        "ctime_utc": datetime.datetime.utcfromtimestamp(st.st_ctime).isoformat() + "Z",
        "ir_version": m.ir_version, "opset": {o.domain or "ai.onnx": o.version for o in m.opset_import},
        "producer": f"{m.producer_name} {m.producer_version}",
        "metadata_props": {kv.key: kv.value[:300] for kv in m.metadata_props},
        "inputs": {i.name: dims(i) for i in m.graph.input}, "outputs": {o.name: dims(o) for o in m.graph.output},
        "n_nodes": len(m.graph.node), "random_nodes": {k: v for k, v in ops.items() if k.startswith("Random")},
        "quantized_ops_by_prefix": {k: dict(v) for k, v in q_by_prefix.items()},
        "fp32_conv_matmul_by_prefix": {k: dict(v) for k, v in fp_conv_by_prefix.items()},
        "any_quantized_op_under_dec": "dec" in q_by_prefix,
        "n_quantized_ops_total": sum(sum(v.values()) for v in q_by_prefix.values()),
    })
    del m
json.dump(inv, open(os.path.join(OUT, "onnx_inventory_inspected_now.json"), "w"), indent=1)

COPY = ["<WSL_PROJECT_HOME>/scripts/ptq_quantize.py", "<WSL_PROJECT_HOME>/scripts/ptq_sweep_diag.txt",
        "<WSL_PROJECT_HOME>/scripts/eval_onnx_ptq_500.py", "<WSL_PROJECT_HOME>/scripts/eval_wer_cheap.py",
        "<WSL_PROJECT_HOME>/scripts/export_and_quantize_onnx.py", "<WSL_PROJECT_HOME>/scripts/onnx_ptq_evals_chain.log",
        "/tmp/export_onnx_fixed.py", "/tmp/bench_rtf_4way.py", "/tmp/seq_stats.py",
        "/tmp/eval_seq_final_fp32_onnx.log", "/tmp/eval_seq_final_int8_onnx.log",
        "<WSL_PROJECT_HOME>/scripts/bench_vnni_baseline.py", "<WSL_PROJECT_HOME>/scripts/bench_vnni_mrf.py",
        "<WSL_PROJECT_HOME>/scripts/bench_vnni_baseline_rerun.log", "<WSL_PROJECT_HOME>/scripts/bench_vnni_mrf.log"]
COPY += sorted(glob.glob("<WSL_PROJECT_HOME>/scripts/eval_*_onnx_500*.log")) + sorted(glob.glob("<WSL_PROJECT_HOME>/scripts/ptq_sweep*.log"))
RES = sorted(glob.glob("/tmp/eval_*_onnx_500*_results.json")) + sorted(glob.glob("/tmp/eval_seq_final_*_onnx_results.json")) \
    + sorted(glob.glob("/tmp/rtf_compare/*_results.json"))
copied = []
for s in COPY + RES:
    if not os.path.exists(s):
        copied.append({"source": s, "status": "MISSING"})
        continue
    sub = "historical_results" if s in RES else "scripts_and_logs"
    name = s.replace("/tmp/rtf_compare/", "rtf_compare__") if "rtf_compare" in s else os.path.basename(s)
    d = os.path.join(OUT, sub, os.path.basename(name))
    shutil.copy2(s, d)
    st = os.stat(s)
    copied.append({"source": s, "copy": os.path.relpath(d, PKG), "sha256": sha(d), "bytes": st.st_size,
                   "source_mtime_utc": datetime.datetime.utcfromtimestamp(st.st_mtime).isoformat() + "Z",
                   "source_ctime_utc": datetime.datetime.utcfromtimestamp(st.st_ctime).isoformat() + "Z"})

DS = "<WSL_PROJECT_HOME>/BanhmiTTS/training_output/ljspeech/medium/dataset.jsonl"
entries = [json.loads(l) for l in open(DS, encoding="utf-8") if l.strip()]
calib = random.Random(42).sample(entries, 60)
ids = [e["audio_norm_path"] for e in calib]
canon = json.load(open("<WSL_PROJECT_HOME>/BanhmiTTS/training_output/ljspeech/medium/canonical_split.json"))
mrf = json.load(open(f"<WORKSPACE>/BanhmiTTS/docs/evidence/q03_bosung_2026-10-04/MRF_SPLIT_RECONSTRUCTED_VALIDATED.json"))["split"]
lj_base = {r["audio_norm_path"] for r in json.load(open("/tmp/eval_true_testset_500_baseline.json"))}
lj_mrf = {r["audio_norm_path"] for r in json.load(open("/tmp/eval_true_testset_500_mrf.json"))}
harv_texts = {r["text"].lower() for r in json.load(open("/tmp/harvard_testset_720.json"))}
cal = {
    "rule": "random.Random(42).sample(all non-empty dataset.jsonl rows in file order, 60) -- ptq_quantize.py JsonlCalibReader",
    "applies_to": "every historical ptq_quantize.py run invoked with --calibration-dataset dataset.jsonl --calibration-sentences 60 "
                  "(the sweep diag and the 2026-09-21/09-30 SEQ commands); assumes dataset.jsonl unchanged (ctime 2026-07-28)",
    "dataset_sha256": sha(DS),
    "ids": ids, "texts": [e["text"] for e in calib],
    "phoneme_id_lengths": [len(e["phoneme_ids"]) for e in calib],
    "membership": {
        "canonical": {k: sum(i in set(v) for i in ids) for k, v in canon.items()},
        "mrf_reconstructed": {k: sum(i in set(v) for i in ids) for k, v in mrf.items()},
        "lj500_baseline_manifest": sum(i in lj_base for i in ids),
        "lj500_mrf_manifest": sum(i in lj_mrf for i in ids),
        "harvard_exact_text": sum(e["text"].lower() in harv_texts for e in calib),
    },
}
json.dump(cal, open(os.path.join(OUT, "historical_calibration60_recomputed.json"), "w"), indent=1, ensure_ascii=False)
json.dump(copied, open(os.path.join(OUT, "copied_files.json"), "w"), indent=1)
for a in inv:
    print(f"{a['path'].replace('<WSL_PROJECT_HOME>/','~/')[:70]:70s} {a['bytes']/1e6:6.1f}MB opset={a['opset'].get('ai.onnx')} "
          f"q={a['n_quantized_ops_total']:3d} dec_q={a['any_quantized_op_under_dec']} q_prefix={ {k: sum(v.values()) for k, v in a['quantized_ops_by_prefix'].items()} } "
          f"rand={a['random_nodes']} meta={list(a['metadata_props'])[:4]}")
print("calibration membership:", json.dumps(cal["membership"]))
print("missing copies:", [c["source"] for c in copied if c.get("status") == "MISSING"])
