"""Read-only evidence collection for the 11 open journal claims (2026-10-07).

Run on the workstation's WSL with the existing interpreter that has onnx:
    <WSL_PROJECT_HOME>/env/bin/python collect_claim_evidence.py

No checkpoint is unpickled (pickle opcodes are scanned statically), no model is
executed, nothing is trained, exported or quantized. Stored PyTorch/ONNX parity
outputs are re-compared; ONNX graphs are read with onnx.load. Outputs: ../evidence/*.json
"""
import collections
import glob
import hashlib
import json
import math
import os
import pickletools
import re
import statistics
import sys
import zipfile

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "evidence")
WIN = "<WINDOWS_USER_HOME>/OneDrive/Documents/Repository/FPT-Graduation-Project"
HIFI_CLONE = "<WSL_PROJECT_HOME>/hifimobinet-train/hifi-mobiNet"
Q05 = "<WSL_PROJECT_HOME>/paper_evidence/q05_2026-10-04T000830Z"
JEP = WIN + "/BanhmiTTS/journal-evidence-package"
CKPT = {
    "baseline": "<WSL_PROJECT_HOME>/baseline_v2_noclip/checkpoints/best-epoch=1489-val_loss_mel=19.7882.ckpt",
    "mrf": "<WSL_PROJECT_HOME>/mrf_111_run_500ep/checkpoints/best-epoch=1386-val_loss_mel=18.7681.ckpt",
    "seq": "<WSL_PROJECT_HOME>/seq111_final/checkpoints/best-epoch=1442-val_loss_mel=19.0136.ckpt",
    "piper": "<WSL_PROJECT_HOME>/piper_original/lj-med_1000.ckpt",
}
os.makedirs(OUT, exist_ok=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def dump(name, obj):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=str)


# ---------------------------------------------------------------- J-C011
def onnx_export_and_parity():
    out = {}
    for m in CKPT:
        rep = json.load(open(f"{Q05}/{m}/REPORT.json"))
        worst_nrmse = worst_abs = 0.0
        cases = []
        for p in sorted(glob.glob(f"{Q05}/{m}/parity_case_*.npz")):
            z = np.load(p, allow_pickle=False)
            a, b = z["pytorch"].astype(np.float64), z["onnx"].astype(np.float64)
            err = b - a
            nrmse = float(np.sqrt((err ** 2).mean()) / np.sqrt((a ** 2).mean()))
            mx = float(np.abs(err).max())
            cases.append({"case": os.path.basename(p), "input_ids": int(z["input"].shape[1]), "batch": int(z["input"].shape[0]),
                          "nrmse": nrmse, "max_abs": mx, "same_shape": a.shape == b.shape,
                          "finite": bool(np.isfinite(a).all() and np.isfinite(b).all())})
            worst_nrmse, worst_abs = max(worst_nrmse, nrmse), max(worst_abs, mx)
        deploy = onnx.load(f"{Q05}/{m}/{m}_fp32.onnx")
        valid = onnx.load(f"{Q05}/{m}/{m}_validation_external_noise.onnx")
        ia = {i.name: numpy_helper.to_array(i) for i in deploy.graph.initializer}
        ib = {i.name: numpy_helper.to_array(i) for i in valid.graph.initializer}
        ca = collections.Counter(n.op_type for n in deploy.graph.node)
        cb = collections.Counter(n.op_type for n in valid.graph.node)
        meta = {p.key: p.value for p in deploy.metadata_props}
        ck_sha = sha(CKPT[m])
        out[m] = {
            "checkpoint_sha256_now": ck_sha,
            "onnx_fp32_sha256": sha(f"{Q05}/{m}/{m}_fp32.onnx"),
            "onnx_metadata": meta,
            "onnx_metadata_checkpoint_matches": meta.get("checkpoint_sha256") == ck_sha,
            "report_status": rep["status"], "report_strict_load": rep.get("strict_load"), "report_epoch": rep.get("epoch"),
            "thresholds": rep.get("thresholds"),
            "recomputed_from_stored_outputs": {"cases": len(cases), "max_nrmse": worst_nrmse, "max_abs": worst_abs,
                                               "all_within_thresholds": worst_nrmse <= 1e-3 and worst_abs <= 5e-3, "per_case": cases},
            "validation_graph_vs_deployed": {
                "initializers": len(ia),
                "all_initializers_identical": set(ia) == set(ib) and all(np.array_equal(ia[k], ib[k]) for k in ia),
                "op_count_differences_deployed_vs_validation": {k: [ca.get(k, 0), cb.get(k, 0)] for k in set(ca) | set(cb)
                                                                if ca.get(k, 0) != cb.get(k, 0)},
            },
            "historical_named_weight_comparison": rep.get("historical_named_weight_comparison"),
        }
    dump("J-C011_onnx_export_parity.json", {
        "method": "parity re-computed from the PyTorch and ONNX outputs stored in Q05 parity_case_*.npz; no model executed",
        "models": out})


# ---------------------------------------------------------------- J-C024 / J-C025
def graph_structure():
    flow = {}
    for m in CKPT:
        g = onnx.load(f"{Q05}/{m}/{m}_fp32.onnx")
        init = {i.name: int(np.prod(i.dims)) for i in g.graph.initializer}
        flow_inputs = set()
        for n in g.graph.node:
            if n.name.startswith("/flow/"):
                flow_inputs.update(n.input)
        ops = collections.Counter(n.op_type for n in g.graph.node if n.name.startswith("/flow/"))
        wg = sum(v for k, v in init.items() if k.startswith("flow.") and k.endswith("weight_g"))
        flow[m] = {"flow_params_in_graph": sum(v for k, v in init.items() if k in flow_inputs or k.startswith("flow.")),
                   "flow_weight_g_elements": wg,
                   "flow_ops": {k: ops.get(k, 0) for k in ("Conv", "Softmax", "MatMul", "LayerNormalization")}}
    ptq = {}
    for label, path in {"baseline_ckpt1489": "<WSL_PROJECT_HOME>/quantized_models/ptq_sweep/baseline_flow_enc_p_dp.onnx",
                        "seq_ckpt1442": "/tmp/seq_final_onnx/seq_final_int8.onnx",
                        "mrf_source_unidentified": "<WSL_PROJECT_HOME>/quantized_models/ptq_sweep/mrf_flow_enc_p_dp.onnx"}.items():
        if not os.path.exists(path):
            ptq[label] = {"path": path, "status": "missing"}
            continue
        g = onnx.load(path)
        init = {i.name: i for i in g.graph.initializer}
        q = [n for n in g.graph.node if n.op_type in ("QLinearConv", "QLinearMatMul")]
        w_types, x_types, per_ch = collections.Counter(), collections.Counter(), 0
        by = collections.defaultdict(collections.Counter)
        for n in q:
            by[n.name.split("/")[1] if n.name.startswith("/") else "?"][n.op_type] += 1
            if n.op_type == "QLinearConv":
                w, ws, xzp = init.get(n.input[3]), init.get(n.input[4]), init.get(n.input[2])
                if w is not None:
                    w_types[TensorProto.DataType.Name(w.data_type)] += 1
                if ws is not None and int(np.prod(ws.dims)) > 1:
                    per_ch += 1
                if xzp is not None:
                    x_types[TensorProto.DataType.Name(xzp.data_type)] += 1
        dec_quant = sum(1 for n in g.graph.node if n.name.startswith("/dec/") and n.op_type.startswith(("QLinear", "Quantize")))
        ptq[label] = {"path": path, "sha256": sha(path), "producer": g.producer_name,
                      "qlinear_ops": dict(collections.Counter(n.op_type for n in q)),
                      "qlinearconv_per_channel_weight_scales": per_ch, "qlinearconv_weight_dtypes": dict(w_types),
                      "qlinearconv_activation_zero_point_dtypes": dict(x_types),
                      "by_prefix": {k: dict(v) for k, v in by.items()}, "quantized_ops_under_dec": dec_quant}
    dump("J-C024_J-C025_graph_structure.json", {
        "flow_in_q05_fp32_graphs": flow,
        "flow_counts_reference": {"internal_models_counted_from_code": 8285568, "piper_counted_weight_norm_folded": 7090560},
        "historical_int8_graphs": ptq,
        "note": "Piper's exported graph keeps weight_g/weight_v; flow_params_in_graph minus flow_weight_g_elements equals the folded count."})


# ---------------------------------------------------------------- J-C001
def checkpoint_key_structure():
    sys.path.insert(0, HIFI_CLONE + "/src")
    os.environ["HIFIMOBINET_HOME"] = HIFI_CLONE
    import warnings
    warnings.filterwarnings("ignore")
    param = re.compile(r"\.(weight|bias|weight_g|weight_v|gamma|beta|emb_rel_k|emb_rel_v|logs|m|alpha|_n_channels|hann_window)$")

    def keys(path):
        z = zipfile.ZipFile(path)
        b = z.read([n for n in z.namelist() if n.endswith("data.pkl")][0])
        return {a for op, a, _ in pickletools.genops(b) if isinstance(a, str) and a.startswith("model_") and param.search(a)}

    from hifimobinet.training.config import load_config
    from hifimobinet.training.registry import get_training_model
    from hifimobinet.training.vendor import vendor_module
    cfg = load_config(HIFI_CLONE + "/configs/training/baseline-resblock2-vits2.yaml")
    banhmi = set(get_training_model(cfg.model_id).builder()(num_symbols=256, **cfg.flat_hparams()).state_dict())
    E = vendor_module("edgetts_vits", "lightning").VitsModel
    edg_v2 = set(E(num_symbols=256, num_speakers=1, dataset=None, use_vits2=True).state_dict())
    edg_a = set(E(num_symbols=256, num_speakers=1, dataset=None).state_dict())
    base, piper = keys(CKPT["baseline"]), keys(CKPT["piper"])

    def cmp(a, b):
        return {"n_left": len(a), "n_right": len(b), "only_left": sorted(a - b), "only_right": sorted(b - a)}

    dump("J-C001_checkpoint_key_structure.json", {
        "method": "parameter/buffer names read statically from each checkpoint's data.pkl (pickletools.genops, no unpickling) "
                  "and compared with state_dict names of models constructed from code (random init)",
        "baseline_ckpt_vs_current_banhmitts_code": cmp(base, banhmi),
        "baseline_ckpt_vs_edgetts_use_vits2": cmp(base, edg_v2),
        "piper_ckpt_vs_edgetts_config_a_piper_base": cmp(piper, edg_a),
        "baseline_ckpt_vits2_markers": {
            "flow_attention_keys": sum(1 for k in base if ".flow.flows." in k and ".attn." in k),
            "duration_discriminator_keys": sum(1 for k in base if k.startswith("model_d_dur.")),
            "posterior_encoder_present": any(k.startswith("model_g.enc_q.") for k in base)},
        "limits": "Name-level structural equality, not byte identity of the executed source files.",
    })


# ---------------------------------------------------------------- J-C010
def calibration_membership():
    split = json.load(open(HIFI_CLONE + "/results/manifests/canonical_split.json"))
    sets = {k: set(v) for k, v in split.items() if isinstance(v, list)}
    path = f"{Q05}/inputs/calibration60.candidate.jsonl"
    rows = [json.loads(l) for l in open(path) if l.strip()]
    member = collections.Counter(tuple(n for n, s in sets.items() if r["audio_norm_path"] in s) for r in rows)
    hist = json.load(open(JEP + "/evidence/E_onnx_ptq/historical_calibration60_recomputed.json"))
    dump("J-C010_calibration.json", {
        "candidate_calibration": {"path": path, "sha256": sha(path), "rows": len(rows),
                                  "canonical_membership": {"+".join(k) or "none": v for k, v in member.items()}},
        "historical_calibration_recomputed_summary": {k: hist[k] for k in hist if k != "items"} if isinstance(hist, dict) else None,
        "ptq_runs_after_2026-10-04": sorted(p for p in glob.glob("<WSL_PROJECT_HOME>/**/*int8*.onnx", recursive=True)
                                           if os.path.getmtime(p) > 1791072000),  # after 2026-10-04T00:00Z
    })


# ---------------------------------------------------------------- J-C023
def onnx_speed_history():
    H = JEP + "/evidence/E_onnx_ptq/historical_results/"

    def load(f):
        return {it["text"]: it for it in json.load(open(H + f, encoding="utf-8"))}

    res = {}
    for a, b in (("eval_seq_final_fp32_onnx_results.json", "eval_baseline_fp32_onnx_500_results.json"),
                 ("eval_seq_final_int8_onnx_results.json", "eval_baseline_ptq_onnx_500_results.json"),
                 ("eval_seq_final_fp32_onnx_results.json", "eval_mrf_fp32_onnx_500_results.json")):
        A, B = load(a), load(b)
        common = [t for t in A if t in B]
        r = [A[t]["rtf"] / B[t]["rtf"] for t in common]
        res[f"{a} vs {b}"] = {"paired_sentences": len(common),
                              "mean_rtf_left": statistics.mean(x["rtf"] for x in A.values()),
                              "mean_rtf_right": statistics.mean(x["rtf"] for x in B.values()),
                              "left_faster_count": sum(1 for x in r if x < 1),
                              "median_ratio": statistics.median(r) if r else None,
                              "geometric_mean_ratio": math.exp(statistics.mean(math.log(x) for x in r)) if r else None}
    dump("J-C023_onnx_speed_history.json", {
        "graphs": {"baseline_fp32": "<WSL_PROJECT_HOME>/quantized_models/ptq_sweep/baseline_fp32.onnx (weights match ckpt 1489, JEP)",
                   "baseline_int8": "ptq_sweep/baseline_flow_enc_p_dp.onnx",
                   "seq_fp32/int8": "/tmp/seq_final_onnx/seq_final_{fp32,int8}.onnx (454/454 named weights match ckpt 1442, Q05)",
                   "mrf": "graph source unidentified (0/478 tensors match ckpt 1386, JEP); different 500 sentences"},
        "runtime": "ONNX Runtime CPU, intra-op 2 threads, inter-op 1, no warm-up, one timing per utterance (round7 audit)",
        "comparisons": res})


# ---------------------------------------------------------------- J-C026
def harvard_exposure():
    pub_cfg = os.path.join(OUT, "public_piper_lj_med_config.json")
    b = json.load(open("<WSL_PROJECT_HOME>/BanhmiTTS/training_output/ljspeech/medium/config.json", encoding="utf-8"))
    out = {"piper_checkpoint_sha256_now": sha(CKPT["piper"]),
           "piper_checkpoint_hparams": {k: v for k, v in json.load(open(WIN + "/BanhmiTTS/docs/evidence/q05_2026-10-04/piper/checkpoint_hparams.json")).items()
                                        if k in ("dataset", "dataset_dir", "max_epochs", "resume_from_checkpoint", "num_speakers",
                                                 "validation_split", "num_test_examples", "max_phoneme_ids")}}
    if os.path.exists(pub_cfg):
        p = json.load(open(pub_cfg, encoding="utf-8"))
        pm, bm = p["phoneme_id_map"], b["phoneme_id_map"]
        extra_ids = {i for k in set(bm) - set(pm) for i in bm[k]}
        harvard = json.load(open(JEP + "/evidence/B_harvard/raw/harvard_testset_720.json", encoding="utf-8"))
        piper_ids = {i for v in pm.values() for i in v}
        out["phoneme_id_maps"] = {"piper_symbols": len(pm), "project_symbols": len(bm),
                                  "common_with_different_ids": sum(1 for k in set(pm) & set(bm) if pm[k] != bm[k]),
                                  "project_only_ids": sorted(extra_ids),
                                  "harvard_items_using_ids_outside_piper_map": sum(1 for it in harvard if set(it["phoneme_ids"]) - piper_ids)}
    rows = sum(1 for _ in open("<WSL_PROJECT_HOME>/BanhmiTTS/training_output/ljspeech/medium/dataset.jsonl"))
    out["internal_training_dataset_rows"] = rows
    out["harvard_vs_ljspeech_text_overlap"] = "see journal-evidence-package evidence/C_data_split/data_split_checks_recomputed.json"
    dump("J-C026_harvard_exposure.json", out)


if __name__ == "__main__":
    for step in (onnx_export_and_parity, graph_structure, checkpoint_key_structure, calibration_membership,
                 onnx_speed_history, harvard_exposure):
        print("running", step.__name__, flush=True)
        step()
    print("done")
