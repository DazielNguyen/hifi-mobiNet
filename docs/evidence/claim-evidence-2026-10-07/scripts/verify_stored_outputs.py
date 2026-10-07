"""Re-derive the claim-evidence numbers from stored records (no model is executed or loaded).

    python verify_stored_outputs.py [--output report.json]
        [--parity-dir DIR]      # transfer bundle q05-parity/: {model}/parity_case_XX.npz
        [--dataset-jsonl FILE]  # transfer bundle dataset/dataset.jsonl
        [--graph-dir DIR]       # any folder holding the ONNX graphs listed in artifact-manifest.json

Paths default to this package and the hifi-mobiNet checkout that contains it. The base
checks need only the Python standard library and the files tracked in Git. --parity-dir
needs numpy; --graph-dir needs numpy and onnx (graphs are parsed, never run); the
WER-normalisation overlap check under --dataset-jsonl also uses jiwer when installed.
Every check records pass/fail; the exit status is 1 if any check fails.
"""
import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ("baseline", "mrf", "seq", "piper")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Report:
    def __init__(self):
        self.sections, self.failed = {}, []

    def check(self, section, name, ok, **detail):
        self.sections.setdefault(section, {"checks": []})["checks"].append({"check": name, "passed": bool(ok), **detail})
        if not ok:
            self.failed.append(f"{section}: {name}")

    def put(self, section, key, value):
        self.sections.setdefault(section, {"checks": []})[key] = value


def package_integrity(r):
    m = load(os.path.join(PKG, "source-manifest.json"))
    bad = []
    for e in m["files"]:
        p = os.path.join(PKG, e["destination"].split("claim-evidence-2026-10-07/", 1)[1])
        if not os.path.exists(p) or sha256(p) != e["destination_sha256"]:
            bad.append(e["destination"])
    r.check("package", "every copied file matches destination_sha256 in source-manifest.json", not bad,
            files=len(m["files"]), mismatched=bad)


def j_c011(r):
    q = os.path.join(PKG, "source-records", "q05")
    cfg = load(os.path.join(q, "RUN_CONFIG.json"))
    graphs = {g["metadata_props"].get("q05_model"): g for g in load(os.path.join(PKG, "source-records", "onnx", "onnx_inventory_extract.json"))["graphs"]
              if g["metadata_props"].get("q05_model")}
    rows = {}
    for m in MODELS:
        rep = load(os.path.join(q, f"REPORT.{m}.json"))
        cases = rep["parity_cases"]
        meta = graphs[m]["metadata_props"]
        r.check("J-C011", f"{m}: graph metadata checkpoint_sha256 equals the report's checkpoint SHA-256",
                meta["checkpoint_sha256"] == rep["checkpoint"]["sha256"] == cfg["checkpoints"][m]["sha256"],
                metadata_checkpoint_sha256=meta["checkpoint_sha256"], compared_with=rep["checkpoint"]["sha256"])
        r.check("J-C011", f"{m}: inventory graph SHA-256 equals the report's graph SHA-256",
                graphs[m]["sha256"] == rep["graph"]["sha256"], graph_sha256=rep["graph"]["sha256"])
        r.check("J-C011", f"{m}: report thresholds equal RUN_CONFIG thresholds", rep["thresholds"] == cfg["thresholds_declared_before_runs"])
        r.check("J-C011", f"{m}: RUN_CONFIG created before this model run started", cfg["created_utc"] < rep["started_utc"],
                run_config_created_utc=cfg["created_utc"], model_started_utc=rep["started_utc"])
        stored_pass = all(c["passed"] and c["same_shape"] and c["finite"] for c in cases)
        r.check("J-C011", f"{m}: all stored parity cases passed", stored_pass and rep["status"] == "passed", cases=len(cases))
        hist = rep.get("historical_named_weight_comparison")
        rows[m] = {"checkpoint_sha256": rep["checkpoint"]["sha256"], "epoch_zero_based": rep["epoch"],
                   "fp32_graph_sha256": rep["graph"]["sha256"], "validation_graph_sha256": rep["validation_graph"]["sha256"],
                   "cases": len(cases), "max_nrmse_stored": max(c["nrmse"] for c in cases),
                   "mean_nrmse_stored": statistics.mean(c["nrmse"] for c in cases),
                   "max_abs_error_stored": max(c["max_abs_error"] for c in cases),
                   "historical_graph_named_weights": None if hist is None else
                   {"graph_sha256": hist["sha256"], "exact": hist["exact_matches"], "mismatch": hist["mismatches"]}}
    r.put("J-C011", "per_model", rows)
    r.put("J-C011", "formula", "err = pytorch - onnx; nrmse = sqrt(mean(err^2)) / max(sqrt(mean(pytorch^2)), 1e-12); "
          "pass = same shape AND finite AND nrmse <= 0.001 AND max|err| <= 0.005; aggregate = all 10 cases pass (max reported)")


def j_c011_parity(r, parity_dir):
    import numpy as np
    q = os.path.join(PKG, "source-records", "q05")
    out = {}
    for m in MODELS:
        rep = load(os.path.join(q, f"REPORT.{m}.json"))
        vals = []
        for i, c in enumerate(rep["parity_cases"]):
            p = os.path.join(parity_dir, m, f"parity_case_{i:02d}.npz")
            ok_hash = sha256(p) == c["arrays_sha256"]
            z = np.load(p)
            a, b = z["pytorch"].astype(np.float64), z["onnx"].astype(np.float64)
            same = a.shape == b.shape
            err = a - b
            nrmse = float(np.sqrt(np.mean(err ** 2)) / max(np.sqrt(np.mean(a ** 2)), 1e-12))
            mx = float(np.max(np.abs(err)))
            vals.append(nrmse)
            r.check("J-C011 parity", f"{m} case {i:02d}: npz hash, recomputed NRMSE and max|err| match the report and thresholds",
                    ok_hash and same and np.isfinite(b).all() and math.isclose(nrmse, c["nrmse"], rel_tol=1e-4, abs_tol=1e-12)
                    and nrmse <= 1e-3 and mx <= 5e-3, nrmse=nrmse, max_abs_error=mx)
        out[m] = {"cases": len(vals), "max_nrmse_recomputed": max(vals)}
    r.put("J-C011 parity", "per_model", out)


def j_c023(r, repo):
    H = os.path.join(repo, "results", "historical", "onnx_historical")

    def rows(f):
        return {it["text"]: it for it in load(os.path.join(H, f))}

    res = {}
    for a, b in (("eval_seq_final_fp32_onnx_results.json", "eval_baseline_fp32_onnx_500_results.json"),
                 ("eval_seq_final_int8_onnx_results.json", "eval_baseline_ptq_onnx_500_results.json"),
                 ("eval_seq_final_fp32_onnx_results.json", "eval_mrf_fp32_onnx_500_results.json")):
        A, B = rows(a), rows(b)
        common = [t for t in A if t in B]
        ratio = [A[t]["rtf"] / B[t]["rtf"] for t in common]
        res[f"{a} vs {b}"] = {"paired_sentences": len(common), "left_faster": sum(x < 1 for x in ratio),
                              "median_rtf_ratio": statistics.median(ratio),
                              "left_sha256": sha256(os.path.join(H, a)), "right_sha256": sha256(os.path.join(H, b))}
    v = list(res.values())
    r.check("J-C023", "SEQ FP32 faster than baseline FP32 on 495/500 paired sentences", (v[0]["paired_sentences"], v[0]["left_faster"]) == (500, 495))
    r.check("J-C023", "SEQ INT8 faster than baseline INT8 on 498/500 paired sentences", (v[1]["paired_sentences"], v[1]["left_faster"]) == (500, 498))
    r.check("J-C023", "SEQ vs MRF historical runs share only 21 sentences", v[2]["paired_sentences"] == 21)
    r.put("J-C023", "comparisons", res)


def j_c019(r, repo):
    p = os.path.join(repo, "results", "prior_audit", "harvard_stats_recomputed.json")
    t = load(p)["paired_tests"]
    w, u = t["seq_vs_mrf_wer"], t["seq_vs_mrf_utmosv2"]
    lo, hi = w["exploratory_paired_bootstrap95_mean_diff"]
    r.check("J-C019", "WER SEQ-MRF bootstrap CI is [+0.0019, +0.0277] (fraction scale)", round(lo, 4) == 0.0019 and round(hi, 4) == 0.0277)
    r.put("J-C019", "values", {
        "source": "results/prior_audit/harvard_stats_recomputed.json", "source_sha256": sha256(p),
        "wer": {"scale": "fraction of words, macro mean over 720 utterances", "direction": "SEQ minus MRF (positive = SEQ worse)",
                "mean_diff": w["mean_diff_x_minus_y"], "ci95": [lo, hi], "ci95_percentage_points": [lo * 100, hi * 100],
                "wilcoxon_p": w["wilcoxon_p_scipy_default"], "holm_p": w["exploratory_holm_adjusted_wilcoxon_p_over_15_tests"]},
        "utmosv2": {"direction": "SEQ minus MRF", "mean_diff": u["mean_diff_x_minus_y"],
                    "ci95": u["exploratory_paired_bootstrap95_mean_diff"], "wilcoxon_p": u["wilcoxon_p_scipy_default"]},
        "interpretation_limit": "a CI that excludes 0 is not an equivalence test; no equivalence margin was declared"})


def j_c010(r, repo):
    split = {k: set(v) for k, v in load(os.path.join(repo, "results", "manifests", "canonical_split.json")).items()}
    cal = os.path.join(PKG, "source-records", "calibration")
    cand = [json.loads(l) for l in open(os.path.join(cal, "calibration60.candidate.jsonl"), encoding="utf-8") if l.strip()]
    mc = collections.Counter(next((k for k, s in split.items() if x["audio_norm_path"] in s), "none") for x in cand)
    r.check("J-C010", "candidate (designed, unused) calibration set is 60/60 canonical train", dict(mc) == {"train": 60})
    hist = load(os.path.join(cal, "historical_calibration60_recomputed.json"))
    mh = collections.Counter(next((k for k, s in split.items() if i in s), "none") for i in hist["ids"])
    test_ids = [i for i in hist["ids"] if i in split["test"]]
    r.check("J-C010", "historical (reconstructed) calibration set is 57 train + 3 test", dict(mh) == {"train": 57, "test": 3})
    r.put("J-C010", "historical_test_items", [hist["texts"][hist["ids"].index(i)] for i in test_ids])
    return hist


def j_c010_dataset(r, dataset, hist):
    entries = []
    with open(dataset, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    sample = random.Random(42).sample(entries, 60)  # ptq_quantize.py JsonlCalibReader, seed 42
    r.check("J-C010 dataset", "dataset.jsonl SHA-256 equals the recorded calibration source", sha256(dataset) == hist["dataset_sha256"])
    r.check("J-C010 dataset", "re-sampled 60 rows equal the reconstructed historical calibration ids",
            [e["audio_norm_path"] for e in sample] == hist["ids"])
    return entries


def j_c026(r, repo, entries):
    pub = os.path.join(PKG, "source-records", "public")
    listing = load(os.path.join(pub, "public_piper_checkpoints_listing.json"))
    lfs = next(x["lfs"]["oid"] for x in listing if x["path"].endswith("lj-med_1000.ckpt"))
    art = load(os.path.join(PKG, "artifact-manifest.json"))
    piper = next(a for a in art["referenced_not_copied"] if a["path"].endswith("lj-med_1000.ckpt"))
    r.check("J-C026", "project Piper checkpoint SHA-256 equals the published LFS oid", piper["sha256"] == lfs, sha256=lfs)
    card = open(os.path.join(pub, "public_piper_voices_ljspeech_medium_MODEL_CARD.txt"), "rb").read()
    blob = hashlib.sha1(b"blob %d\0" % len(card) + card).hexdigest()
    rev = load(os.path.join(pub, "public_revision_record_2026-10-07.json"))
    r.check("J-C026", "stored MODEL_CARD git blob SHA-1 equals the published blob oid", blob == rev["stored_model_card"]["git_blob_sha1"], blob=blob)
    r.check("J-C026", "MODEL_CARD states training from scratch on LJ Speech",
            b"Trained from scratch for 1000 epochs" in card and b"LJ Speech" in card)
    pm = load(os.path.join(pub, "public_piper_lj_med_config.json"))["phoneme_id_map"]
    bm = load(os.path.join(PKG, "source-records", "project", "ljspeech_medium_config.json"))["phoneme_id_map"]
    harv = load(os.path.join(repo, "results", "manifests", "harvard720.json"))
    piper_ids = {i for v in pm.values() for i in v}
    outside = sum(1 for h in harv if set(h["phoneme_ids"]) - piper_ids)
    r.check("J-C026", "shared phoneme symbols have identical ids; no Harvard item uses an id outside Piper's map",
            sum(1 for k in set(pm) & set(bm) if pm[k] != bm[k]) == 0 and outside == 0,
            piper_symbols=len(pm), project_symbols=len(bm))
    if entries is None:
        rec = load(os.path.join(PKG, "source-records", "data-overlap", "data_split_checks_recomputed.json"))
        r.put("J-C026", "text_overlap", {"status": "taken from stored record (pass --dataset-jsonl to recompute)",
                                         **{k: rec[k] for k in rec if k.startswith(("exact_", "harvard_sentences_sharing"))}})
        return
    n1 = lambda t: " ".join(re.sub(r"[^a-z ]", "", t.lower()).split())
    res = {"n_ljspeech": len(entries), "n_harvard": len(harv)}
    lj = {n1(x["text"]) for x in entries}
    res["exact_sentence_overlap_alnum_lower"] = sum(n1(h["text"]) in lj for h in harv)
    try:
        import jiwer
        wt = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
        ljw = {wt(x["text"]) for x in entries}
        res["exact_sentence_overlap_wer_normalization"] = sum(wt(h["text"]) in ljw for h in harv)
    except ImportError:
        res["exact_sentence_overlap_wer_normalization"] = "skipped (jiwer not installed)"
    for n in (4, 5, 6):
        grams = set()
        for x in entries:
            w = n1(x["text"]).split()
            grams.update(" ".join(w[i:i + n]) for i in range(len(w) - n + 1))
        res[f"harvard_sentences_sharing_any_{n}gram"] = sum(
            1 for h in harv if any(" ".join(n1(h["text"]).split()[i:i + n]) in grams for i in range(len(n1(h["text"]).split()) - n + 1)))
    lj_ph = {tuple(x["phoneme_ids"]) for x in entries}
    res["exact_phoneme_id_sequence_overlap"] = sum(tuple(h["phoneme_ids"]) in lj_ph for h in harv)
    rec = load(os.path.join(PKG, "source-records", "data-overlap", "data_split_checks_recomputed.json"))
    keys = [k for k in res if k in rec and not isinstance(res[k], str)]
    r.check("J-C026", "recomputed text/phoneme overlap equals the stored record", all(res[k] == rec[k] for k in keys), compared=keys)
    r.put("J-C026", "text_overlap", res)


def j_c001(r):
    d = load(os.path.join(PKG, "outputs", "J-C001_key_inventories.json"))
    inv = d["inventories"]
    out = {}
    for label, c in d["comparisons"].items():
        lname, rname = label.split("_vs_")
        L, R = set(inv[lname]["names"]), set(inv[rname]["names"])
        ok = (len(L), len(R), len(L & R)) == (c["left_count"], c["right_count"], c["common"])
        r.check("J-C001", f"{label}: counts re-derived from the stored name inventories", ok)
        out[label] = {"left": len(L), "right": len(R), "common": len(L & R), "only_right": sorted(R - L), "only_left": sorted(L - R)}
    r.check("J-C001", "baseline checkpoint names equal the relocated BanhmiTTS code names (876/876)",
            out["baseline_checkpoint_vs_code_banhmitts_baseline_recipe"]["common"] == 876
            and not out["baseline_checkpoint_vs_code_banhmitts_baseline_recipe"]["only_left"]
            and not out["baseline_checkpoint_vs_code_banhmitts_baseline_recipe"]["only_right"])
    r.put("J-C001", "comparisons", out)
    r.put("J-C001", "basis", d["comparison_basis"])


def graphs(r, graph_dir):
    import numpy as np
    import onnx
    from onnx import TensorProto
    art = load(os.path.join(PKG, "artifact-manifest.json"))
    want = {a["sha256"]: a for a in art["referenced_not_copied"] if a["evidence_type"].startswith("onnx")}
    found = {}
    for root, _, files in os.walk(graph_dir):
        for f in files:
            if f.endswith(".onnx"):
                p = os.path.join(root, f)
                h = sha256(p)
                if h in want:
                    found[want[h]["path"]] = p
    r.put("graphs", "matched_by_sha256", sorted(found))

    by_path = {a["path"]: a for a in want.values()}
    sha_of = {}

    def pick(suffix, evidence_type):
        hits = [k for k in found if k.endswith(suffix) and by_path[k]["evidence_type"] == evidence_type]
        if not hits:
            return None
        sha_of[found[hits[0]]] = by_path[hits[0]]["sha256"]
        return found[hits[0]]

    flow = {}
    for m in MODELS:
        p = pick(f"/{m}/{m}_fp32.onnx", "onnx_fp32_q05")
        if not p:
            continue
        g = onnx.load(p)
        init = {i.name: int(np.prod(i.dims)) for i in g.graph.initializer}
        flow_nodes = [n for n in g.graph.node if n.name.startswith("/flow/")]
        used_in_flow = {i for n in flow_nodes for i in n.input if i in init}
        used_elsewhere = {i for n in g.graph.node if not n.name.startswith("/flow/") for i in n.input if i in init}
        names = used_in_flow | {k for k in init if k.startswith("flow.")}
        const_elems = sum(int(np.prod(a.t.dims)) for n in flow_nodes if n.op_type == "Constant" for a in n.attribute if a.name == "value")
        ops = collections.Counter(n.op_type for n in flow_nodes)
        flow[m] = {"graph_sha256": sha_of[p], "flow_initializer_elements": sum(init[k] for k in names),
                   "flow_bias_elements": sum(init[k] for k in names if k.endswith(".bias")),
                   "flow_weight_g_elements": sum(init[k] for k in names if k.endswith("weight_g")),
                   "flow_initializers_shared_with_non_flow_nodes": sorted(names & used_elsewhere),
                   "flow_constant_node_elements": const_elems,
                   "flow_ops": {k: ops.get(k, 0) for k in ("Conv", "MatMul", "Softmax", "LayerNormalization")}}
    if flow:
        r.put("J-C025", "flow_in_q05_graphs", flow)
        for m, v in flow.items():
            expect = 7090560 if m == "piper" else 8285568
            r.check("J-C025", f"{m}: flow initializer elements (weight_g removed) equal {expect}",
                    v["flow_initializer_elements"] - v["flow_weight_g_elements"] == expect)
    ptq = {}
    for name in ("baseline_flow_enc_p_dp.onnx", "mrf_flow_enc_p_dp.onnx", "seq_final_int8.onnx"):
        p = pick("/" + name, "onnx_historical")
        if not p:
            continue
        g = onnx.load(p)
        init = {i.name: i for i in g.graph.initializer}
        q = [n for n in g.graph.node if n.op_type in ("QLinearConv", "QLinearMatMul")]
        by = collections.defaultdict(collections.Counter)
        wt, zt, perch = collections.Counter(), collections.Counter(), 0
        for n in q:
            by[n.name.split("/")[1] if n.name.startswith("/") else "?"][n.op_type] += 1
            if n.op_type == "QLinearConv":
                w, ws, xz = init.get(n.input[3]), init.get(n.input[4]), init.get(n.input[2])
                if w is not None:
                    wt[TensorProto.DataType.Name(w.data_type)] += 1
                if ws is not None and int(np.prod(ws.dims)) > 1:
                    perch += 1
                if xz is not None:
                    zt[TensorProto.DataType.Name(xz.data_type)] += 1
        ptq[name] = {"graph_sha256": sha_of[p], "qlinear_ops": dict(collections.Counter(n.op_type for n in q)), "by_top_module": {k: dict(v) for k, v in by.items()},
                     "qlinearconv_weight_dtypes": dict(wt), "qlinearconv_input_zero_point_dtypes": dict(zt),
                     "qlinearconv_per_channel": perch,
                     "quantized_ops_under_dec": sum(1 for n in g.graph.node if n.name.startswith("/dec/") and n.op_type.startswith(("QLinear", "Quantize")))}
        r.check("J-C024", f"{name}: 133 QLinearConv, all INT8 per-channel weights, UINT8 activations, none under /dec/",
                ptq[name]["qlinear_ops"].get("QLinearConv") == 133 and wt == {"INT8": 133} and perch == 133
                and zt == {"UINT8": 133} and ptq[name]["quantized_ops_under_dec"] == 0)
    if ptq:
        r.put("J-C024", "int8_graphs", ptq)
    # Lineage of each historical INT8 graph: the decoder stays FP32 under PTQ, so its initializers
    # should be byte-identical to those of the FP32 graph the INT8 graph was produced from.
    links = {}
    for int8, fp32 in (("baseline_flow_enc_p_dp.onnx", "ptq_sweep/baseline_fp32.onnx"),
                       ("mrf_flow_enc_p_dp.onnx", "ptq_sweep/mrf_fp32.onnx"),
                       ("seq_final_int8.onnx", "seq_final_onnx/seq_final_fp32.onnx")):
        pi, pf = pick("/" + int8, "onnx_historical"), pick("/" + fp32, "onnx_historical")
        if not (pi and pf):
            continue
        gi, gf = onnx.load(pi), onnx.load(pf)
        fi = {i.name: i for i in gf.graph.initializer}
        dec_in = {x for n in gi.graph.node if n.name.startswith("/dec/") for x in n.input}
        dec = [i for i in gi.graph.initializer if i.name in dec_in]
        same = sum(1 for i in dec if i.name in fi and onnx.numpy_helper.to_array(i).tobytes() == onnx.numpy_helper.to_array(fi[i.name]).tobytes()
                   and i.dims == fi[i.name].dims)
        links[f"{int8} <- {fp32}"] = {"int8_sha256": sha_of[pi], "fp32_sha256": sha_of[pf],
                                      "decoder_initializers": len(dec), "byte_identical_in_fp32_graph": same}
    if links:
        r.put("J-C023", "int8_to_fp32_decoder_initializer_match", links)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", default=os.path.abspath(os.path.join(PKG, "..", "..", "..")))
    ap.add_argument("--output", default=os.path.join(PKG, "outputs", "verify_report.json"))
    ap.add_argument("--parity-dir")
    ap.add_argument("--dataset-jsonl")
    ap.add_argument("--graph-dir")
    a = ap.parse_args()
    r = Report()
    package_integrity(r)
    j_c001(r)
    j_c011(r)
    if a.parity_dir:
        j_c011_parity(r, a.parity_dir)
    j_c023(r, a.repo_root)
    j_c019(r, a.repo_root)
    hist = j_c010(r, a.repo_root)
    entries = j_c010_dataset(r, a.dataset_jsonl, hist) if a.dataset_jsonl else None
    j_c026(r, a.repo_root, entries)
    if a.graph_dir:
        graphs(r, a.graph_dir)
    out = {"options": {"parity_dir": bool(a.parity_dir), "dataset_jsonl": bool(a.dataset_jsonl), "graph_dir": bool(a.graph_dir)},
           "python": sys.version.split()[0], "checks_total": sum(len(s["checks"]) for s in r.sections.values()),
           "checks_failed": r.failed, "sections": r.sections}
    with open(a.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"{out['checks_total']} checks, {len(r.failed)} failed -> {os.path.relpath(a.output)}")
    for x in r.failed:
        print("FAILED:", x)
    sys.exit(1 if r.failed else 0)


if __name__ == "__main__":
    main()
