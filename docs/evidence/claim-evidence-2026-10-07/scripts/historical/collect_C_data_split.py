"""Part C: dataset/split evidence. RECOMPUTED now from stored manifests; nothing is retrained.
- copies canonical_split.json, the reconstructed MRF split, Piper download metadata (byte copies)
- Harvard-vs-LJSpeech text overlap under two normalizations + n-gram + phoneme-ID checks
- split overlap numbers (95.4% etc.) with explicit numerators/denominators
- Piper checkpoint hash: WSL copy vs Windows download copy
- compact hparams table from the four checkpoints (via evidence/A_inventory/checkpoint_metadata_now.json)
Usage: collect_C_data_split.py <package_root>
"""
import hashlib
import json
import os
import re
import shutil
import sys

import jiwer

PKG = sys.argv[1]
OUT = os.path.join(PKG, "evidence", "C_data_split")
os.makedirs(OUT, exist_ok=True)
DS = "<WSL_PROJECT_HOME>/BanhmiTTS/training_output/ljspeech/medium/dataset.jsonl"
CANON = "<WSL_PROJECT_HOME>/BanhmiTTS/training_output/ljspeech/medium/canonical_split.json"
WIN = "<WINDOWS_USER_HOME>/OneDrive/Documents/Repository/FPT-Graduation-Project"
MRF_SPLIT = f"{WIN}/BanhmiTTS/docs/evidence/q03_bosung_2026-10-04/MRF_SPLIT_RECONSTRUCTED_VALIDATED.json"
PIPER_DL = f"{WIN}/piper/download/checkpoints/en_US-ljspeech-medium"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


copies = []
for s, name in [(CANON, "canonical_split.json"), (MRF_SPLIT, "MRF_SPLIT_RECONSTRUCTED_VALIDATED.json"),
                (f"{PIPER_DL}/MODEL_CARD", "piper_MODEL_CARD"), (f"{PIPER_DL}/config.json", "piper_config.json"),
                (f"{PIPER_DL}/download_manifest.json", "piper_download_manifest.json")]:
    d = os.path.join(OUT, name)
    shutil.copy2(s, d)
    copies.append({"source": s, "copy": os.path.relpath(d, PKG), "sha256": sha(d)})

ds = [json.loads(l) for l in open(DS, encoding="utf-8")]
harv = json.load(open("/tmp/harvard_testset_720.json"))
n1 = lambda t: " ".join(re.sub(r"[^a-z ]", "", t.lower()).split())          # build_harvard_testset.py normalization
wt = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
n2 = lambda t: wt(t)                                                         # WER normalization used for scoring
res = {"dataset_jsonl": DS, "dataset_jsonl_sha256": sha(DS), "n_ljspeech_utterances_checked": len(ds),
       "n_harvard": len(harv)}
for tag, f in (("alnum_lower", n1), ("wer_normalization", n2)):
    lj = {f(x["text"]) for x in ds}
    res[f"exact_sentence_overlap_{tag}"] = sum(f(h["text"]) in lj for h in harv)
    res[f"harvard_internal_duplicates_{tag}"] = len(harv) - len({f(h["text"]) for h in harv})
for N in (4, 5, 6):
    grams = set()
    for x in ds:
        w = n1(x["text"]).split()
        grams.update(" ".join(w[i:i + N]) for i in range(len(w) - N + 1))
    hits = [h["text"] for h in harv if any(" ".join(n1(h["text"]).split()[i:i + N]) in grams
                                           for i in range(len(n1(h["text"]).split()) - N + 1))]
    res[f"harvard_sentences_sharing_any_{N}gram"] = len(hits)
    if N <= 5:
        res[f"examples_{N}gram"] = hits[:10]
lj_ph = {tuple(x["phoneme_ids"]) for x in ds}
res["exact_phoneme_id_sequence_overlap"] = sum(tuple(h["phoneme_ids"]) in lj_ph for h in harv)
res["scope_note"] = ("Checked only against the 13,100 LJSpeech transcripts in BanhmiTTS dataset.jsonl. "
                     "Not checked: Piper's own training transcripts, audio-level similarity, the scorers' (Whisper/UTMOSv2) "
                     "training data, or any other corpus. Exact/n-gram non-overlap is not proof that no model saw related data.")

canon = json.load(open(CANON))
mrf = json.load(open(MRF_SPLIT))["split"]
inter = len(set(canon["train"]) & set(mrf["train"]))
res["split_overlap"] = {
    "train_canonical_intersect_train_mrf": inter,
    "denominator_train_size": len(canon["train"]),
    "fraction": inter / len(canon["train"]),
    "formula": "|train_canonical ∩ train_MRF_reconstructed| / 12,500",
    "canonical_test_in_mrf": {k: len(set(canon["test"]) & set(v)) for k, v in mrf.items()},
    "mrf_test_in_canonical": {k: len(set(mrf["test"]) & set(v)) for k, v in canon.items()},
    "canonical_val_in_mrf": {k: len(set(canon["val"]) & set(v)) for k, v in mrf.items()},
    "sources": {"canonical": "canonical_split.json (current file; birth=ctime=mtime 2026-09-19T21:55Z)",
                "mrf": "reconstructed split, corroborated by TensorBoard val heads and the 2026-09-15 test manifest"},
}

res["piper_checkpoint_hash"] = {"wsl_copy": "<WSL_PROJECT_HOME>/piper_original/lj-med_1000.ckpt",
                                "wsl_sha256": sha("<WSL_PROJECT_HOME>/piper_original/lj-med_1000.ckpt"),
                                "windows_download": f"{PIPER_DL}/lj-med_1000.ckpt",
                                "windows_sha256": sha(f"{PIPER_DL}/lj-med_1000.ckpt")}
res["piper_checkpoint_hash"]["identical"] = res["piper_checkpoint_hash"]["wsl_sha256"] == res["piper_checkpoint_hash"]["windows_sha256"]
res["copies"] = copies

meta = json.load(open(os.path.join(PKG, "evidence", "A_inventory", "checkpoint_metadata_now.json")))
KEYS = ["resblock", "resblock_kernel_sizes", "resblock_dilation_sizes", "mb_expansion", "upsample_rates", "upsample_kernel_sizes",
        "upsample_initial_channel", "hidden_channels", "inter_channels", "filter_channels", "n_layers", "n_heads", "kernel_size",
        "p_dropout", "flow_n_flows", "flow_kernel_size", "flow_dilation_rate", "posterior_encoder_layers",
        "mas_noise_scale_initial", "mas_noise_scale_decay", "c_mel", "c_kl", "learning_rate", "betas", "eps", "weight_decay",
        "lr_decay", "precision", "batch_size", "devices", "strategy", "seed", "gradient_clip_val", "max_epochs",
        "num_val_examples", "num_test_examples", "validation_split", "max_phoneme_ids", "segment_size", "sample_rate",
        "use_snake", "use_mrd", "use_vocos", "use_f0", "num_workers", "dataset"]
table = {}
for name, m in meta.items():
    if name == "note":
        continue
    hp = m["stored_in_checkpoint"]["hyper_parameters"]
    table[name] = {k: hp.get(k, "<absent>") for k in KEYS}
    table[name]["_epoch"] = m["stored_in_checkpoint"]["epoch"]
    table[name]["_global_step"] = m["stored_in_checkpoint"]["global_step"]
    table[name]["_pl_version"] = m["stored_in_checkpoint"]["pytorch_lightning_version"]
    table[name]["_optimizer_param_groups"] = m["stored_in_checkpoint"]["optimizer_param_groups"]
    table[name]["_lr_schedulers"] = m["stored_in_checkpoint"]["lr_schedulers"]
    table[name]["_callbacks"] = m["stored_in_checkpoint"]["callbacks"]
    table[name]["_sha256"] = m["sha256_now"]
res["hparams_from_checkpoints"] = table
json.dump(res, open(os.path.join(OUT, "data_split_checks_recomputed.json"), "w"), indent=1, ensure_ascii=False, default=str)
print(json.dumps({k: v for k, v in res.items() if k not in ("hparams_from_checkpoints", "copies")}, indent=1, ensure_ascii=False))
