"""Reconstructs MRF's train/val/test split with the pre-2026-09-19 procedure
(train.py: torch.manual_seed(seed) -> VitsModel(...) weight init -> ambient-RNG
random_split) and validates it against two independent anchors:
  1. the 5 validation-sample tags MRF's own TensorBoard logs recorded during
     all three training phases (contemporaneous, ordered = val[0:5]);
  2. /tmp/eval_true_testset_500_mrf.json, written 2026-09-15T22:34Z by
     /tmp/build_mrf_testset.py with the then-current code (ordered test list).
A reconstruction is only accepted if it reproduces both exactly, in order.
No training, no inference: the model is constructed and discarded.
"""
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, "<WSL_PROJECT_HOME>/BanhmiTTS")
import torch
from torch.utils.data import random_split

torch.serialization.add_safe_globals([pathlib.PosixPath])
from banhmi_train.vits.dataset import VitsDataset
from banhmi_train.vits.training import VitsModel

DATASET = "<WSL_PROJECT_HOME>/BanhmiTTS/training_output/ljspeech/medium/dataset.jsonl"
CKPT = "<WSL_PROJECT_HOME>/mrf_111_run_500ep/checkpoints/best-epoch=1386-val_loss_mel=18.7681.ckpt"
TB = "/tmp/q03_tb_audio_tags.json"
OLD_TEST = "/tmp/eval_true_testset_500_mrf.json"
OUT = "/tmp/q03_mrf_split_reconstructed.json"

hp = dict(torch.load(CKPT, map_location="cpu", weights_only=False)["hyper_parameters"])
hp["dataset"] = [DATASET]
torch.manual_seed(hp.get("seed", 1234))
VitsModel(**hp)  # consumes the global RNG exactly as model construction did (current code
                 # loads canonical_split.json inside, which draws no random numbers)
full = VitsDataset(DATASET, max_phoneme_ids=hp.get("max_phoneme_ids"))
nv, nt = hp["num_val_examples"], hp["num_test_examples"]
tr, va, te = random_split(full, [len(full) - nv - nt, nv, nt])
ids = lambda s: [str(full.utterances[i].audio_norm_path) for i in s.indices]
split = {"train": ids(tr), "val": ids(va), "test": ids(te)}
text = {str(u.audio_norm_path): u.text for u in full.utterances}

tb = json.load(open(TB))
mrf_tags = [v["distinct_tag_sequences"] for p, v in tb.items() if "mrf_111_run_500ep" in p and v["distinct_tag_sequences"]]
logged = {json.dumps(s[0]["tags"], ensure_ascii=False) for s in mrf_tags}
assert len(logged) == 1, "MRF phases logged different val heads"
logged_head = json.loads(logged.pop())
recon_head = [text[p] for p in split["val"][:5]]
old_test = [r["audio_norm_path"] for r in json.load(open(OLD_TEST))]

checks = {
    "val_head_matches_tensorboard_in_order": recon_head == logged_head,
    "test_matches_2026-09-15_manifest_in_order": split["test"] == old_test,
    "test_matches_2026-09-15_manifest_as_set": set(split["test"]) == set(old_test),
    "partition_sizes": {k: len(v) for k, v in split.items()},
    "disjoint_and_complete": len(set().union(*map(set, split.values()))) == len(full),
}
canon = json.load(open("<WSL_PROJECT_HOME>/BanhmiTTS/training_output/ljspeech/medium/canonical_split.json"))
checks["train_overlap_with_canonical_train"] = len(set(split["train"]) & set(canon["train"]))
checks["mrf_test_in_canonical"] = {k: len(set(split["test"]) & set(canon[k])) for k in canon}
checks["canonical_test_in_mrf"] = {k: len(set(canon["test"]) & set(split[k])) for k in split}
blob = json.dumps(split, sort_keys=True).encode()
out = {"procedure": __doc__, "checkpoint": CKPT, "seed": hp.get("seed", 1234),
       "checks": checks, "split_sha256": hashlib.sha256(blob).hexdigest(), "split": split}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(checks, indent=1))
print("split sha256:", out["split_sha256"])
