import json
import pathlib
import sys

sys.path.insert(0, "<WSL_PROJECT_HOME>/BanhmiTTS")
import torch

torch.serialization.add_safe_globals([pathlib.PosixPath])

from banhmi_train.vits.training import VitsModel

DATASET = "<WSL_PROJECT_HOME>/BanhmiTTS/training_output/ljspeech/medium/dataset.jsonl"
MRF_CKPT = "<WSL_PROJECT_HOME>/mrf_111_run_500ep/checkpoints/best-epoch=836-val_loss_mel=19.5387.ckpt"
OUT = "/tmp/eval_true_testset_500_mrf.json"

# Build a lookup from the raw dataset.jsonl (has audio_path/phonemes fields
# that the in-memory Utterance dataclass drops).
raw_by_norm_path = {}
with open(DATASET) as f:
    for line in f:
        d = json.loads(line)
        raw_by_norm_path[d["audio_norm_path"]] = d

ckpt = torch.load(MRF_CKPT, map_location="cpu", weights_only=False)
hparams = dict(ckpt["hyper_parameters"])
hparams["dataset"] = [DATASET]
hparams.setdefault("use_f0", False)
torch.manual_seed(1234)
model = VitsModel(**hparams)

test_ds = model._test_dataset
full = test_ds.dataset
out_records = []
missing = 0
for idx in test_ds.indices:
    utt = full.utterances[idx]
    key = str(utt.audio_norm_path)
    raw = raw_by_norm_path.get(key)
    if raw is None:
        missing += 1
        continue
    out_records.append(raw)

print(f"Built {len(out_records)} records (missing lookups: {missing})")
with open(OUT, "w") as f:
    json.dump(out_records, f, indent=2)
print(f"Wrote {OUT}")

# Sanity: confirm zero overlap with MRF's own train set.
train_ids = {
    str(model._train_dataset.dataset.utterances[i].audio_norm_path)
    for i in model._train_dataset.indices
}
new_test_ids = {r["audio_norm_path"] for r in out_records}
leak = new_test_ids & train_ids
print(f"Sanity check -- leakage into MRF's own train set: {len(leak)} / {len(new_test_ids)} (must be 0)")
