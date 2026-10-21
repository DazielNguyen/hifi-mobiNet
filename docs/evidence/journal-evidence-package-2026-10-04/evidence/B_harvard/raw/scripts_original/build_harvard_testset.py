"""Builds the out-of-domain test set: the 720 Harvard sentences (IEEE 1969),
phonemized exactly like banhmi_train.preprocess.worker does (espeak en-us,
banhmi_phonemize -- verified to reproduce the stored canonical test-set ids).
Also checks the sentences against every LJSpeech transcript, so we can state
that no model (BanhmiTTS or Piper) has seen them during training.
"""
import html
import json
import re

from banhmi_phonemize import phonemize_espeak, phoneme_ids_espeak

HTML = "/tmp/harvard.html"
LJ_DATASET = "<WSL_PROJECT_HOME>/BanhmiTTS/training_output/ljspeech/medium/dataset.jsonl"
OUT = "/tmp/harvard_testset_720.json"

raw = open(HTML, encoding="utf-8", errors="replace").read()
texts = [html.unescape(t).strip() for t in re.findall(r"<li>([^<]*)", raw)]
texts = [re.sub(r"\s+", " ", t) for t in texts if t]
assert len(texts) == 720, len(texts)


def norm(t):
    return re.sub(r"[^a-z ]", "", t.lower()).split()


lj_texts = [json.loads(l)["text"] for l in open(LJ_DATASET, encoding="utf-8")]
lj_sentences = {" ".join(norm(t)) for t in lj_texts}
lj_ngrams = set()
N = 6
for t in lj_texts:
    w = norm(t)
    lj_ngrams.update(" ".join(w[i:i + N]) for i in range(len(w) - N + 1))

exact = [t for t in texts if " ".join(norm(t)) in lj_sentences]
ngram_hits = [t for t in texts
              if any(" ".join(norm(t)[i:i + N]) in lj_ngrams for i in range(len(norm(t)) - N + 1))]
print(f"LJSpeech utterances checked: {len(lj_texts)}")
print(f"Exact sentence overlap: {len(exact)}")
print(f"Sentences sharing any {N}-gram with LJSpeech: {len(ngram_hits)}")
for t in ngram_hits:
    print("  ", t)

samples = []
for i, t in enumerate(texts):
    ph = [p for sen in phonemize_espeak(t, "en-us") for p in sen]
    samples.append({"idx": i, "list": i // 10 + 1, "text": t, "phonemes": ph,
                    "phoneme_ids": phoneme_ids_espeak(ph)})

with open(OUT, "w") as f:
    json.dump(samples, f, indent=1)
lens = [len(s["phoneme_ids"]) for s in samples]
print(f"Wrote {len(samples)} samples to {OUT}; phoneme_ids len min/mean/max = "
      f"{min(lens)}/{sum(lens)/len(lens):.1f}/{max(lens)}")
