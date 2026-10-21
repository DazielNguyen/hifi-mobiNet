#!/usr/bin/env python3
"""Cheap WER-only PTQ-sweep eval: no UTMOS (slow to load/run), just Whisper
WER on a small fixed sample set -- matches QAT-Training's own PTQ-sweep
methodology (measure WER on 10-30 sentences per config, ~5-10 min each,
before spending real GPU-hours on QAT training for whichever exclusion
config actually works)."""
import argparse
import json
import random
import re
import sys
import time

import jiwer
import numpy as np
import onnxruntime as ort

_NORM_RE = re.compile(r"[^a-z0-9' ]+")


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = _NORM_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n-samples", type=int, default=25)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    random.seed(args.seed)
    entries = []
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    test_entries = random.sample(entries, args.n_samples)

    print("Loading Whisper (base.en) ...", flush=True)
    import whisper
    whisper_model = whisper.load_model("base.en")

    sess = ort.InferenceSession(args.model_path, providers=["CPUExecutionProvider"])
    wers = []
    t_start = time.perf_counter()

    for i, e in enumerate(test_entries):
        ids = np.array(e["phoneme_ids"], dtype=np.int64)[None, :]
        feed = {
            "input": ids,
            "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
            "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32),
        }
        try:
            audio = sess.run(None, feed)[0]
        except Exception as ex:
            print(f"  [{i+1}] INFERENCE FAILED: {type(ex).__name__}: {str(ex)[:200]}", flush=True)
            wers.append(1.0)
            continue
        audio = np.squeeze(audio).astype(np.float32)
        if audio.ndim == 0 or audio.size < 1000 or not np.isfinite(audio).all():
            print(f"  [{i+1}] DEGENERATE OUTPUT (shape={audio.shape})", flush=True)
            wers.append(1.0)
            continue

        whisper_result = whisper_model.transcribe(audio, language="en", fp16=False)
        hyp = normalize(whisper_result["text"])
        ref = normalize(e["text"])
        utt_wer = jiwer.wer(ref, hyp) if ref else 1.0
        wers.append(utt_wer)

    elapsed = time.perf_counter() - t_start
    result = {"n": len(wers), "wer": float(np.mean(wers)), "elapsed_s": elapsed}
    print(f"RESULT {args.model_path}: n={result['n']} WER={result['wer']:.4f} ({elapsed:.0f}s)", flush=True)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
