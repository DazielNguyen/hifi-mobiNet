"""Rigorous RTF-only benchmark for baseline vs SEQ(1,1,1), FP32 vs PTQ INT8.
No WER/UTMOSv2 (SEQ isn't converged yet -- RTF only depends on the graph,
not weight quality). Fixed intra_op_num_threads (matches established VNNI
methodology), n=500 real held-out test sentences (same canonical_split.json
test set for every config, enabling paired stats), warmup before timing.
"""
import json
import sys
import time

import numpy as np
import onnxruntime as ort

MODEL_PATH = sys.argv[1]
LABEL = sys.argv[2]
OUT_JSON = sys.argv[3]
NUM_THREADS = int(sys.argv[4]) if len(sys.argv) > 4 else 2
N_WARMUP = 5

TEST_SET = "/tmp/eval_true_testset_500_baseline.json"

with open(TEST_SET) as f:
    samples = json.load(f)
print(f"[{LABEL}] loaded {len(samples)} test samples", flush=True)

opts = ort.SessionOptions()
opts.intra_op_num_threads = NUM_THREADS
opts.inter_op_num_threads = 1
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
sess = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])
print(f"[{LABEL}] session loaded, intra_op_num_threads={NUM_THREADS}", flush=True)

SR = 22050

# Warmup (exclude JIT/first-call overhead from timed measurements)
warm_ids = np.array(samples[0]["phoneme_ids"], dtype=np.int64)[None, :]
warm_feed = {
    "input": warm_ids,
    "input_lengths": np.array([warm_ids.shape[1]], dtype=np.int64),
    "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32),
}
for _ in range(N_WARMUP):
    sess.run(None, warm_feed)
print(f"[{LABEL}] warmup done ({N_WARMUP} runs)", flush=True)

results = []
t_start_all = time.perf_counter()
for i, s in enumerate(samples):
    ids = np.array(s["phoneme_ids"], dtype=np.int64)[None, :]
    feed = {
        "input": ids,
        "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
        "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32),
    }
    t0 = time.perf_counter()
    audio = sess.run(None, feed)[0]
    synth_time = time.perf_counter() - t0

    wave = np.squeeze(audio).astype(np.float32)
    dur = len(wave) / SR
    rtf = synth_time / dur if dur > 0 else float("nan")

    results.append({
        "idx": i,
        "utt_id": s.get("utt_id", s.get("audio_norm_path", str(i))),
        "n_phonemes": len(s["phoneme_ids"]),
        "synth_time_s": synth_time,
        "audio_dur_s": dur,
        "rtf": rtf,
    })
    if (i + 1) % 100 == 0:
        elapsed = time.perf_counter() - t_start_all
        print(f"[{LABEL}] {i+1}/{len(samples)} done, elapsed={elapsed:.1f}s", flush=True)

total_elapsed = time.perf_counter() - t_start_all
rtfs = np.array([r["rtf"] for r in results])
print(f"[{LABEL}] DONE. n={len(results)} total_wall_time={total_elapsed:.1f}s", flush=True)
print(f"[{LABEL}] RTF: mean={rtfs.mean():.5f} median={np.median(rtfs):.5f} std={rtfs.std():.5f} "
      f"min={rtfs.min():.5f} max={rtfs.max():.5f} p95={np.percentile(rtfs,95):.5f}", flush=True)

with open(OUT_JSON, "w") as f:
    json.dump({
        "label": LABEL,
        "model_path": MODEL_PATH,
        "num_threads": NUM_THREADS,
        "n": len(results),
        "total_wall_time_s": total_elapsed,
        "rtf_mean": float(rtfs.mean()),
        "rtf_median": float(np.median(rtfs)),
        "rtf_std": float(rtfs.std()),
        "rtf_min": float(rtfs.min()),
        "rtf_max": float(rtfs.max()),
        "rtf_p95": float(np.percentile(rtfs, 95)),
        "per_sample": results,
    }, f, indent=2)
print(f"[{LABEL}] wrote {OUT_JSON}", flush=True)
