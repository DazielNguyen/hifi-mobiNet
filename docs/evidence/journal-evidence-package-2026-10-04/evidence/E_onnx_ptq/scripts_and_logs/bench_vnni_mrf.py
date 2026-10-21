"""FP32 vs real INT8 (QOperator PTQ) speed benchmark under a 1-2 thread CPU
budget, on this actual machine (i7-14700, has AVX-VNNI) -- adapted from
QAT-Training/qat_transfer/bench_constrained.py, same methodology (fixed
intra_op_num_threads, warmup + multiple timed runs)."""
import json
import time

import numpy as np
import onnxruntime as ort


def make_session(path, num_threads):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])


def bench(path, ids, num_threads, n_warmup=3, n_runs=15):
    sess = make_session(path, num_threads)
    feed = {
        "input": ids,
        "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
        "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32),
    }
    for _ in range(n_warmup):
        sess.run(None, feed)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = sess.run(None, feed)
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    audio_sec = out[0].shape[-1] / 22050
    return times.mean(), times.std(), audio_sec


def main():
    entries = []
    with open("<WSL_PROJECT_HOME>/BanhmiTTS/training_output/ljspeech/medium/dataset.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    # A handful of sentences spanning a range of lengths
    test_entries = [entries[i] for i in (10, 500, 2000, 5000, 9000)]

    models = [
        ("FP32", "<WSL_PROJECT_HOME>/quantized_models/ptq_sweep/mrf_fp32.onnx"),
        ("INT8 flow+enc_p+dp (dec=FP32)", "<WSL_PROJECT_HOME>/quantized_models/ptq_sweep/mrf_flow_enc_p_dp.onnx"),
    ]

    for threads in (2, 1):
        print(f"\n=== intra_op_num_threads={threads} ===")
        all_results = {name: [] for name, _ in models}
        for e in test_entries:
            ids = np.array(e["phoneme_ids"], dtype=np.int64)[None, :]
            print(f"  Sentence ({ids.shape[1]} phonemes): {e['text'][:50]}")
            for name, path in models:
                mean_t, std_t, audio_sec = bench(path, ids, threads)
                rtf = mean_t / audio_sec
                all_results[name].append(mean_t)
                print(f"    {name:38s}: {mean_t*1000:8.1f}ms (+/-{std_t*1000:.1f})  RTF={rtf:.4f}")
        fp32_mean = np.mean(all_results["FP32"])
        print("  --- average speedup relative to FP32 ---")
        for name, times in all_results.items():
            m = np.mean(times)
            print(f"    {name:38s}: {fp32_mean/m:.2f}x")


if __name__ == "__main__":
    main()
