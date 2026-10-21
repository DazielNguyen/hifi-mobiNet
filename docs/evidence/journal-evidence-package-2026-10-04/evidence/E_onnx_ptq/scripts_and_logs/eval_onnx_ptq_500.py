import json
import sys
import time

import numpy as np
import onnxruntime as ort

MODEL_PATH = sys.argv[1]
TEST_SET = sys.argv[2]
LABEL = sys.argv[3]
OUT_JSON = sys.argv[4]
WAV_DIR = sys.argv[5]
NUM_THREADS = int(sys.argv[6]) if len(sys.argv) > 6 else 2

import os
os.makedirs(WAV_DIR, exist_ok=True)

sys.path.insert(0, "<WSL_PROJECT_HOME>/BanhmiTTS")
from banhmi_train.audio_utils import audio_float_to_int16
from banhmi_train import wavfile

SR = 22050

with open(TEST_SET) as f:
    samples = json.load(f)
print(f"Loaded {len(samples)} test samples for {LABEL}", flush=True)

opts = ort.SessionOptions()
opts.intra_op_num_threads = NUM_THREADS
opts.inter_op_num_threads = 1
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
sess = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])
print(f"Loaded ONNX model: {MODEL_PATH} (intra_op_num_threads={NUM_THREADS})", flush=True)

print("Loading UTMOSv2 predictor ...", flush=True)
import utmosv2
utmosv2_model = utmosv2.create_model(pretrained=True, device="cpu")

print("Loading Whisper (small, CPU) ...", flush=True)
import whisper
asr = whisper.load_model("small", device="cpu")

import jiwer
wer_transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

results = []
t_start_all = time.perf_counter()
for i, s in enumerate(samples):
    phoneme_ids = s["phoneme_ids"]
    ids = np.array(phoneme_ids, dtype=np.int64)[None, :]
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

    wav_path = f"{WAV_DIR}/sample_{i:04d}.wav"
    wavfile.write(wav_path, SR, audio_float_to_int16(wave))

    utmosv2_score = float(utmosv2_model.predict(data=wave, sr=SR, device="cpu"))

    asr_out = asr.transcribe(wav_path, language="en", fp16=False)
    hyp = asr_out["text"]
    ref = s["text"]
    wer = jiwer.wer(ref, hyp, reference_transform=wer_transform, hypothesis_transform=wer_transform)

    results.append({
        "idx": i, "text": ref, "asr_text": hyp,
        "synth_time_s": synth_time, "audio_duration_s": dur, "rtf": rtf,
        "utmosv2": utmosv2_score, "wer": wer,
    })

    if (i + 1) % 10 == 0 or i == len(samples) - 1:
        elapsed = time.perf_counter() - t_start_all
        eta = elapsed / (i + 1) * (len(samples) - i - 1)
        print(f"[{i+1}/{len(samples)}] rtf={rtf:.3f} utmosv2={utmosv2_score:.2f} wer={wer:.3f} "
              f"| elapsed={elapsed/60:.1f}min ETA={eta/60:.1f}min", flush=True)
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=2)

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

rtfs = np.array([r["rtf"] for r in results])
u2 = np.array([r["utmosv2"] for r in results])
wers = np.array([r["wer"] for r in results])

print(f"\n=== FINAL SUMMARY: {LABEL} (n={len(results)}) ===")
print(f"RTF:      mean={rtfs.mean():.4f}  std={rtfs.std():.4f}  median={np.median(rtfs):.4f}")
print(f"UTMOSv2:  mean={u2.mean():.3f}  std={u2.std():.3f}  median={np.median(u2):.3f}")
print(f"WER:      mean={wers.mean():.4f}  std={wers.std():.4f}  median={np.median(wers):.4f}")
total_time = time.perf_counter() - t_start_all
print(f"Total wall time: {total_time/60:.1f} min")
