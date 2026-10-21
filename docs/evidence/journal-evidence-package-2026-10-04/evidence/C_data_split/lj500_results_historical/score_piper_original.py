import json
import time

import numpy as np
import soundfile as sf

SYNTH_JSON = "/tmp/eval_piper_original_500_synth.json"
OUT_JSON = "/tmp/eval_piper_original_500_full_results.json"
SR = 22050

with open(SYNTH_JSON) as f:
    synth = json.load(f)
print(f"Loaded {len(synth)} synthesized samples", flush=True)

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
for i, s in enumerate(synth):
    wav_path = s["wav_path"]
    wave, sr_read = sf.read(wav_path, dtype="float32")
    assert sr_read == SR, f"sample rate mismatch: {sr_read} != {SR}"

    utmosv2_score = float(utmosv2_model.predict(data=wave, sr=SR, device="cpu"))

    asr_out = asr.transcribe(wav_path, language="en", fp16=False)
    hyp = asr_out["text"]
    ref = s["text"]
    wer = jiwer.wer(ref, hyp, reference_transform=wer_transform, hypothesis_transform=wer_transform)

    results.append({
        "idx": s["idx"], "text": ref, "asr_text": hyp,
        "synth_time_s": s["synth_time_s"], "audio_duration_s": s["audio_duration_s"], "rtf": s["rtf"],
        "utmosv2": utmosv2_score, "wer": wer,
    })

    if (i + 1) % 10 == 0 or i == len(synth) - 1:
        elapsed = time.perf_counter() - t_start_all
        eta = elapsed / (i + 1) * (len(synth) - i - 1)
        print(f"[piper_original_score {i+1}/{len(synth)}] utmosv2={utmosv2_score:.2f} wer={wer:.3f} "
              f"| elapsed={elapsed/60:.1f}min ETA={eta/60:.1f}min", flush=True)
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=2)

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

rtfs = np.array([r["rtf"] for r in results])
u2 = np.array([r["utmosv2"] for r in results])
wers = np.array([r["wer"] for r in results])

print(f"\n=== FINAL SUMMARY: Piper original (lj-med_1000, epoch=999) (n={len(results)}) ===")
print(f"RTF:      mean={rtfs.mean():.4f}  std={rtfs.std():.4f}  median={np.median(rtfs):.4f}")
print(f"UTMOSv2:  mean={u2.mean():.3f}  std={u2.std():.3f}  median={np.median(u2):.3f}")
print(f"WER:      mean={wers.mean():.4f}  std={wers.std():.4f}  median={np.median(wers):.4f}")
total_time = time.perf_counter() - t_start_all
print(f"Total wall time: {total_time/60:.1f} min")
