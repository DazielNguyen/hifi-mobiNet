"""Scores one model's Harvard wavs with UTMOSv2 + Whisper-small WER, the same
scorers and WER normalization as the canonical n=500 evals.
Usage: score.py <label>
"""
import json
import sys
import time

import numpy as np
import soundfile as sf

LABEL = sys.argv[1]
SYNTH_JSON = f"/tmp/harvard_{LABEL}_synth.json"
OUT_JSON = f"/tmp/harvard_{LABEL}_results.json"
SR = 22050

synth = json.load(open(SYNTH_JSON))
import utmosv2
utmosv2_model = utmosv2.create_model(pretrained=True, device="cpu")
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
t_all = time.perf_counter()
for i, s in enumerate(synth):
    wave, sr_read = sf.read(s["wav_path"], dtype="float32")
    assert sr_read == SR, sr_read
    u2 = float(utmosv2_model.predict(data=wave, sr=SR, device="cpu"))
    hyp = asr.transcribe(s["wav_path"], language="en", fp16=False)["text"]
    wer = jiwer.wer(s["text"], hyp, reference_transform=wer_transform, hypothesis_transform=wer_transform)
    results.append({**{k: s[k] for k in ("idx", "text", "synth_time_s", "audio_duration_s", "rtf")},
                    "asr_text": hyp, "utmosv2": u2, "wer": wer})
    if (i + 1) % 50 == 0 or i == len(synth) - 1:
        el = time.perf_counter() - t_all
        print(f"[{LABEL} score {i+1}/{len(synth)}] {el/60:.1f}min ETA={el/(i+1)*(len(synth)-i-1)/60:.1f}min", flush=True)
        json.dump(results, open(OUT_JSON, "w"), indent=1)

json.dump(results, open(OUT_JSON, "w"), indent=1)
r = {k: np.array([x[k] for x in results]) for k in ("rtf", "utmosv2", "wer")}
print(f"=== {LABEL} Harvard n={len(results)}: RTF={r['rtf'].mean():.4f} "
      f"UTMOSv2={r['utmosv2'].mean():.3f} WER={r['wer'].mean():.4f} ===", flush=True)
