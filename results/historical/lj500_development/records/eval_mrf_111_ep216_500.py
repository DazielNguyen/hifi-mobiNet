import json
import pathlib
import sys
import time

sys.path.insert(0, "<WSL_PROJECT_HOME>/BanhmiTTS")
import numpy as np
import torch

torch.serialization.add_safe_globals([pathlib.PosixPath])

from banhmi_train.vits.training import VitsModel
from banhmi_train.audio_utils import audio_float_to_int16
from banhmi_train import wavfile

CKPT = "<WSL_PROJECT_HOME>/mrf_111_run_500ep/checkpoints/best-epoch=216-val_loss_mel=21.0430.ckpt"
DATASET = "<WSL_PROJECT_HOME>/BanhmiTTS/training_output/ljspeech/medium/dataset.jsonl"
WAV_DIR = "<WSL_PROJECT_HOME>/quantized_models/eval_mrf_111_ep216_500"
OUT_JSON = "/tmp/eval_mrf_111_ep216_500_results.json"
TEST_SET = "/tmp/eval_true_testset_500_baseline.json"
LABEL = "mrf_111_run_500ep (resblock=mrf, expansion=(1,1,1), epoch=216, val_loss_mel=21.0430)"

import os
os.makedirs(WAV_DIR, exist_ok=True)

with open(TEST_SET) as f:
    samples = json.load(f)
print(f"Loaded {len(samples)} TRUE held-out test samples for {LABEL}", flush=True)

print(f"Loading model (CPU) from {CKPT} ...", flush=True)
model = VitsModel.load_from_checkpoint(CKPT, map_location="cpu", dataset=[DATASET])
model.eval()
model.freeze()
model_g = model.model_g
with torch.no_grad():
    model_g.dec.remove_weight_norm()
SR = model.hparams.sample_rate

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
    x = torch.LongTensor(phoneme_ids).unsqueeze(0)
    x_lengths = torch.LongTensor([len(phoneme_ids)])

    t0 = time.perf_counter()
    with torch.no_grad():
        audio, *_ = model_g.infer(x, x_lengths, noise_scale=0.667, length_scale=1.0, noise_scale_w=0.8)
    synth_time = time.perf_counter() - t0

    wave = audio.squeeze().numpy()
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
        print(f"[MRF_111 {i+1}/{len(samples)}] rtf={rtf:.3f} utmosv2={utmosv2_score:.2f} wer={wer:.3f} "
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
