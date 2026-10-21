"""Synthesizes the Harvard test set with one BanhmiTTS checkpoint (PyTorch,
CPU, FP32) -- same inference settings as the canonical n=500 evals
(noise 0.667, length 1.0, noise_w 0.8, weight norm removed). Seeds before
every utterance (like the Piper synth script) so each sample's noise is
independent of order. Usage: synth_banhmi.py <label> <ckpt>
"""
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, "<WSL_PROJECT_HOME>/BanhmiTTS")
import torch

torch.serialization.add_safe_globals([pathlib.PosixPath])

from banhmi_train.vits.training import VitsModel
from banhmi_train.audio_utils import audio_float_to_int16
from banhmi_train import wavfile

LABEL, CKPT = sys.argv[1], sys.argv[2]
DATASET = "<WSL_PROJECT_HOME>/BanhmiTTS/training_output/ljspeech/medium/dataset.jsonl"
TEST_SET = "/tmp/harvard_testset_720.json"
WAV_DIR = f"<WSL_PROJECT_HOME>/quantized_models/harvard_{LABEL}"
OUT_JSON = f"/tmp/harvard_{LABEL}_synth.json"
os.makedirs(WAV_DIR, exist_ok=True)

samples = json.load(open(TEST_SET))
print(f"[{LABEL}] {len(samples)} samples, ckpt={CKPT}", flush=True)
model = VitsModel.load_from_checkpoint(CKPT, map_location="cpu", dataset=[DATASET], use_f0=False)
model.eval()
model.freeze()
model_g = model.model_g
with torch.no_grad():
    model_g.dec.remove_weight_norm()
SR = model.hparams.sample_rate
print(f"[{LABEL}] resblock={model.hparams.resblock} sr={SR}", flush=True)

results = []
t_all = time.perf_counter()
for i, s in enumerate(samples):
    x = torch.LongTensor(s["phoneme_ids"]).unsqueeze(0)
    x_lengths = torch.LongTensor([x.shape[1]])
    torch.manual_seed(1234)
    t0 = time.perf_counter()
    with torch.no_grad():
        audio, *_ = model_g.infer(x, x_lengths, noise_scale=0.667, length_scale=1.0, noise_scale_w=0.8)
    synth_time = time.perf_counter() - t0
    wave = audio.squeeze().numpy()
    dur = len(wave) / SR
    wav_path = f"{WAV_DIR}/sample_{i:04d}.wav"
    wavfile.write(wav_path, SR, audio_float_to_int16(wave))
    results.append({"idx": i, "text": s["text"], "wav_path": wav_path,
                    "synth_time_s": synth_time, "audio_duration_s": dur,
                    "rtf": synth_time / dur if dur > 0 else float("nan")})
    if (i + 1) % 100 == 0:
        print(f"[{LABEL} synth {i+1}/{len(samples)}] {(time.perf_counter()-t_all)/60:.1f}min", flush=True)

json.dump(results, open(OUT_JSON, "w"), indent=1)
print(f"[{LABEL}] synth done -> {OUT_JSON}", flush=True)
