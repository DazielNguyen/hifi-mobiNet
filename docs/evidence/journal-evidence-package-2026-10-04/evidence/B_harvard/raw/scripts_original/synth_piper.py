"""Synthesizes the Harvard test set with the original Piper checkpoint through
Piper's own unmodified piper_train code (run with piper_env's python), same
settings and per-utterance seeding as synth_banhmi.py.
"""
import json
import pathlib
import sys
import time

sys.path.insert(0, "<WORKSPACE>/piper/src/python")
import torch

torch.serialization.add_safe_globals([pathlib.PosixPath])

from piper_train.vits.lightning import VitsModel
from piper_train.vits.utils import audio_float_to_int16
from piper_train.vits.wavfile import write as write_wav

LABEL = "piper"
CKPT = "<WSL_PROJECT_HOME>/piper_original/lj-med_1000.ckpt"
TEST_SET = "/tmp/harvard_testset_720.json"
OUT_DIR = pathlib.Path(f"<WSL_PROJECT_HOME>/quantized_models/harvard_{LABEL}")
OUT_JSON = f"/tmp/harvard_{LABEL}_synth.json"
SR = 22050
OUT_DIR.mkdir(parents=True, exist_ok=True)

model = VitsModel.load_from_checkpoint(CKPT, map_location="cpu", dataset=None)
model.eval()
with torch.no_grad():
    model.model_g.dec.remove_weight_norm()
samples = json.load(open(TEST_SET))
print(f"[{LABEL}] {len(samples)} samples", flush=True)

results = []
t_all = time.perf_counter()
for i, s in enumerate(samples):
    text = torch.LongTensor(s["phoneme_ids"]).unsqueeze(0)
    text_lengths = torch.LongTensor([text.shape[1]])
    torch.manual_seed(1234)
    t0 = time.perf_counter()
    with torch.no_grad():
        audio = model(text, text_lengths, [0.667, 1.0, 0.8], sid=None).detach().numpy()
    synth_time = time.perf_counter() - t0
    wave = audio.squeeze().astype("float32")
    dur = len(wave) / SR
    wav_path = OUT_DIR / f"sample_{i:04d}.wav"
    write_wav(str(wav_path), SR, audio_float_to_int16(wave))
    results.append({"idx": i, "text": s["text"], "wav_path": str(wav_path),
                    "synth_time_s": synth_time, "audio_duration_s": dur,
                    "rtf": synth_time / dur if dur > 0 else float("nan")})
    if (i + 1) % 100 == 0:
        print(f"[{LABEL} synth {i+1}/{len(samples)}] {(time.perf_counter()-t_all)/60:.1f}min", flush=True)

json.dump(results, open(OUT_JSON, "w"), indent=1)
print(f"[{LABEL}] synth done -> {OUT_JSON}", flush=True)
