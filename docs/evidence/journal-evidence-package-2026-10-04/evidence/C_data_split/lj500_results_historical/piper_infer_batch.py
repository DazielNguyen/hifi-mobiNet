"""Runs the ORIGINAL Piper checkpoint through Piper's OWN (unmodified)
piper_train.vits.lightning.VitsModel -- guarantees architecturally-faithful
inference (no risk of BanhmiTTS's newer VITS2/SnakeBeta/MRD additions
silently corrupting a plain-VITS1 checkpoint). Reads our canonical n=500
test set (phoneme_ids already in Piper's own id scheme -- banhmi_phonemize
mirrors piper_phonemize's table), writes one wav per sample + a results
JSON with per-sample RTF, matching the fields our own WER/UTMOSv2 scoring
step expects to merge in afterward.
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

CKPT = "<WSL_PROJECT_HOME>/piper_original/lj-med_1000.ckpt"
TEST_SET = "/tmp/eval_true_testset_500_baseline.json"
OUT_DIR = pathlib.Path("<WSL_PROJECT_HOME>/quantized_models/eval_piper_original_500")
OUT_JSON = "/tmp/eval_piper_original_500_synth.json"
SR = 22050

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading original Piper checkpoint via piper_train's own VitsModel ...", flush=True)
model = VitsModel.load_from_checkpoint(CKPT, map_location="cpu", dataset=None)
model.eval()
with torch.no_grad():
    model.model_g.dec.remove_weight_norm()
print("Loaded OK.", flush=True)

with open(TEST_SET) as f:
    samples = json.load(f)
print(f"Loaded {len(samples)} test samples", flush=True)

results = []
t_start_all = time.perf_counter()
for i, s in enumerate(samples):
    ids = s["phoneme_ids"]
    text = torch.LongTensor(ids).unsqueeze(0)
    text_lengths = torch.LongTensor([len(ids)])
    scales = [0.667, 1.0, 0.8]  # noise_scale, length_scale, noise_w -- same convention as our own evals

    torch.manual_seed(1234)
    t0 = time.perf_counter()
    with torch.no_grad():
        audio = model(text, text_lengths, scales, sid=None).detach().numpy()
    synth_time = time.perf_counter() - t0

    wave = audio.squeeze().astype("float32")
    dur = len(wave) / SR
    rtf = synth_time / dur if dur > 0 else float("nan")

    wav_path = OUT_DIR / f"sample_{i:04d}.wav"
    write_wav(str(wav_path), SR, audio_float_to_int16(wave))

    results.append({
        "idx": i, "text": s["text"],
        "wav_path": str(wav_path),
        "synth_time_s": synth_time, "audio_duration_s": dur, "rtf": rtf,
    })

    if (i + 1) % 25 == 0 or i == len(samples) - 1:
        elapsed = time.perf_counter() - t_start_all
        eta = elapsed / (i + 1) * (len(samples) - i - 1)
        print(f"[piper_original {i+1}/{len(samples)}] rtf={rtf:.3f} elapsed={elapsed/60:.1f}min ETA={eta/60:.1f}min", flush=True)
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=2)

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nDone. wrote {OUT_JSON}", flush=True)
