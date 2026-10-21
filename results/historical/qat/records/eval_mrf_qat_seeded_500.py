import json
import pathlib
import sys
import time

sys.path.insert(0, "<WSL_PROJECT_HOME>/BanhmiTTS")
import numpy as np
import torch

torch.serialization.add_safe_globals([pathlib.PosixPath])

from banhmi_train.vits.training import VitsModel
from banhmi_train.vits.quantize import (
    prepare_qat,
    remove_all_weight_norm,
    set_fake_quant_enabled,
    set_observer_enabled,
)
from banhmi_train.audio_utils import audio_float_to_int16
from banhmi_train import wavfile

CKPT = "<WSL_PROJECT_HOME>/mrf_111_qat_flow_enc_p_dp/lightning_logs/version_0/checkpoints/qat-best-epoch=97-val_loss_mel=19.2989.ckpt"
WAV_DIR = "<WSL_PROJECT_HOME>/quantized_models/eval_mrf_qat_seeded_500"
OUT_JSON = "/tmp/eval_mrf_qat_seeded_500_results.json"
TEST_SET = "/tmp/eval_true_testset_500_mrf.json"
LABEL = "mrf_111_qat_flow_enc_p_dp SEEDED (epoch=97, seed=1234)"

import os
os.makedirs(WAV_DIR, exist_ok=True)

with open(TEST_SET) as f:
    samples = json.load(f)
print(f"Loaded {len(samples)} TRUE held-out test samples for {LABEL}", flush=True)

print(f"Loading QAT model (CPU) from {CKPT} ...", flush=True)
model = VitsModel.load_from_checkpoint(
    CKPT, dataset=None, map_location="cpu", weights_only=False,
    strict=False, use_f0=False,
)
model_g = model.model_g
remove_all_weight_norm(model_g)
prepare_qat(
    model_g, submodule_names=("flow", "enc_p", "dp"), exclude_leaf_names=(),
    wrap_types=(torch.nn.Conv1d, torch.nn.ConvTranspose1d),
)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
state_dict = {k[len("model_g."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model_g.")}
missing, unexpected = model_g.load_state_dict(state_dict, strict=False)
assert not missing, f"missing keys: {missing}"

model_g.eval()
set_observer_enabled(model_g, False)
set_fake_quant_enabled(model_g, True)
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

torch.manual_seed(1234)  # fixed seed for noise_scale/noise_scale_w stochastic sampling -- was previously unseeded, confounding cross-eval comparisons
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
        print(f"[MRF_QAT_FINAL {i+1}/{len(samples)}] rtf={rtf:.3f} utmosv2={utmosv2_score:.2f} wer={wer:.3f} "
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
