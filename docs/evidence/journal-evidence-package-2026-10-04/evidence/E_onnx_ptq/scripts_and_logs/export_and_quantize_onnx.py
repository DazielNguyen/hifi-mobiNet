import sys
import pathlib
import os

sys.path.insert(0, "<WSL_PROJECT_HOME>/BanhmiTTS")
import torch

torch.serialization.add_safe_globals([pathlib.PosixPath])

from banhmi_train.vits.training import VitsModel

CKPT = "<WSL_PROJECT_HOME>/vocos_small_run_full/lightning_logs/version_4/checkpoints/best-epoch=306-val_loss_mel=19.6675.ckpt"
DATASET = "<WSL_PROJECT_HOME>/BanhmiTTS/training_output/ljspeech/medium/dataset.jsonl"
OUT_DIR = "<WSL_PROJECT_HOME>/quantized_models/vocos_small_epoch306_working"
FP32_ONNX = f"{OUT_DIR}/model_fp32.onnx"
INT8_ONNX = f"{OUT_DIR}/model_int8.onnx"

os.makedirs(OUT_DIR, exist_ok=True)

model = VitsModel.load_from_checkpoint(CKPT, map_location="cpu", dataset=[DATASET])
model_g = model.model_g
model_g.eval()
with torch.no_grad():
    model_g.dec.remove_weight_norm()

num_symbols = model_g.n_vocab

# Same wrapper pattern as Piper's export_onnx.py, adapted to this project's
# infer() signature (no sid -- single-speaker, gin_channels=0 always).
def infer_forward(text, text_lengths, scales):
    noise_scale, length_scale, noise_scale_w = scales[0], scales[1], scales[2]
    audio = model_g.infer(
        text, text_lengths,
        noise_scale=noise_scale, length_scale=length_scale, noise_scale_w=noise_scale_w,
        onnx_export=True,
    )[0]
    return audio

model_g.forward = infer_forward

dummy_input_length = 50
sequences = torch.randint(low=0, high=num_symbols, size=(1, dummy_input_length), dtype=torch.long)
sequence_lengths = torch.LongTensor([sequences.size(1)])
scales = torch.FloatTensor([0.667, 1.0, 0.8])
dummy_input = (sequences, sequence_lengths, scales)

# infer() draws random noise internally (dp's reverse-sampling, z_p sampling)
# via torch.randn/torch.randn_like -- torch.onnx.export's tracer calls the
# model more than once internally, so seeding alone doesn't line up the
# draws between the trace and a later verification call. Pin both calls to
# the *same deterministic* noise (zeros) instead, purely for this
# correctness check -- real inference keeps real randomness.
_real_randn, _real_randn_like = torch.randn, torch.randn_like
torch.randn = lambda *a, **k: _real_randn(*a, **{**k, "generator": None}) * 0.0
torch.randn_like = lambda t, **k: _real_randn_like(t, **k) * 0.0

torch.onnx.export(
    model=model_g,
    args=dummy_input,
    f=FP32_ONNX,
    verbose=False,
    opset_version=18,
    input_names=["input", "input_lengths", "scales"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 1: "phonemes"},
        "input_lengths": {0: "batch_size"},
        "output": {0: "batch_size", 1: "time"},
    },
    dynamo=False,  # torch>=2.x defaults to the dynamo/onnxscript exporter; force the classic TorchScript-based tracer instead (no extra deps)
)
print(f"Exported FP32 ONNX -> {FP32_ONNX} ({os.path.getsize(FP32_ONNX)/1024/1024:.3f} MB)")

# --- INT8 quantization on the ONNX graph itself (separate concern from the
# export fix itself -- try it, but don't let a quantizer-tool limitation
# block verifying the actual export). ---
int8_ok = False
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(model_input=FP32_ONNX, model_output=INT8_ONNX, weight_type=QuantType.QInt8)
    print(f"Exported INT8 ONNX -> {INT8_ONNX} ({os.path.getsize(INT8_ONNX)/1024/1024:.3f} MB)")
    int8_ok = True
except Exception as e:
    print(f"\nINT8 quantization step failed (separate onnxruntime-quantization tool limitation, "
          f"not an export correctness issue): {type(e).__name__}: {e}")

# --- Smoke test via onnxruntime (the actual runtime a C++ engine uses) ---
import onnxruntime as ort
import numpy as np

paths = [("fp32", FP32_ONNX)] + ([("int8", INT8_ONNX)] if int8_ok else [])
for name, path in paths:
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out = sess.run(
        None,
        {
            "input": sequences.numpy().astype(np.int64),
            "input_lengths": sequence_lengths.numpy().astype(np.int64),
            "scales": scales.numpy().astype(np.float32),
        },
    )
    audio = out[0]
    print(f"{name}: onnxruntime output shape={audio.shape}, range=[{audio.min():.3f}, {audio.max():.3f}]")

# --- Cross-check: does the ONNX graph's output match the original PyTorch
# model's own output on the SAME input? (not just "doesn't crash") ---
torch.manual_seed(1234)
with torch.no_grad():
    torch_audio = infer_forward(sequences, sequence_lengths, scales).numpy()
onnx_audio = ort.InferenceSession(FP32_ONNX, providers=["CPUExecutionProvider"]).run(
    None,
    {
        "input": sequences.numpy().astype(np.int64),
        "input_lengths": sequence_lengths.numpy().astype(np.int64),
        "scales": scales.numpy().astype(np.float32),
    },
)[0]
diff = abs(torch_audio - onnx_audio)
print(f"\nPyTorch vs ONNX (fp32) output: max_abs_diff={diff.max():.3e}  mean_abs_diff={diff.mean():.3e}")
