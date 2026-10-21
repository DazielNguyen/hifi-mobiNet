"""Export ONLY the Generator (dec) submodule to ONNX for baseline and MRF,
in isolation (no text encoder/flow/dp) -- direct, controlled A/B test of
ONNX Runtime's Conv/depthwise-conv performance for the exact same module
already benchmarked via PyTorch earlier this session. Rules out any
confound from the full-pipeline profiler's name-prefix bucketing.
"""
import sys
import time

sys.path.insert(0, "<WSL_PROJECT_HOME>/BanhmiTTS")
import numpy as np
import torch
import onnxruntime as ort

from banhmi_train.vits.modules.generator import Generator

COMMON = dict(
    initial_channel=192,
    resblock_kernel_sizes=(3, 5, 7),
    resblock_dilation_sizes=((1, 2), (2, 6), (3, 12)),
    upsample_rates=(8, 8, 4),
    upsample_initial_channel=256,
    upsample_kernel_sizes=(16, 16, 8),
)

T = 32  # latent frames (matches earlier param_breakdown.py / test-harness convention)


def export_and_bench(kind, gen, label):
    gen.eval()
    with torch.no_grad():
        gen.remove_weight_norm()
    dummy = torch.randn(1, 192, T)

    onnx_path = f"/tmp/dec_only_{kind}.onnx"
    torch.onnx.export(
        gen, (dummy,), onnx_path,
        input_names=["z"], output_names=["audio"],
        dynamic_axes={"z": {2: "time"}, "audio": {2: "audio_time"}},
        opset_version=18, dynamo=False,
    )

    # PyTorch benchmark (native), 1 thread -- matches this session's own
    # established test-harness methodology exactly.
    torch.set_num_threads(1)
    with torch.no_grad():
        for _ in range(3):
            gen(dummy)
        t0 = time.perf_counter()
        for _ in range(15):
            gen(dummy)
        pt_ms = (time.perf_counter() - t0) / 15 * 1000

    # ONNX Runtime benchmark, 1 and 2 threads.
    results = {"pytorch_1thread_ms": pt_ms}
    for threads in (1, 2):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
        feed = {"z": dummy.numpy()}
        for _ in range(3):
            sess.run(None, feed)
        t0 = time.perf_counter()
        for _ in range(15):
            sess.run(None, feed)
        ort_ms = (time.perf_counter() - t0) / 15 * 1000
        results[f"onnxruntime_{threads}thread_ms"] = ort_ms

    n_params = sum(p.numel() for p in gen.parameters())
    print(f"=== {label} (params={n_params:,}) ===")
    print(f"  PyTorch (1 thread):        {results['pytorch_1thread_ms']:.2f} ms")
    print(f"  ONNX Runtime (1 thread):   {results['onnxruntime_1thread_ms']:.2f} ms")
    print(f"  ONNX Runtime (2 threads):  {results['onnxruntime_2thread_ms']:.2f} ms")
    return results


torch.manual_seed(0)
baseline_gen = Generator(resblock="2", use_snake=False, **COMMON)
r_baseline = export_and_bench("baseline", baseline_gen, "Baseline dec (resblock=2, dense conv)")
print()

torch.manual_seed(0)
mrf_gen = Generator(resblock="mrf", use_snake=True, mb_expansion=(1, 1, 1), **COMMON)
r_mrf = export_and_bench("mrf", mrf_gen, "MRF dec (resblock=mrf, depthwise-separable)")

print()
print("=== So sanh truc tiep ===")
for key in ("pytorch_1thread_ms", "onnxruntime_1thread_ms", "onnxruntime_2thread_ms"):
    b, m = r_baseline[key], r_mrf[key]
    faster = "MRF nhanh hon" if m < b else "baseline nhanh hon"
    print(f"  {key:28s}: baseline={b:7.2f}ms  MRF={m:7.2f}ms  -> {faster} ({abs(m-b)/b*100:.1f}%)")
