"""Same T-sweep methodology as bench_dec_vary_T.py, adding the SEQUENTIAL
resblock="mb" variant (2 ResBlockInverted blocks stacked, not 3 parallel
branches) alongside baseline and MRF-parallel -- cheap, untrained
forward-pass-only check of whether sequential's smaller total depthwise
compute meaningfully shifts the memory-bound crossover point before
committing to a full retrain.
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


def make_onnx(kind, gen):
    gen.eval()
    with torch.no_grad():
        gen.remove_weight_norm()
    dummy = torch.randn(1, 192, 32)
    onnx_path = f"/tmp/dec_only_seqcmp_{kind}.onnx"
    torch.onnx.export(
        gen, (dummy,), onnx_path,
        input_names=["z"], output_names=["audio"],
        dynamic_axes={"z": {2: "time"}, "audio": {2: "audio_time"}},
        opset_version=18, dynamo=False,
    )
    n_params = sum(p.numel() for p in gen.parameters())
    return onnx_path, n_params


def bench_onnx(path, T, threads, n_warmup=3, n_runs=10):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
    z = np.random.randn(1, 192, T).astype(np.float32)
    feed = {"z": z}
    for _ in range(n_warmup):
        sess.run(None, feed)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        sess.run(None, feed)
    return (time.perf_counter() - t0) / n_runs * 1000


torch.manual_seed(0)
baseline_gen = Generator(resblock="2", use_snake=False, **COMMON)
baseline_onnx, baseline_params = make_onnx("baseline", baseline_gen)

torch.manual_seed(0)
mrf_gen = Generator(resblock="mrf", use_snake=True, mb_expansion=(1, 1, 1), **COMMON)
mrf_onnx, mrf_params = make_onnx("mrf", mrf_gen)

torch.manual_seed(0)
seq_gen = Generator(resblock="mb", use_snake=True, mb_expansion=(1, 1, 1), mb_kernel_size=3, mb_blocks_per_stage=2, **COMMON)
seq_onnx, seq_params = make_onnx("seq", seq_gen)

print(f"Params: baseline={baseline_params:,}  MRF-parallel={mrf_params:,}  Sequential={seq_params:,}")
print()
print(f"{'T':>6s} {'audio_sec':>10s} {'baseline':>10s} {'mrf_par':>10s} {'sequential':>11s} {'mrf vs base':>12s} {'seq vs base':>12s}")
for T in (32, 64, 128, 256, 512, 700):
    audio_sec = T * 256 / 22050
    b_ms = bench_onnx(baseline_onnx, T, threads=2)
    m_ms = bench_onnx(mrf_onnx, T, threads=2)
    s_ms = bench_onnx(seq_onnx, T, threads=2)
    mrf_vs = f"{(m_ms-b_ms)/b_ms*100:+.1f}%"
    seq_vs = f"{(s_ms-b_ms)/b_ms*100:+.1f}%"
    print(f"{T:6d} {audio_sec:10.2f} {b_ms:10.2f} {m_ms:10.2f} {s_ms:11.2f} {mrf_vs:>12s} {seq_vs:>12s}")
