"""Harvard-720 synthesis through a matched-noise ONNX graph (protocol section 4).

Per sentence, in manifest order:
1. ``torch.manual_seed(1234)`` and the PyTorch forward exactly as ``harvard/synthesize.py``,
   capturing its duration noise and acoustic noise. The PyTorch WAV bytes are compared with
   the stored PyTorch evaluation (``--reference-synth``): a determinism check.
2. The graph runs with the same duration noise and an acoustic buffer that starts with the same
   acoustic noise (``common.acoustic_buffer``).
3. The ONNX WAV is written as ``harvard/synthesize.py`` writes WAVs, and parity with the
   PyTorch waveform is recorded (equal length, NRMSE, max abs error).

The output is a synthesis record in the ``harvard/synthesize.py`` schema, so
``harvard/score.py`` scores it unchanged. Its timing fields come from this mixed loop and must
not be used; speed is measured by ``time_graphs.py``.

    python scripts/evaluation/onnx/synthesize.py --precision int8 \
        --config configs/training/piper-seqir-no-vits2-cpn.yaml --checkpoint <selected .ckpt> \
        --graph training_output/onnx-ptq/<label>/<label>_int8_matched_noise.onnx \
        --reference-synth training_output/piper-seqir-no-vits2-cpn/eval/harvard720/synth.json \
        --wav-dir training_output/onnx-ptq/<label>/harvard720/int8/wavs \
        --output training_output/onnx-ptq/<label>/harvard720/int8/synth.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--precision", required=True, choices=["fp32", "int8"])
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--graph", required=True, type=Path)
    p.add_argument("--reference-synth", required=True, type=Path)
    p.add_argument("--manifest", default=Path("results/manifests/harvard720.json"), type=Path)
    p.add_argument("--wav-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--limit", type=int, help="first N sentences only (smoke tests; not for evaluation)")
    a = p.parse_args(argv)
    if a.output.exists() or (a.wav_dir.exists() and any(a.wav_dir.iterdir())):
        p.error("output or WAV directory already exists; choose new paths")

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    module, info = C.load_module(a.config, a.checkpoint)
    graph_props = {x.key: x.value for x in onnx.load(a.graph, load_external_data=False).metadata_props}
    if graph_props.get("checkpoint_sha256") != info["checkpoint_sha256"]:
        raise SystemExit("graph metadata names a different checkpoint")
    reference = json.loads(a.reference_synth.read_text(encoding="utf-8"))
    if reference["checkpoint_sha256"] != info["checkpoint_sha256"]:
        raise SystemExit("reference synthesis is of a different checkpoint")
    # PyTorch reference with the stored evaluation's thread count, so that the WAV comparison is a
    # determinism check and not a thread-count effect
    torch_threads = int(reference["protocol"]["torch_threads"])
    torch.set_num_threads(torch_threads)
    samples = json.loads(a.manifest.read_text(encoding="utf-8"))
    ref_rows = reference["rows"]
    if [r["text"] for r in ref_rows] != [s["text"] for s in samples]:
        raise SystemExit("reference synthesis rows do not match the manifest")
    if a.limit:
        samples = samples[:a.limit]
    sess = C.ort_session(a.graph)
    duration_input = next(i.name for i in sess.get_inputs() if i.name not in ("input", "input_lengths", "scales", C.NOISE_INPUT))
    sr = info["sample_rate"]
    a.wav_dir.mkdir(parents=True, exist_ok=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    scales = np.array(C.SCALES, dtype=np.float32)

    rows = []
    t_all = time.perf_counter()
    for i, s in enumerate(samples):
        x = torch.LongTensor(s["phoneme_ids"]).unsqueeze(0)
        x_lengths = torch.LongTensor([x.shape[1]])
        torch.manual_seed(C.SEED)
        with torch.no_grad(), C.capture_noise() as noise:
            audio = module(x, x_lengths, list(C.SCALES))
        if [k for k, _ in noise] != ["randn", "randn_like"]:
            raise SystemExit(f"sentence {i}: unexpected noise draws {[k for k, _ in noise]}")
        ref_wave = audio.squeeze().numpy().astype("float32")
        pytorch_wav_sha = hashlib.sha256(C.wav_bytes(ref_wave, sr)).hexdigest()
        buffer = C.acoustic_buffer(noise[1][1], i)
        feeds = {"input": x.numpy(), "input_lengths": x_lengths.numpy(), "scales": scales,
                 duration_input: noise[0][1], C.NOISE_INPUT: buffer}
        t0 = time.perf_counter()
        out = sess.run(None, feeds)[0]
        synth_time = time.perf_counter() - t0
        wave = out.squeeze().astype("float32")
        wav_path = a.wav_dir / f"sample_{i:04d}.wav"
        wav_path.write_bytes(C.wav_bytes(wave, sr))
        par = C.parity(audio.numpy(), out)
        duration = len(wave) / sr
        rows.append({"idx": i, "text": s["text"], "wav": wav_path.name, "wav_sha256": C.sha256(wav_path),
                     "synth_time_s": synth_time, "audio_duration_s": duration,
                     "rtf": synth_time / duration if duration > 0 else float("nan"),
                     "finite": bool(np.isfinite(wave).all()),
                     "pytorch_frames": int(noise[1][1].shape[2]), "acoustic_buffer_frames": int(buffer.shape[2]),
                     "pytorch_samples": int(ref_wave.shape[0]), "onnx_samples": int(wave.shape[0]),
                     "pytorch_wav_matches_reference": pytorch_wav_sha == ref_rows[i]["wav_sha256"],
                     "parity": par})
        if (i + 1) % 100 == 0:
            print(f"[{a.precision} {i + 1}/{len(samples)}] {(time.perf_counter() - t_all) / 60:.1f} min", flush=True)

    same = [r for r in rows if r["parity"]["same_shape"]]
    nrmse = np.array([r["parity"]["nrmse"] for r in same if r["parity"].get("finite")])
    summary = {"n": len(rows), "finite": sum(r["finite"] for r in rows),
               "pytorch_wav_matches_reference": sum(r["pytorch_wav_matches_reference"] for r in rows),
               "same_length_as_pytorch": len(same),
               "parity_passed": sum(r["parity"]["passed"] for r in rows),
               "nrmse_median": float(np.median(nrmse)) if len(nrmse) else None,
               "nrmse_max": float(nrmse.max()) if len(nrmse) else None,
               "max_abs_error_max": max((r["parity"]["max_abs_error"] for r in same if r["parity"].get("finite")),
                                        default=None)}
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **{k: info[k] for k in ("model_id", "config", "config_sha256", "checkpoint", "checkpoint_sha256",
                                "checkpoint_epoch", "checkpoint_global_step", "state_dict_entries", "strict_load")},
        "precision": a.precision, "graph": a.graph.name, "graph_sha256": C.sha256(a.graph),
        "graph_metadata": graph_props,
        "reference_synth": {"file": str(a.reference_synth), "sha256": C.sha256(a.reference_synth)},
        "manifest": str(a.manifest), "manifest_sha256": C.sha256(a.manifest), "n": len(rows),
        "wav_dir": str(a.wav_dir),
        "protocol": {"runtime": "onnxruntime", "provider": "CPUExecutionProvider",
                     "intra_op_threads": C.QUALITY_THREADS, "inter_op_threads": 1, "execution_mode": "sequential",
                     "pytorch_reference_threads": torch_threads, "limit": a.limit,
                     "scales": list(C.SCALES), "noise": "PyTorch noise per sentence (seed 1234) via matched-noise graph",
                     "sample_rate": sr, "timing_fields": "not for use: PyTorch reference runs in the same loop",
                     "document": "docs/evaluation-onnx-ptq.md"},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__,
                        "numpy": np.__version__},
        "script_sha256": C.sha256(Path(__file__)), "common_sha256": C.sha256(Path(C.__file__)),
        "summary": summary, "wall_time_s": time.perf_counter() - t_all, "rows": rows,
    }
    a.output.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"done {a.precision}: n={summary['n']} finite={summary['finite']} same length={summary['same_length_as_pytorch']} "
          f"parity passed={summary['parity_passed']} PyTorch WAV = reference "
          f"{summary['pytorch_wav_matches_reference']}/{summary['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
