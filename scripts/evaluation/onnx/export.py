"""Export one checkpoint to an FP32 ONNX deployment graph and verify it (protocol section 2).

Writes, into ``--out-dir`` (outside Git):
- ``<label>_fp32.onnx``: the deployment graph (two RandomNormalLike nodes);
- ``<label>_fp32_matched_noise.onnx``: the same graph with the random nodes replaced by
  external noise (validation and quality evaluation only);
- ``export_record.json``: hashes, structure, wrapper equality, the 13 parity cases, native-random
  checks and the ORT RNG behaviour. The run stops at the first failed check, after writing the
  record with ``status: failed``.

    python scripts/evaluation/onnx/export.py --label piper-seqir-no-vits2-cpn \
        --config configs/training/piper-seqir-no-vits2-cpn.yaml \
        --checkpoint "training_output/piper-seqir-no-vits2-cpn/checkpoints/best-epoch=1455-val_loss_mel=19.0332.ckpt" \
        --out-dir training_output/onnx-ptq/piper-seqir-no-vits2-cpn

Run from the training clone root with relative paths, in an environment with torch, onnx and
onnxruntime (the Q05 versions: torch 2.13.0, onnx 1.22.0, onnxruntime 1.23.2).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", required=True)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--manifest", default=Path("results/manifests/harvard720.json"), type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    a = p.parse_args(argv)
    if a.out_dir.exists() and any(a.out_dir.iterdir()):
        p.error("output directory exists and is not empty; choose a new one")
    a.out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from onnx import TensorProto

    torch.set_num_threads(C.QUALITY_THREADS)
    record_path = a.out_dir / "export_record.json"
    record = {"label": a.label, "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
              "protocol": "docs/evaluation-onnx-ptq.md", "script_sha256": C.sha256(Path(__file__)),
              "common_sha256": C.sha256(Path(C.__file__)), "parity_thresholds": C.PARITY,
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__,
                              "numpy": np.__version__, "provider": "CPUExecutionProvider",
                              "intra_op_threads": C.QUALITY_THREADS, "inter_op_threads": 1}}

    def check(name, ok, **detail):
        record.setdefault("checks", {})[name] = {"passed": bool(ok), **detail}
        C.write_json(record_path, record)
        if not ok:
            raise SystemExit(f"check failed: {name} {detail}")

    try:
        module, info = C.load_module(a.config, a.checkpoint)
        record["model"] = info
        g = module.model_g
        samples = json.loads(a.manifest.read_text(encoding="utf-8"))
        record["manifest"] = {"file": str(a.manifest), "sha256": C.sha256(a.manifest)}
        scales_t = torch.tensor(C.SCALES, dtype=torch.float32)

        def tensors(s, scales=C.SCALES):
            x = torch.LongTensor(s["phoneme_ids"]).unsqueeze(0)
            return x, torch.LongTensor([x.shape[1]]), torch.tensor(scales, dtype=torch.float32)

        def infer(s):
            x, xl, _ = tensors(s)
            torch.manual_seed(C.SEED)
            with torch.no_grad():
                return module(x, xl, list(C.SCALES))

        # 1. infer() before the int64 cast of the WN buffers, infer() after it, and the wrapper:
        #    all three must be identical (same seed, same ops)
        check_rows = samples[:3]
        before = [infer(s) for s in check_rows]
        wrapper, cast = C.export_wrapper(g)
        record["wn_n_channels_buffers_cast_to_int64"] = cast
        after = [infer(s) for s in check_rows]
        wrapped = []
        for s in check_rows:
            x, xl, sc = tensors(s)
            torch.manual_seed(C.SEED)
            with torch.no_grad():
                wrapped.append(wrapper(x, xl, sc))
        check("int64_cast_leaves_infer_unchanged", all(torch.equal(b, c) for b, c in zip(before, after)),
              sentences=[s["idx"] for s in check_rows], buffers_cast=cast)
        check("wrapper_equals_infer", all(torch.equal(b, w) for b, w in zip(before, wrapped)),
              sentences=[s["idx"] for s in check_rows])
        record["flow_weight_norm_removed"] = False  # EdgeTTS ResidualCouplingBlock has no remove_weight_norm
        check("flow_has_no_remove_weight_norm", not hasattr(g.flow, "remove_weight_norm"))

        # 2. export the deployment graph
        deploy = a.out_dir / f"{a.label}_fp32.onnx"
        dummy = tensors(samples[0])
        with torch.no_grad():
            torch.onnx.export(wrapper, dummy, str(deploy), opset_version=C.OPSET, dynamo=False,
                              input_names=["input", "input_lengths", "scales"], output_names=["output"],
                              dynamic_axes={"input": {0: "batch_size", 1: "phonemes"}, "input_lengths": {0: "batch_size"},
                                            "output": {0: "batch_size", 2: "samples"}})
        graph = onnx.load(deploy)
        onnx.helper.set_model_props(graph, {
            "label": a.label, "model_id": info["model_id"], "checkpoint_sha256": info["checkpoint_sha256"],
            "checkpoint_epoch_zero_based": str(info["checkpoint_epoch"]), "protocol": "docs/evaluation-onnx-ptq.md",
            "output_contract": "float32 [batch, 1, samples]", "precision": "FP32", "purpose": "deployment graph"})
        onnx.checker.check_model(graph, full_check=True)
        onnx.save(graph, deploy)
        out_dims = graph.graph.output[0].type.tensor_type.shape.dim
        check("output_rank_3_dynamic_samples", len(out_dims) == 3 and out_dims[2].dim_param == "samples")
        check("no_enc_q_initializer", not any("enc_q" in i.name for i in graph.graph.initializer))
        check("no_quantized_node", not any("QuantizeLinear" in n.op_type or "QLinear" in n.op_type for n in graph.graph.node))
        check("no_fp16_initializer", not any(i.data_type == TensorProto.FLOAT16 for i in graph.graph.initializer))
        duration, acoustic = C.random_nodes(graph.graph)
        ops = Counter(n.op_type for n in graph.graph.node)
        record["graph"] = {"file": deploy.name, "sha256": C.sha256(deploy), "bytes": deploy.stat().st_size,
                           "opset": C.OPSET, "nodes": len(graph.graph.node), "initializers": len(graph.graph.initializer),
                           "op_counts": dict(sorted(ops.items())),
                           "conv_by_module": dict(sorted(Counter(C.module_of(n.name) for n in graph.graph.node
                                                                 if n.op_type == "Conv").items())),
                           "random_nodes": [{"name": n.name, "op": n.op_type, "output": list(n.output)}
                                            for n in (duration, acoustic)]}
        C.write_json(record_path, record)

        # 3. matched-noise graph: only the two random nodes are replaced
        matched, surgery = C.matched_noise_graph(graph, info["inter_channels"])
        check("matched_noise_strips_to_deployment_without_random",
              C.strip_matched_noise(matched).SerializeToString() == C.without_random_nodes(graph).SerializeToString())
        onnx.helper.set_model_props(matched, {**{p.key: p.value for p in graph.metadata_props},
                                              "purpose": "validation graph with external noise; not a deployment artifact"})
        onnx.checker.check_model(matched, full_check=True)
        matched_path = a.out_dir / f"{a.label}_fp32_matched_noise.onnx"
        onnx.save(matched, matched_path)
        record["matched_noise_graph"] = {"file": matched_path.name, "sha256": C.sha256(matched_path), **surgery}

        # 4. parity gate: 13 cases against PyTorch infer() with the same noise
        sess = C.ort_session(matched_path)
        cases = [(s["idx"], s, C.SCALES, C.SEED) for s in samples[:10]]
        cases += [(samples[2]["idx"], samples[2], C.SCALES, C.SEED + 1),
                  (samples[2]["idx"], samples[2], (C.NOISE_SCALE, 0.8, C.NOISE_SCALE_W), C.SEED),
                  (samples[2]["idx"], samples[2], (C.NOISE_SCALE, 1.2, C.NOISE_SCALE_W), C.SEED)]
        record["parity_cases"] = []
        for idx, s, scales, seed in cases:
            x, xl, sc = tensors(s, scales)
            torch.manual_seed(seed)
            with torch.no_grad(), C.capture_noise() as noise:
                ref = module(x, xl, list(scales)).numpy()
            kinds = [k for k, _ in noise]
            feeds = {"input": x.numpy(), "input_lengths": xl.numpy(), "scales": sc.numpy(),
                     surgery["duration_input"]: noise[0][1], C.NOISE_INPUT: C.acoustic_buffer(noise[1][1], idx)}
            got = sess.run(None, feeds)[0]
            result = {"idx": idx, "seed": seed, "scales": list(scales), "noise_kinds": kinds, **C.parity(ref, got)}
            record["parity_cases"].append(result)
            C.write_json(record_path, record)
            check(f"parity_idx{idx}_seed{seed}_len{scales[1]}", kinds == ["randn", "randn_like"] and result["passed"],
                  nrmse=result.get("nrmse"), max_abs_error=result.get("max_abs_error"))
        del sess

        # 5. native-random deployment graph and the ORT RNG behaviour (as Q05)
        native = C.ort_session(deploy)
        record["native_random_smoke"] = []
        for s in (samples[0], samples[359], samples[719]):
            x, xl, sc = tensors(s)
            wave = native.run(None, {"input": x.numpy(), "input_lengths": xl.numpy(), "scales": sc.numpy()})[0]
            ok = wave.ndim == 3 and wave.shape[:2] == (1, 1) and wave.shape[2] > 4410 and bool(np.isfinite(wave).all()) \
                and float(np.std(wave)) > 1e-6
            record["native_random_smoke"].append({"idx": s["idx"], "shape": list(wave.shape), "std": float(np.std(wave)),
                                                  "passed": ok})
            check(f"native_random_idx{s['idx']}", ok)
        del native
        feeds = {"input": dummy[0].numpy(), "input_lengths": dummy[1].numpy(), "scales": scales_t.numpy()}
        ort.set_seed(C.SEED)
        s1 = C.ort_session(deploy)
        v1, v2 = s1.run(None, feeds)[0], s1.run(None, feeds)[0]
        ort.set_seed(C.SEED)
        v3 = C.ort_session(deploy).run(None, feeds)[0]
        record["ort_rng_check"] = {"seed": C.SEED, "fresh_sessions_same_seed_equal": bool(np.array_equal(v1, v3)),
                                   "consecutive_calls_same_session_equal": bool(np.array_equal(v1, v2)),
                                   "scope": "this CPU runtime and input only"}
        check("checkpoint_unchanged", C.sha256(a.checkpoint) == info["checkpoint_sha256"])
        record.update(status="passed", finished_utc=datetime.now(timezone.utc).isoformat())
        C.write_json(record_path, record)
        print(f"PASSED {a.label} {record['graph']['sha256'][:16]} max NRMSE "
              f"{max(c['nrmse'] for c in record['parity_cases']):.2e}")
        return 0
    except BaseException as error:  # record any failure, including SystemExit from check()
        record.update(status="failed", error="".join(traceback.format_exception(error)),
                      finished_utc=datetime.now(timezone.utc).isoformat())
        C.write_json(record_path, record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
