"""Shared pieces of the ONNX export / PTQ round (protocol: ``docs/evaluation-onnx-ptq.md``).

Everything that must be identical across export, quantization, quality synthesis and timing
lives here: checkpoint loading (as ``scripts/evaluation/harvard/synthesize.py``), the export
wrapper, PyTorch noise capture, the matched-noise graph surgery, the PTQ scope, the
calibration-set check, parity metrics and the ONNX Runtime session settings.

Heavy imports (torch, onnx, onnxruntime) happen inside the functions, so the pure helpers can
be tested without them.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
NOISE_SCALE, LENGTH_SCALE, NOISE_SCALE_W = 0.667, 1.0, 0.8
SCALES = (NOISE_SCALE, LENGTH_SCALE, NOISE_SCALE_W)
SEED = 1234
OPSET = 18
PARITY = {"nrmse_max": 1e-3, "max_abs_error_max": 5e-3}
PTQ_SCOPE = ("enc_p", "dp", "flow")
PTQ_OP_TYPES = ("Conv",)
QUALITY_THREADS = 2
CALIBRATION = REPO / "docs/evidence/claim-evidence-2026-10-07/source-records/calibration/calibration60.candidate.jsonl"
NOISE_INPUT = "noise_acoustic"
SURGERY_PREFIX = "matched_noise/"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- checkpoint and PyTorch side

def load_module(config: Path, checkpoint: Path):
    """Build the training module and load the checkpoint exactly as harvard/synthesize.py does.

    Returns (module, info). The decoder's weight norm is removed, as in that script.
    """
    os.environ.setdefault("HIFIMOBINET_HOME", str(REPO))
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    import json as _json

    import torch

    from hifimobinet.training.config import load_config
    from hifimobinet.training.registry import get_training_model

    def plain(value):
        return _json.loads(_json.dumps(value))

    cfg = load_config(config)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    stored = ckpt["hyper_parameters"]
    expected = cfg.flat_hparams()
    mismatch = {k: [plain(stored.get(k)), plain(v)] for k, v in expected.items() if plain(stored.get(k)) != plain(v)}
    if mismatch:
        raise SystemExit(f"checkpoint hyper-parameters differ from {config}: {mismatch}")
    module = get_training_model(cfg.model_id).builder()(num_symbols=stored["num_symbols"], **expected)
    module.load_state_dict(ckpt["state_dict"], strict=True)
    module.eval()
    with torch.no_grad():
        module.model_g.dec.remove_weight_norm()
    info = {"model_id": cfg.model_id, "config": str(config), "config_sha256": cfg.sha256,
            "checkpoint": checkpoint.name, "checkpoint_sha256": sha256(checkpoint),
            "checkpoint_epoch": ckpt.get("epoch"), "checkpoint_global_step": ckpt.get("global_step"),
            "state_dict_entries": len(ckpt["state_dict"]), "strict_load": True,
            "sample_rate": int(stored["sample_rate"]), "inter_channels": int(stored["inter_channels"])}
    return module, info


def nan_to_num(x, nan: float, posinf: float, neginf: float):
    """``torch.nan_to_num`` with explicit values, written with ops the ONNX exporter supports."""
    import torch

    x = torch.where(torch.isnan(x), torch.full_like(x, nan), x)
    x = torch.where(x == math.inf, torch.full_like(x, posinf), x)
    return torch.where(x == -math.inf, torch.full_like(x, neginf), x)


FOLD_TOLERANCE = {"nrmse_max": 1e-5, "max_abs_error_max": 1e-4}


def fold_flow_weight_norm(model_g) -> int:
    """Fold the weight norm of every flow coupling layer's ``WN`` (EdgeTTS ``WN.remove_weight_norm``).

    The first ONNX/PTQ round (2026-10-21) left it in place: the 32 ``WN`` convolutions of the flow
    then compute weight_g * weight_v / ||weight_v|| in the graph at every run (``ReduceL2``), and
    ``quantize_static`` can only quantize such weights at run time, per tensor, as UINT8
    (protocol docs/evaluation-onnx-ptq.md, section 7). Folding is the same function up to float
    rounding; callers check the outputs before and after against ``FOLD_TOLERANCE``.
    Returns the number of ``WN`` modules folded.
    """
    folded = 0
    for layer in model_g.flow.flows:
        enc = getattr(layer, "enc", None)
        if enc is not None and hasattr(enc, "remove_weight_norm"):
            enc.remove_weight_norm()
            folded += 1
    left = [n for n, _ in model_g.flow.named_parameters() if n.endswith(("weight_g", "weight_v"))]
    if left:
        raise SystemExit(f"weight norm left in the flow after folding: {left[:4]}")
    return folded


def export_wrapper(model_g):
    """An ``nn.Module`` whose forward equals EdgeTTS ``SynthesizerTrn.infer()`` (Config A path).

    Differences, none of them numerical: the frame count stays a tensor (``infer()`` calls
    ``.item()``, which tracing would freeze), and ``nan_to_num`` is spelled out (same values).

    EdgeTTS's ``WN`` keeps its channel count in an int32 buffer ``_n_channels`` that
    ``fused_add_tanh_sigmoid_multiply`` uses as a slice bound. Exported as is, it becomes an
    int32 ``ends`` input next to int64 ``starts``, which the ONNX Slice spec does not allow
    (``onnx.checker``: "ends has inconsistent type tensor(int32)"). The buffers are therefore
    cast to int64 (same value) in ``model_g`` itself. Returns (wrapper, number of buffers cast);
    callers compare outputs before and after to show nothing numerical changed.
    """
    import torch
    from torch import nn

    from hifimobinet.training.vendor import vendor_module

    commons = vendor_module("edgetts_vits", "commons")
    if model_g.n_speakers > 1 or not model_g.use_sdp or model_g.use_f0:
        raise SystemExit("export wrapper covers the Config A path only (one speaker, SDP, no F0)")
    cast = 0
    for part in (model_g.enc_p, model_g.dp, model_g.flow, model_g.dec):
        for sub in part.modules():
            buf = getattr(sub, "_n_channels", None)
            if isinstance(buf, torch.Tensor) and buf.dtype != torch.int64:
                sub._n_channels = buf.to(torch.int64)
                cast += 1

    class ExportWrapper(nn.Module):
        # The submodules are attributes of the wrapper itself, so traced node names keep the
        # Q05 form "/enc_p/...", "/dp/...", "/flow/...", "/dec/..." that the PTQ scope uses.
        def __init__(self, g):
            super().__init__()
            self.enc_p, self.dp, self.flow, self.dec = g.enc_p, g.dp, g.flow, g.dec

        def forward(self, text, text_lengths, scales):
            g = self
            noise_scale, length_scale, noise_scale_w = scales[0], scales[1], scales[2]
            x, m_p, logs_p, x_mask = g.enc_p(text, text_lengths)
            logw = g.dp(x, x_mask, g=None, reverse=True, noise_scale=noise_scale_w)
            logw = nan_to_num(logw, 0.0, 6.0, -6.0).clamp(-6, 6)
            w = torch.exp(logw) * x_mask * length_scale
            w = nan_to_num(w, 1.0, 403.0, 0.0)
            w_ceil = torch.ceil(w).clamp(min=0, max=500)
            y_lengths = torch.clamp(torch.sum(w_ceil, [1, 2]), min=1, max=10000).long()
            y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_lengths.max()), 1).type_as(x_mask)
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = commons.generate_path(w_ceil, attn_mask)
            m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
            logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
            z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
            z = g.flow(z_p, y_mask, g=None, reverse=True)
            return g.dec(z * y_mask, g=None, f0=None)

    return ExportWrapper(model_g).eval(), cast


@contextmanager
def capture_noise():
    """Record every ``torch.randn`` / ``torch.randn_like`` value drawn inside the block (as Q05)."""
    import torch

    original_r, original_l = torch.randn, torch.randn_like
    captured = []

    def randn(*a, **kw):
        value = original_r(*a, **kw)
        captured.append(("randn", value.detach().cpu().numpy().copy()))
        return value

    def randn_like(*a, **kw):
        value = original_l(*a, **kw)
        captured.append(("randn_like", value.detach().cpu().numpy().copy()))
        return value

    torch.randn, torch.randn_like = randn, randn_like
    try:
        yield captured
    finally:
        torch.randn, torch.randn_like = original_r, original_l


def acoustic_buffer(noise, idx: int):
    """PyTorch acoustic noise [1, C, F] followed by F + 200 frames from default_rng(SEED + idx)."""
    import numpy as np

    extra = np.random.default_rng(SEED + idx).standard_normal(
        (noise.shape[0], noise.shape[1], noise.shape[2] + 200)).astype(np.float32)
    return np.concatenate([noise.astype(np.float32), extra], axis=2)


def wav_bytes(wave, sample_rate: int) -> bytes:
    """The WAV file harvard/synthesize.py writes (peak-normalized int16), as bytes."""
    from hifimobinet.training.vendor import vendor_module

    utils = vendor_module("edgetts_vits", "utils")
    wavfile = vendor_module("edgetts_vits", "wavfile")
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, utils.audio_float_to_int16(wave))
    return buf.getvalue()


# ---------------------------------------------------------------- numerics

def parity(reference, candidate) -> dict:
    """Q05 parity metrics; passed requires equal shape, finite values and both thresholds."""
    import numpy as np

    if reference.shape != candidate.shape:
        return {"same_shape": False, "reference_shape": list(reference.shape),
                "candidate_shape": list(candidate.shape), "passed": False}
    if not (np.isfinite(reference).all() and np.isfinite(candidate).all()):
        return {"same_shape": True, "finite": False, "passed": False}
    err = reference.astype(np.float64) - candidate.astype(np.float64)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    norm = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
    nrmse = rmse / max(norm, 1e-12)
    max_abs = float(np.max(np.abs(err)))
    return {"same_shape": True, "finite": True, "max_abs_error": max_abs, "rmse": rmse, "reference_rms": norm,
            "nrmse": nrmse, "passed": nrmse <= PARITY["nrmse_max"] and max_abs <= PARITY["max_abs_error_max"]}


def ort_session(path: Path, intra_threads: int = QUALITY_THREADS):
    import onnxruntime as ort

    opt = ort.SessionOptions()
    opt.intra_op_num_threads = intra_threads
    opt.inter_op_num_threads = 1
    opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=opt, providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------- graph helpers

def random_nodes(graph):
    """(duration, acoustic) RandomNormalLike nodes of a deployment graph; exactly two are allowed."""
    nodes = [n for n in graph.node if n.op_type.startswith("Random")]
    if len(nodes) != 2 or any(n.op_type != "RandomNormalLike" for n in nodes):
        raise SystemExit(f"expected two RandomNormalLike nodes, found {[(n.name, n.op_type) for n in nodes]}")
    duration = [n for n in nodes if n.name.startswith("/dp/")]
    if len(duration) != 1:
        raise SystemExit(f"expected one duration-noise node under /dp/, found {[n.name for n in nodes]}")
    acoustic = next(n for n in nodes if n is not duration[0])
    return duration[0], acoustic


def matched_noise_graph(model, inter_channels: int):
    """Copy of a deployment graph whose two random nodes are replaced by external noise.

    - duration noise: a graph input with the random node's output name, [batch, 2, phonemes];
    - acoustic noise: ``noise_acoustic[:, :, :frames]``, frames = dim 2 of the tensor the
      random node was shaped like.
    New nodes go where the random nodes were, so the node order stays topological.
    """
    import onnx
    from onnx import TensorProto, helper

    out = onnx.ModelProto()
    out.CopyFrom(model)
    g = out.graph
    duration, acoustic = random_nodes(g)
    p = SURGERY_PREFIX
    consts = [helper.make_tensor(p + "two", TensorProto.INT64, [1], [2]),
              helper.make_tensor(p + "three", TensorProto.INT64, [1], [3]),
              helper.make_tensor(p + "zero", TensorProto.INT64, [1], [0])]
    g.initializer.extend(consts)
    replacement = [
        helper.make_node("Shape", [acoustic.input[0]], [p + "like_shape"], name=p + "Shape"),
        helper.make_node("Slice", [p + "like_shape", p + "two", p + "three"], [p + "frames"], name=p + "SliceShape"),
        helper.make_node("Slice", [NOISE_INPUT, p + "zero", p + "frames", p + "two"], [acoustic.output[0]],
                         name=p + "SliceNoise"),
    ]
    nodes = []
    for n in g.node:
        if n.name == duration.name:
            continue
        if n.name == acoustic.name:
            nodes.extend(replacement)
            continue
        nodes.append(n)
    del g.node[:]
    g.node.extend(nodes)
    g.input.append(helper.make_tensor_value_info(duration.output[0], TensorProto.FLOAT, ["batch_size", 2, "phonemes"]))
    g.input.append(helper.make_tensor_value_info(NOISE_INPUT, TensorProto.FLOAT,
                                                 ["batch_size", inter_channels, "noise_frames"]))
    return out, {"duration_input": duration.output[0], "acoustic_input": NOISE_INPUT,
                 "replaced_nodes": [duration.name, acoustic.name]}


def strip_matched_noise(model):
    """Inverse check: drop what matched_noise_graph added; the result must equal the deployment
    graph with its random nodes removed."""
    import onnx

    out = onnx.ModelProto()
    out.CopyFrom(model)
    g = out.graph
    kept = [n for n in g.node if not n.name.startswith(SURGERY_PREFIX)]
    del g.node[:]
    g.node.extend(kept)
    inits = [i for i in g.initializer if not i.name.startswith(SURGERY_PREFIX)]
    del g.initializer[:]
    g.initializer.extend(inits)
    del g.input[-2:]
    return out


def without_random_nodes(model):
    import onnx

    out = onnx.ModelProto()
    out.CopyFrom(model)
    kept = [n for n in out.graph.node if not n.op_type.startswith("Random")]
    del out.graph.node[:]
    out.graph.node.extend(kept)
    return out


def qlinearconv_weight_kinds(model) -> dict:
    """Per module, how each QLinearConv gets its weight: a constant INT8 initializer with a
    per-channel (or per-tensor) scale, or a tensor produced at run time (e.g. a QuantizeLinear of
    a weight computed in the graph). Used to check that the declared PTQ recipe was applied."""
    import numpy as np
    from onnx import numpy_helper

    inits = {i.name: i for i in model.graph.initializer}
    kinds: dict = {}
    for n in model.graph.node:
        if n.op_type != "QLinearConv":
            continue
        w, w_scale = n.input[3], n.input[4]
        if w in inits:
            arr = numpy_helper.to_array(inits[w])
            scale = numpy_helper.to_array(inits[w_scale]) if w_scale in inits else np.zeros(0)
            kind = f"constant {arr.dtype} " + ("per-channel" if scale.size > 1 else "per-tensor")
        else:
            kind = "computed at run time"
        mod = kinds.setdefault(module_of(n.name) or "other", {})
        mod[kind] = mod.get(kind, 0) + 1
    return kinds


def module_of(node_name: str) -> str:
    """Top-level submodule of a traced node name ("/flow/flows.0/..." -> "flow"; "/MatMul" -> "")."""
    parts = node_name.split("/")
    return parts[1] if len(parts) > 2 else ""


def ptq_nodes(nodes):
    """Split (name, op_type) pairs into (to_quantize, to_exclude) for the fixed PTQ scope."""
    target = [(n, t) for n, t in nodes if t in PTQ_OP_TYPES]
    quantize = sorted(n for n, _ in target if module_of(n) in PTQ_SCOPE)
    exclude = sorted(n for n, _ in target if module_of(n) not in PTQ_SCOPE)
    return quantize, exclude


def check_calibration(rows, split, dataset_rows) -> dict:
    """The calibration set must be canonical train only, with phoneme IDs equal to dataset.jsonl."""
    membership = {k: set(v) for k, v in split.items()}
    by_path = {r["audio_norm_path"]: r for r in dataset_rows}
    problems = []
    for r in rows:
        path = r["audio_norm_path"]
        where = [k for k, v in membership.items() if path in v]
        if where != ["train"]:
            problems.append(f"{r['id']}: split membership {where}")
        if path not in by_path:
            problems.append(f"{r['id']}: not in dataset.jsonl")
        elif by_path[path]["phoneme_ids"] != r["phoneme_ids"]:
            problems.append(f"{r['id']}: phoneme IDs differ from dataset.jsonl")
    if len({r["id"] for r in rows}) != len(rows):
        problems.append("duplicate IDs")
    return {"n": len(rows), "train": sum(r["audio_norm_path"] in membership["train"] for r in rows),
            "val": sum(r["audio_norm_path"] in membership["val"] for r in rows),
            "test": sum(r["audio_norm_path"] in membership["test"] for r in rows), "problems": problems,
            "passed": not problems}
