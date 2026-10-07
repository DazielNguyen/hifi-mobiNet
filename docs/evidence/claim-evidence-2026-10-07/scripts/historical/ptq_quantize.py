#!/usr/bin/env python3
"""Generic PTQ (no training) exclusion-sweep quantizer for a BanhmiTTS FP32
ONNX export. Mirrors QAT-Training/qat_transfer/piper_train/export_int8.py's
approach (static QOperator INT8, per-channel weights) but parameterized by
which top-level model_g submodules + op types to quantize, so the same
script drives the whole PTQ sweep across many exclusion configs for both
the HiFi-GAN ("baseline") and Vocos ("vocos_small") architectures.

Node names come straight from the traced ONNX graph (verified once via
inspection): "/enc_p/...", "/dp/...", "/flow/...", "/dec/...". A node is
INCLUDED for quantization only if:
  1. its op_type is in --op-types, AND
  2. its name starts with "/<submodule>/" for some submodule in --submodules, AND
  3. its name does not contain any of --exclude-leaf-substrings, AND
  4. its exact name is not in the hardcoded ALWAYS_EXCLUDE set (fixed/
     non-trainable ops: the ISTFT's identity conv_transpose1d scatter
     kernels in Vocos's head, and the duration-alignment matmuls computed
     directly in SynthesizerTrn.infer() against attn/m_p/logs_p -- neither
     has real learned weights, so "quantizing" them would just inject noise
     into a deterministic reshape/alignment op for no size benefit).
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

# Present in both architectures identically (SynthesizerTrn.infer()'s
# attn-expansion matmuls for m_p/logs_p -- no learned weights).
ALWAYS_EXCLUDE_EXACT = {"/MatMul", "/MatMul_1"}
# Vocos-only: ISTFTHead's fixed identity-scatter overlap-add kernels.
ALWAYS_EXCLUDE_VOCOS = {"/dec/head/ConvTranspose", "/dec/head/ConvTranspose_1"}


class JsonlCalibReader(CalibrationDataReader):
    def __init__(self, dataset_jsonl: Path, n_samples: int, seed: int = 42):
        entries = []
        with open(dataset_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        rng = random.Random(seed)
        self.samples = rng.sample(entries, min(n_samples, len(entries)))
        self.i = 0

    def get_next(self):
        if self.i >= len(self.samples):
            return None
        entry = self.samples[self.i]
        self.i += 1
        ids = np.array(entry["phoneme_ids"], dtype=np.int64)[None, :]
        return {
            "input": ids,
            "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
            "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fp32_onnx")
    parser.add_argument("output_onnx")
    parser.add_argument("--submodules", required=True, help="Comma-separated: enc_p,dp,flow,dec (which to QUANTIZE)")
    parser.add_argument("--op-types", required=True, help="Comma-separated: e.g. Conv,ConvTranspose or Conv,ConvTranspose,MatMul")
    parser.add_argument("--exclude-leaf-substrings", default="", help="Comma-separated substrings; matching nodes skip quantization (e.g. conv_post or head/out)")
    parser.add_argument("--is-vocos", action="store_true", help="Also apply the Vocos-specific always-exclude (fixed ISTFT ConvTranspose nodes)")
    parser.add_argument("--calibration-dataset", required=True)
    parser.add_argument("--calibration-sentences", type=int, default=60)
    args = parser.parse_args()

    submodules = tuple(s.strip() for s in args.submodules.split(",") if s.strip())
    op_types = tuple(s.strip() for s in args.op_types.split(",") if s.strip())
    exclude_substrings = tuple(s.strip() for s in args.exclude_leaf_substrings.split(",") if s.strip())

    always_exclude = set(ALWAYS_EXCLUDE_EXACT)
    if args.is_vocos:
        always_exclude |= ALWAYS_EXCLUDE_VOCOS

    model = onnx.load(args.fp32_onnx)
    all_target_type_nodes = [n.name for n in model.graph.node if n.op_type in op_types]

    def in_scope(name: str) -> bool:
        return any(name.startswith(f"/{sm}/") for sm in submodules)

    nodes_to_quantize_candidates = [n for n in all_target_type_nodes if in_scope(n)]
    exclude_nodes = [n for n in all_target_type_nodes if not in_scope(n)]
    # exclude_leaf_substrings further removes matches from the in-scope set
    exclude_nodes += [
        n for n in nodes_to_quantize_candidates
        if any(sub in n for sub in exclude_substrings)
    ]
    exclude_nodes += [n for n in all_target_type_nodes if n in always_exclude]
    exclude_nodes = sorted(set(exclude_nodes))

    n_quantized = len(all_target_type_nodes) - len(exclude_nodes)
    print(f"Quantizing {n_quantized}/{len(all_target_type_nodes)} target-type nodes "
          f"(submodules={submodules}, op_types={op_types}, exclude_substrings={exclude_substrings})")

    calib_reader = JsonlCalibReader(Path(args.calibration_dataset), args.calibration_sentences)

    quantize_static(
        args.fp32_onnx,
        args.output_onnx,
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QOperator,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        op_types_to_quantize=list(op_types),
        nodes_to_exclude=exclude_nodes,
        per_channel=True,
    )
    print(f"Wrote {args.output_onnx}")


if __name__ == "__main__":
    main()
