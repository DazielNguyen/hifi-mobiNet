# Evidence map

How to read this map:
- Each claim lists the number, the record it comes from (path#locator) and the evidence class.
- All paths are relative to this package unless prefixed `repo:`.
- SHA-256 values are in `claim-evidence-index.json` and `source-manifest.json`.
- All claims: `journal_verification: NOT_CHECKED`.

## J-C001: baseline relation to Piper and VITS2

- **Names compared:** parameter and persistent-buffer names only. Shapes, values, forward behaviour, executed code and code origin are not compared.
- **Inventories:** `outputs/J-C001_key_inventories.json` holds the full name lists (current check, 2026-10-07).

| Left | Right | Left | Right | Common |
|---|---|---:|---:|---:|
| baseline checkpoint (epoch 1489) | relocated BanhmiTTS code, `configs/training/baseline-resblock2-vits2.yaml` | 876 | 876 | 876 |
| baseline checkpoint | vendored EdgeTTS a73a897, `use_vits2=True` | 876 | 881 | 876 |
| published Piper checkpoint | EdgeTTS Config A (Piper-base comparator) | 784 | 789 | 784 |
| baseline checkpoint | Piper checkpoint | 92 only left | 0 only right | 784 |

- The 92 baseline-only names are 72 `model_g.flow.*.attn.*` names and 20 `model_d_dur.*` names.
- The 5 names that exist only in EdgeTTS code are `_n_channels` buffers. They show that the checkpoint differs from current EdgeTTS code. They do not show which code trained the baseline.
- **Author account (kept separate):** the baseline was rewritten following Piper's design and adds selected VITS2 components; Piper was not forked.
- **VITS2 differences (code/config reading):** the posterior encoder takes 513-bin linear spectrograms, not 80-band mel, and blank tokens are kept. Source: VITS2 paper, Interspeech 2023, doi 10.21437/Interspeech.2023-534.

## J-C010: calibration source and scope selection

- **Actually used:** `source-records/calibration/historical_calibration60_recomputed.json`.
  - Rule from `scripts/historical/ptq_quantize.py#JsonlCalibReader`: `random.Random(42).sample(all non-empty dataset.jsonl rows, 60)`.
  - Canonical split membership: **57 train, 3 test**.
  - No manifest was saved at run time; the list is reconstructed.
  - Re-sampling from `dataset.jsonl` (SHA-256 `f4a72ae0…295b`) reproduces the same 60 ids (`outputs/verify_report.json#J-C010 dataset`).
- **The 3 test sentences:**
  1. "He assassinated the President, shot Officer Tippit, resisted arrest and tried to kill another policeman in the process."
  2. "the Director and Deputy Director for Plans of the CIA testified concerning that Agency's limited knowledge of Oswald before the assassination."
  3. "there were two flat lightweight curtain rods belonging to Ruth Paine but they were still there on Friday afternoon after Oswald's arrest."
- **Designed, not used:**
  - `source-records/calibration/calibration60.candidate.jsonl`: 60/60 canonical train.
  - `PROTOCOL_Q04_v0.2.json`: candidate, with `selection_rule` null.
  - No INT8 graph was produced after 2026-10-04.
- **Scope selection:** there is no record of a separate validation set being used to choose the scope.

## J-C011: checkpoint → ONNX FP32 → parity (Q05 graphs)

The graph SHA-256 values below are also the published files at `DazielNguyen/hifi-mobiNet-inference` revision `f760e85a` (`repo:docs/huggingface-inference-release.json`).

| Model | Checkpoint SHA-256 (metadata = run record) | FP32 graph SHA-256 | Cases | Max NRMSE | Max abs error |
|---|---|---|---:|---:|---:|
| baseline (ep 1489) | `921b0f6d7bfc82e4…` | `fdfda3ac0ce86ca7…` | 10 | 9.23e-5 | 2.73e-4 |
| MRF (ep 1386) | `7ab302dcd80af326…` | `14c9f547208403b4…` | 10 | 1.14e-4 | 5.58e-4 |
| SEQ (ep 1442) | `c4670261a9c775da…` | `4052629a26250858…` | 10 | 4.41e-5 | 1.42e-4 |
| Piper (ep 999) | `dcf2449bdbdaad09…` | `2c87a6c368b3f03f…` | 10 | 7.06e-5 | 1.34e-4 |

- **Formula:** `err = pytorch − onnx`, `NRMSE = sqrt(mean(err²)) / max(sqrt(mean(pytorch²)), 1e-12)`.
- **Pass rule:** same shape, finite, NRMSE ≤ 1e-3 and max|err| ≤ 5e-3. All 10 cases must pass.
- **Recomputation:** recomputed from the 40 stored `parity_case_*.npz` files; every value equals the run report.
- **Thresholds were prespecified:** `source-records/q05/RUN_CONFIG.json` was created 2026-10-04T00:08:30Z with `thresholds_declared_before_runs`. The first model started at 00:09:02Z (`REPORT.baseline.json#started_utc`).
- **Identical initializers, not identical pipeline:**
  - The validation graphs (noise as inputs) and the deployed graphs have identical initializers (422/478/454/382).
  - They differ only in 2 `RandomNormalLike` nodes.
  - Source: `outputs/pc/J-C011_onnx_export_parity.json#validation_graph_vs_deployed`.
- **Metadata hash limit:** the metadata hash shows what the exporter recorded. It does not prove on its own which checkpoint bytes were read at export.

## J-C012, J-C013, J-C015

`source-records/search-records.md` (AI search log). Conclusion for each: **not found in the scanned scope**.

## J-C019: SEQ vs MRF quality

Source: `repo:results/prior_audit/harvard_stats_recomputed.json#paired_tests.seq_vs_mrf_*`, produced by `repo:scripts/evaluation/original/recompute_B_stats.py`. Direction: SEQ − MRF.

| Metric | Mean diff | 95% CI | Wilcoxon p | Holm p |
|---|---:|---|---:|---:|
| WER (fraction of words, macro over 720) | +0.0149 | [+0.0019, +0.0277] (= +0.19 to +2.77 pp) | 0.052 | 0.094 |
| UTMOSv2 | −0.0237 | [−0.0458, −0.0010] | 0.047 | 0.094 |
| Mean RTF reduction | 9.08% | n/a | 7.0e-37 | n/a |

- **Bootstrap:** paired, 10,000 resamples, percentile, `default_rng(20261004)`, with the RNG stream shared across 15 tests.
- **Reading:** equivalence was not established (no margin, no equivalence test). The data describe a runtime–quality trade-off.
- **Not CONTRADICTED:** a CI that excludes 0 is not an equivalence test.

## J-C023: speed across ONNX FP32 and INT8

The historical ONNX results are in `repo:results/historical/onnx_historical/`.

| Comparison | Paired sentences | SEQ faster | Median RTF ratio |
|---|---:|---:|---:|
| SEQ FP32 vs **baseline** FP32 | 500 | 495 | 0.890 |
| SEQ INT8 vs **baseline** INT8 | 500 | 498 | 0.838 |
| SEQ FP32 vs MRF FP32 (different sentence sets) | 21 | 21 | 0.821 |

Benchmark graphs and how they link to the Q05 graphs and checkpoints:

| Model | Q05 FP32 (identified) | Benchmark FP32 | Link to checkpoint | Benchmark INT8 | INT8 → FP32 link |
|---|---|---|---|---|---|
| baseline | `fdfda3ac…` | `ptq_sweep/baseline_fp32.onnx` `16a8943b…` | **confirmed**: 422/422 named weights = ckpt 1489 | `baseline_flow_enc_p_dp.onnx` `6f02e294…` | **confirmed**: 45/45 FP32 decoder initializers byte-identical |
| SEQ | `4052629a…` | `seq_final_fp32.onnx` `3226c018…` | **confirmed**: 454/454 = ckpt 1442 | `seq_final_int8.onnx` `e6e30645…` | **confirmed**: 77/77 |
| MRF | `14c9f547…` | `ptq_sweep/mrf_fp32.onnx` `8f89ebc4…` | **not linked**: 0/478 = ckpt 1386 | `mrf_flow_enc_p_dp.onnx` `28b69a59…` | 101/101, so equally unidentified |

- **Why MRF is identified in Q05 but not in the benchmark:**
  - The Q05 MRF graph was exported on 2026-10-04 from checkpoint 1386 and records its hash.
  - The benchmark graph has no metadata, and none of its named weights equal checkpoint 1386. It came from another checkpoint/export.
- **Protocol (from records):**
  - ORT CPU, intra-op 2, inter-op 1, no warm-up, one timing per utterance.
  - Runs were made on different days.
  - SEQ's INT8 scope is wider (see J-C024).
- **PyTorch level:** Harvard-720 mean RTF of SEQ is 9.08% below MRF (J-C019 source).

## J-C024: PTQ recipe actually applied (historical graphs)

From `outputs/verify_report.json#J-C024` and `outputs/pc/J-C024_J-C025_graph_structure.json`:

| INT8 graph | QLinearConv | QLinearMatMul | By module | Weights / activations | Under `/dec/` |
|---|---:|---:|---|---|---:|
| baseline `6f02e294…` | 133 | 0 | enc_p 37, dp 32, flow 64 | 133/133 per-channel INT8, UINT8 | 0 |
| SEQ `e6e30645…` | 133 | 40 | + enc_p 24, flow 16 MatMul | same | 0 |
| MRF (unidentified) `28b69a59…` | 133 | 0 | as baseline | same | 0 |

- **Calibration:** as recorded under J-C010 (57 train / 3 test).
- **Not used:** the designed 60/60 train-only set.

## J-C025: flow description, parameter count, `/flow/` scope

Counted from the Q05 FP32 graphs (`outputs/verify_report.json#J-C025`).

**Counting method:**
- Count the unique initializers consumed by `/flow/*` nodes or named `flow.*`.
- Biases are included; Piper's `weight_g` is subtracted.
- No initializer is shared with nodes outside the flow.
- Constant-node tensors are not counted.

| Graph | Elements | of which bias | Conv | MatMul | Softmax | LayerNorm |
|---|---:|---:|---:|---:|---:|---:|
| baseline / MRF / SEQ (`fdfda3ac…`, `14c9f547…`, `4052629a…`) | 8,285,568 | 18,048 | 64 | 16 | 4 | 8 |
| Piper (`2c87a6c3…`) | 7,102,080 − 11,520 `weight_g` = 7,090,560 | 12,672 | 40 | 0 | 0 | 0 |

- Op counts name operators, not module design.
- The architecture description is separate from the PTQ `/flow/` scope (J-C024): all 64 flow convs were quantized, and the flow MatMuls only in SEQ.

## J-C026: Harvard-720 exposure

- **Project models:**
  - 13,100 LJSpeech transcripts versus 720 Harvard sentences.
  - Exact matches: 0 under both normalizations (n1 alphanumeric-lowercase, n2 jiwer WER normalization).
  - Shared n-grams: 37 sentences share a 4-gram, 7 a 5-gram, 0 a 6-gram.
  - Identical phoneme-ID sequences: 0.
  - Recomputed from the transfer `dataset.jsonl`; equal to `source-records/data-overlap/data_split_checks_recomputed.json`.
- **Piper:**
  - The project's `lj-med_1000.ckpt` SHA-256 `dcf2449b…` equals the published LFS oid in `rhasspy/piper-checkpoints` (revision `37d1bea2`, accessed 2026-10-07).
  - The MODEL_CARD (`rhasspy/piper-voices` revision `c10ece1a`, git blob `324d7470…` = stored copy) says: "Trained from scratch for 1000 epochs on medium quality settings using the LJ Speech dataset."
  - The checkpoint hparams record `dataset training/lj-med`, `max_epochs 1000` and `resume_from_checkpoint null`.
- **Phoneme maps:**
  - The project map has 159 symbols and Piper's has 157. The 157 shared symbols have the same IDs.
  - No Harvard item uses an ID outside Piper's map.
- **Scope:** text-level comparison only, not "never seen".
