# Results

*Tiếng Việt: [README.vi.md](README.vi.md)*

This folder holds every stored result of hifi-mobiNet's two generations of models:
- **Gen 1:** the historical BanhmiTTS models: HiFi-GAN ResBlock2 baseline with VITS2 components, Parallel-IR (MRF), Sequential-IR (SEQ) and the external Piper checkpoint.
- **Gen 2:** the controlled Config A models of October 2026: Piper_no_VITS2_cpn (ResBlock2), Piper-ParIR and Piper-SeqIR.

`manifest.json` lists every file with its SHA-256, classification and limits.

| Section | What it covers |
|---|---|
| [Historical results](#historical-results-gen-1) | `historical/`, `prior_audit/`, `manifests/` |
| [New measurements](#new-measurements-gen-2-pytorch) | `piper-no-vits2-cpn/`, `piper-parir-no-vits2-cpn/`, `piper-seqir-no-vits2-cpn/`, `piper-ir-timing-session/` |
| [ONNX export and INT8 PTQ](#onnx-export-and-int8-ptq) | `piper-ir-onnx-ptq/` and the historical ONNX, PTQ and QAT records |
| [Seven-model comparison](#seven-model-comparison) | one table with every metric of the seven Harvard-720 models |
| [Model size](#model-size) | `model_parameters.json` |

The overview of all SEQ/PAR evidence and its readiness for a journal paper is `docs/q2-readiness-seq-par.md`.

## Historical results (Gen 1)

**Where the files come from:**
- **Results:** the result files under `historical/harvard/`, `historical/lj500/` and `historical/onnx_historical/` are byte-identical to the files of the journal evidence package. Their hashes, source locators and limitations are in `manifest.json` and `docs/source-map.json`.
- **Later imports:** the folders imported on 2026-10-22 are described below.
- **Nothing new:** no new research measurement was produced to fill these folders.

| Model | Harvard mean RTF | UTMOSv2 predicted | Macro WER |
|---|---:|---:|---:|
| HiFi-GAN ResBlock2 | 0.04177 | 3.0260 | 0.1476 |
| Parallel-IR | 0.03388 | 3.5680 | 0.1073 |
| Sequential-IR | 0.03081 | 3.5443 | 0.1222 |
| Piper external checkpoint | 0.03916 | 3.4507 | 0.1747 |

**Limits of these observations:**
- They are **historical PyTorch CPU observations** on 720 common Harvard sentences, from the development workstation under WSL. They are not edge-device or ONNX results.
- UTMOSv2 is not human MOS.
- There was one training run per model, one synthesis draw per sentence, and no repeated timing protocol.
- The historical runtime versions were inferred from nearby records; they were not hashed inside every run.
- **Q05 exports:** the Q05 FP32 exports are identified and passed engineering checks, but the quality scores above are not bound to them. The demo is a functional interface, not evidence that the new code reproduces the table above.

**Folders:**
- **`lj500/`:** keeps both the selected MRF epoch-1386 evaluation (`eval_mrf_final1500_500_results.json`) and an earlier MRF table; file names and values are unchanged.
  - The MRF set and the canonical LJSpeech test set share only 21 IDs. Do not read the 500-row files as one common paired test.
- **`manifests/`:**
  - It includes the canonical split, which was observed later, and the frozen ordered IDs of Q04.
  - This does not prove that every manifest was recorded at training time.
- **`onnx_historical/`:** explicitly exploratory, kept for transparency, not as final ONNX/PTQ claims (see [Historical PTQ and QAT](#historical-ptq-and-qat-gen-1)):
  - the legacy MRF graph is not bound to the selected checkpoint;
  - the historical calibration overlaps test data;
  - the PTQ scopes differ between models.
- **`prior_audit/`:** an unchanged later statistical audit.
  - The portable statistical script reproduces it into `outputs/`, leaving the raw inputs untouched.
  - Paired tests and bootstrap intervals describe sentence-level variation, not variation across training runs or hardware.
  - Non-significance does not establish equivalence.
  - Recipe and split differences prevent a claim of decoder-only causation.

Imported on 2026-10-22 (`docs/evidence/historical-import-2026-10-22.json`). Result bytes are unchanged, and the scripts have their host paths redacted.
- **`lj500_development/`:** further historical LJ-500 tables.
  - Earlier baseline checkpoints.
  - The MRF training trajectory: epochs 74, 216, 353, 480, 836 and 1170.
  - The SEQ precursors (`mbconv`), whose checkpoints are deleted.
  - ResBlock2 variants with Snake, MRD and BigVGAN.
  - The scripts are in `records/`. Logs are not tracked (the repository ignores `*.log`); the import record lists each one by hash.
- **`qat/`:** QAT of baseline and MRF. These tables include extra training epochs and PyTorch fake-quant timing, so they are not INT8 runtime results.
- **`ptq_sweep/`:** the historical PTQ scope sweep.
  - Its WER evaluator had a known sample-rate bug (J-E026).
  - The result files are derived copies without their Vocos entries (`*.without_vocos.json`). Logs that include Vocos runs are referenced by hash only.
- **Sentence sets:** the LJ-500 tables use three different 500-sentence sets. Compare tables only within one set.

## New measurements (Gen 2, PyTorch)

**`piper-no-vits2-cpn/`** (2026-10-10/11): the first new measurements in this repository.
- It is the Harvard-720 evaluation of `Piper_no_VITS2_cpn`, the baseline without the VITS2 components.
- It contains its own README, per-sentence results, scorer records, paired comparisons with the historical files above, and same-session controls.
- These files add to the historical tables; they do not replace them.
- The historical RTF values are not comparable with the RTF of that session: the same baseline model measured a timing drift of +8.3%.

**`piper-parir-no-vits2-cpn/`** (2026-10-16): the Harvard-720 quality evaluation of `Piper_ParIR_no_VITS2_cpn`.
- It uses the same backbone with the released Parallel-IR decoder bundle: inverted-residual decoder, SnakeBeta and MRD.
- It is compared by sentence with `Piper_no_VITS2_cpn` and the four historical systems.
- It includes a same-session scorer control that rescored the `Piper_no_VITS2_cpn` WAVs.
- Its RTF was measured later, in the 2026-10-21 session.

**`piper-seqir-no-vits2-cpn/`** (2026-10-21): the evaluation of `Piper_SeqIR_no_VITS2_cpn`, the same backbone with the released Sequential-IR decoder bundle (two IR blocks per stage, SnakeBeta, MRD).
- It is compared by sentence with the two other Config A models and the four historical systems.
- It includes same-session rescoring of the `Piper_no_VITS2_cpn` and ParIR WAVs.
- Its README also reports the idle-machine timing session, which timed the three Config A models back to back with `Piper_no_VITS2_cpn` as the control. The records are in `piper-ir-timing-session/`.

| New model | UTMOSv2 (predicted) | Macro WER | Mean RTF, 2026-10-21 session | Relative RTF |
|---|---:|---:|---:|---:|
| Piper_no_VITS2_cpn (ResBlock2, control) | 3.517 | 0.0612 | 0.0510 | — |
| Piper_ParIR_no_VITS2_cpn | 3.530 | 0.0726 | 0.0392 | −23.2% |
| Piper_SeqIR_no_VITS2_cpn | 3.493 | 0.0696 | 0.0340 | −33.3% |

The RTF column comes from one session and is comparable only within it. The control ran 1.21× slower there than in its own 2026-10-10 session (0.0421), so neither that value nor the historical RTF table can be set beside these values.

## ONNX export and INT8 PTQ

### Gen 2: the 2026-10-21 round

`piper-ir-onnx-ptq/` ([README](piper-ir-onnx-ptq/README.md)) holds the ONNX FP32 export, static INT8 PTQ and Harvard-720 evaluation of the three Config A models. The protocol, `docs/evaluation-onnx-ptq.md`, was fixed before the runs. The scripts are in `scripts/evaluation/onnx/`.

**Setup:**
- **Export:**
  - opset 18, the Q05 interface;
  - an export wrapper equal to EdgeTTS `infer()`;
  - parity gate NRMSE ≤ 1e-3 and max abs ≤ 5e-3.
- **PTQ** (the recipe of the historical baseline/MRF graphs):
  - ONNX Runtime `quantize_static`, QOperator, QInt8 per-channel weights, QUInt8 activations, MinMax calibration;
  - `Conv` nodes under `/enc_p/`, `/dp/` and `/flow/`; the decoder stays FP32.
  - **Calibration:** the 60-sentence train-only set (0 validation or test sentences).
- **Quality:** each graph synthesized Harvard-720 through a matched-noise copy that uses the PyTorch noise of every sentence. All six sets were scored in one session with `harvard/score.py`.
- **Speed:**
  - one idle session; ORT CPU, sequential, intra-op 2 and 12 threads;
  - round A, then round B in reverse order;
  - one pass per sentence, no warm-up.

**Checks:**

| | Piper_no_VITS2_cpn | Piper-ParIR | Piper-SeqIR |
|---|---:|---:|---:|
| Parity gate, 13 cases: max NRMSE | 1.07e-4 | 8.90e-5 | 8.59e-5 |
| Harvard-720 parity, FP32 | 720 / 720 | 720 / 720 | 720 / 720 |
| QLinearConv (enc_p / dp / flow) | 37 / 32 / 40 | 37 / 32 / 40 | 37 / 32 / 40 |
| Decoder initializers byte-identical to FP32 | yes | yes | yes |
| Second quantization byte-identical | yes | yes | yes |
| PyTorch reference WAVs = stored evaluation | 720 / 720 | 720 / 720 | 720 / 720 |

**Quality on Harvard-720:**

| Model | PyTorch (stored) UTMOSv2 / WER | ONNX FP32 UTMOSv2 / WER | ONNX INT8 UTMOSv2 / WER | INT8 − FP32: ΔUTMOSv2 [95% CI] | INT8 − FP32: ΔWER [95% CI] |
|---|---|---|---|---|---|
| Piper_no_VITS2_cpn | 3.517 / 0.0612 | 3.508 / 0.0612 | 3.503 / **0.0620** | −0.005 [−0.023, +0.013] | +0.001 [−0.006, +0.008] |
| Piper-ParIR | 3.530 / 0.0726 | 3.524 / 0.0726 | **3.524** / 0.0772 | +0.000 [−0.018, +0.019] | +0.005 [−0.003, +0.012] |
| Piper-SeqIR | 3.493 / 0.0696 | 3.495 / 0.0696 | 3.459 / 0.0739 | **−0.036 [−0.056, −0.016]** (Holm 0.0013) | +0.004 [−0.003, +0.011] |

- **Export:** FP32-ONNX against PyTorch has an identical WER on every sentence, and UTMOSv2 within 0.01, which is scorer variation.
- **INT8:** no detectable change for Piper_no_VITS2_cpn and ParIR. SeqIR loses 0.036 UTMOSv2, with no detectable WER change.

**Speed** (mean RTF, round A; round B in brackets; every INT8-vs-FP32 CI excludes 0):

| Model | 2 threads FP32 | 2 threads INT8 | INT8 vs FP32 | 12 threads FP32 | 12 threads INT8 | INT8 vs FP32 |
|---|---:|---:|---:|---:|---:|---:|
| Piper_no_VITS2_cpn | 0.0420 (0.0439) | 0.0376 (0.0394) | −10.5% | 0.0292 (0.0271) | 0.0277 (0.0247) | −4.9% |
| Piper-ParIR | 0.0397 (0.0453) | 0.0341 (0.0399) | −13.9% | 0.0397 (0.0362) | 0.0366 (0.0348) | −7.8% |
| Piper-SeqIR | **0.0323** (0.0323) | **0.0269** (0.0270) | −16.8% | 0.0312 (0.0319) | 0.0297 (0.0291) | −4.8% |

- **Thread dependence:** at 2 threads SeqIR is fastest, 23–28% below ResBlock2. At 12 threads ResBlock2 is fastest and ParIR is 32–36% slower.
- **Cause, from the kernel profile in `piper-ir-onnx-ptq/profile/`:** SnakeBeta's `Sin` does not use ONNX Runtime 1.23's thread pool and even slows down under idle-thread spinning, while ResBlock2's dense convolutions scale with threads.
- **ORT against PyTorch:** with 12 threads, ORT FP32 runs the ResBlock2 model at 0.59× the PyTorch RTF of the same session.

**Graph size:**

| Model | ONNX FP32 | ONNX INT8 | INT8 / FP32 |
|---|---:|---:|---:|
| Piper_no_VITS2_cpn | 63.5 MB | 43.3 MB | 0.681 |
| Piper-ParIR | 61.6 MB | 41.3 MB | 0.671 |
| Piper-SeqIR | 61.3 MB | 41.1 MB | 0.670 |

**Known defect of this round (found 2026-10-22): the flow's weight norm was not folded before export.**
- **What went wrong:**
  - The export removed weight norm only in the decoder, as `harvard/synthesize.py` does.
  - EdgeTTS's flow coupling layers (`WN`) keep it. In the graphs, their 32 convolution weights are therefore recomputed at every run (weight_g · weight_v / ‖weight_v‖, the 32 `ReduceL2` nodes).
  - The quantizer could not treat those weights as constants. It quantized them **at run time, per tensor, as UINT8**, instead of the declared static per-channel INT8.
  - Only 8 of the 40 flow convolutions follow the declared recipe.
  - The Q05 exporter folds the flow's weight norm when the flow module provides `remove_weight_norm()`. EdgeTTS's flow block does not; only its inner `WN` layers do.
- **Consequences:**
  - **Sizes:** the INT8 graphs still store 27.8 MB of FP32 flow weights. The sizes above therefore understate what this PTQ scope can reach; for comparison, the historical INT8 graphs are 22–24 MB.
  - **Speed:** the INT8 RTF includes the per-run weight-norm and weight-quantization work.
  - **Quality:** the measured INT8 results are those of these graphs. Whether SeqIR's UTMOSv2 loss comes from the per-tensor flow weights is not known.
  - **What stays valid:** the FP32 results, the parity results and the decoder speed analysis are not affected. Both FP32 and INT8 graphs carry the same unfolded flow, and the decoder is FP32 in both.
- **Correction:** export again with `remove_weight_norm()` on every flow coupling layer, then quantize, evaluate and time again. This has not been run yet; it is prepared in `docs/plan-onnx-ptq-v2.md`.

### Historical PTQ and QAT (Gen 1)

These records are kept for transparency. They differ from the Gen 2 round in ways that prevent a direct comparison:

| | Historical PTQ (Sept 2026) | Gen 2 round (2026-10-21) |
|---|---|---|
| Graphs | benchmark graphs. Baseline and SEQ are linked to their checkpoints; the MRF graph is **not** (0/478 named weights equal checkpoint 1386) | exported from the selected checkpoints, hash in the graph metadata |
| Calibration | 60 random sentences, 57 train and **3 test** (J-E025) | 60 train-only sentences |
| Scope | baseline/MRF: Conv in enc_p/dp/flow (133 QLinearConv); **SEQ also MatMul** (+40 QLinearMatMul) | the same Conv scope for all three (109 QLinearConv) |
| Scope selection | sweep with a faulty evaluator (J-E026) | fixed in the protocol before the runs |
| Flow weight norm | not recorded; the 22–24 MB INT8 sizes suggest the flow weights were stored as INT8 | **not folded** (see the defect above) |
| Evaluation set | LJ-500, MRF on its own sentence set | Harvard-720, matched noise |
| INT8 file size | baseline 24.1 MB, MRF 22.2 MB, SEQ 22.0 MB | 41.1–43.3 MB |

**Historical ONNX results** (LJ-500, ORT CPU 2 threads, `historical/onnx_historical/`):

| Model (set) | FP32 RTF / UTMOSv2 / WER | INT8 RTF / UTMOSv2 / WER |
|---|---|---|
| baseline 1489 (baseline set) | 0.0388 / 3.132 / 0.1230 | 0.0345 / 3.121 / 0.1135 |
| MRF, unlinked graph (MRF set) | 0.0416 / 3.537 / 0.1043 | 0.0371 / 3.493 / 0.1030 (rerun 3.490 / 0.1124) |
| SEQ 1442, wider INT8 scope (baseline set) | 0.0346 / 3.571 / 0.1064 | 0.0289 / 3.586 / 0.1081 |

**Historical QAT** (`historical/qat/`, LJ-500):
- baseline QAT at epoch 144: UTMOSv2 3.549–3.569, WER 0.079–0.082;
- MRF QAT at epoch 97: UTMOSv2 3.630–3.650, WER 0.098–0.108.
- These gains cannot be attributed to quantization: QAT added 97–144 training epochs, and its RTF is PyTorch with fake-quant ops. There is no QAT record for SEQ.

**Historical scope sweep** (`historical/ptq_sweep/`): on the baseline, quantizing flow + enc_p + dp gave the lowest WER of the swept scopes (0.1215, 23.0 MB). The evaluator had the sample-rate bug, so this is a ranking only.

## Seven-model comparison

All seven models on Harvard-720 (720 sentences, 5,742 reference words), with the same synthesis and scoring protocol (`docs/evaluation.md`).
- Every value is copied from the records named under the table; nothing was re-measured for it.
- **Bold** marks the best value in a row. For RTF, the best is marked within each group of timings only (note 3).

| Metric | Piper_no_VITS2_cpn | Piper-ParIR | Piper-SeqIR | Baseline ResBlock2 | Parallel-IR (MRF) | Sequential-IR (SEQ) | Piper (external) ⁶ |
|---|---:|---:|---:|---:|---:|---:|---:|
| ***Model*** | | | | | | | |
| Decoder | ResBlock2 | Parallel-IR | Sequential-IR | ResBlock2 | Parallel-IR | Sequential-IR | ResBlock2 |
| Backbone / recipe | Config A, no VITS2 | Config A, no VITS2 | Config A, no VITS2 | VITS2 baseline | VITS2 baseline | VITS2 baseline | published Piper |
| Selected epoch | 1403 | 1486 | 1455 | 1489 | 1386 | 1442 | 999 |
| Validation `val_loss_mel` ¹ | 19.7719 | **19.0069** | 19.0332 | 19.7882 | (18.7681) | (19.0136) | — |
| ***Quality, PyTorch*** | | | | | | | |
| UTMOSv2, mean (predicted) ² | 3.517 | 3.530 | 3.493 | 3.026 | **3.568** | 3.544 | 3.451 |
| UTMOSv2, SD | 0.250 | 0.251 | 0.267 | 0.303 | 0.236 | 0.246 | 0.259 |
| Macro WER | **0.0612** | 0.0726 | 0.0696 | 0.1476 | 0.1073 | 0.1222 | 0.1747 |
| Corpus WER | **0.0596** | 0.0707 | 0.0665 | 0.1437 | 0.1033 | 0.1186 | 0.1696 |
| Substitutions / deletions / insertions | **283** / **46** / **13** | 321 / 71 / 14 | 292 / 76 / 14 | 681 / 101 / 43 | 467 / 109 / 17 | 531 / 111 / 39 | 812 / 112 / 50 |
| Sentences with WER 0 (of 720) | **502** | 464 | 476 | 318 | 381 | 351 | 228 |
| Scoring session | 2026-10-10 | 2026-10-16 | 2026-10-21 | historical | historical | historical | historical |
| ***Speed, PyTorch FP32, CPU, 12 threads*** ³ | | | | | | | |
| Mean RTF | 0.0510 | 0.0392 | **0.0340** | 0.0418 | 0.0339 | **0.0308** | 0.0392 |
| Median RTF | 0.0490 | 0.0374 | **0.0320** | 0.0412 | 0.0329 | **0.0296** | 0.0379 |
| Mean RTF relative to the ResBlock2 model of the same group | — | −23.2% | −33.3% | — | −18.9% | −26.2% | (−6.2%) |
| Timing session | 2026-10-21 | 2026-10-21 | 2026-10-21 | historical | historical | historical | historical |
| ***Deployment, ONNX Runtime CPU*** ⁴ | | | | | | | |
| ONNX FP32: UTMOSv2 / macro WER | 3.508 / **0.0612** | **3.524** / 0.0726 | 3.495 / 0.0696 | — | — | — | — |
| ONNX INT8: UTMOSv2 / macro WER | 3.503 / **0.0620** | **3.524** / 0.0772 | 3.459 / 0.0739 | — | — | — | — |
| ORT RTF, 2 threads: FP32 / INT8 | 0.0420 / 0.0376 | 0.0397 / 0.0341 | **0.0323 / 0.0269** | — | — | — | — |
| ORT RTF, 12 threads: FP32 / INT8 | **0.0292 / 0.0277** | 0.0397 / 0.0366 | 0.0312 / 0.0297 | — | — | — | — |
| ***Size*** ⁵ | | | | | | | |
| Parameters (Q05 definition) | 16,425,776 | 15,918,032 | **15,870,768** | 17,620,784 | 17,113,040 | 17,065,776 | 16,425,776 |
| Parameters used at inference | 15,663,771 | 15,156,027 | **15,108,763** | 16,858,779 | 16,351,035 | 16,303,771 | 15,663,771 |
| Decoder parameters | 1,664,768 | 1,157,024 | **1,109,760** | 1,664,768 | 1,157,024 | **1,109,760** | 1,664,768 |
| FP32 weights | 65.7 MB | 63.7 MB | **63.5 MB** | 70.5 MB | 68.5 MB | 68.3 MB | 65.7 MB |
| ONNX FP32 file | 63.5 MB | 61.6 MB | **61.3 MB** | 68.5 MB | 66.5 MB | 66.3 MB | 63.5 MB |
| ONNX INT8 file | 43.3 MB | 41.3 MB | 41.1 MB | 24.1 MB | (22.2 MB) | (**22.0 MB**) | — |
| Training checkpoint | 845.9 MB | 843.4 MB | **842.7 MB** | 867.0 MB | 864.5 MB | 863.9 MB | 845.9 MB |

Notes:
1. **Validation loss** is comparable only between the three Config A models and the baseline. Those four use the same 100 canonical validation utterances.
   - MRF was trained on a different (reconstructed) split.
   - SEQ's validation set was not verified to be the same.
   - Their values, taken from the checkpoint file names, are in parentheses.
   - The published Piper checkpoint has no validation record here.
2. **UTMOSv2** is a predicted score, not human MOS. Its random cropping is not seeded, so a model's mean moves by about ±0.01–0.02 between scoring sessions.
   - In the 2026-10-21 session, the stored WAVs of Piper_no_VITS2_cpn and ParIR were rescored next to SeqIR. They scored 3.508 and 3.526, against SeqIR's 3.493.
   - The WER values reproduce exactly between sessions.
   - Paired tests with confidence intervals are in each model's README.
3. **PyTorch RTF** comes from two groups of measurements that cannot be compared with each other.
   - The three Config A models were timed back to back in one session on 2026-10-21.
   - The four historical values come from their original runs. These were not shown to be one session, and Piper ran on a different PyTorch version from the internal models, hence the parentheses.
   - The 2026-10-21 control ran 1.21× slower than in its own 2026-10-10 session. On 2026-10-10 the baseline itself ran 8.3% slower than its historical value.
   - So compare RTF only within a group. The "relative" row does this, using the ResBlock2 model of the same group as the reference.
4. **Deployment rows:** the three Config A models only, from `piper-ir-onnx-ptq/`, Harvard-720. Round-A ORT timings come from one idle session.
   - The historical ONNX graphs were evaluated on LJ-500 only (see the historical ONNX table above), so their Harvard-720 cells are empty.
   - The Config A INT8 results carry the flow weight-norm defect described above.
5. **Size:** definitions and the record are in [Model size](#model-size) (`model_parameters.json`; MB = 10^6 bytes).
   - The ONNX file holds only the inference graph, so it is smaller than the FP32 weights.
   - **Historical INT8 files** are the benchmark graphs. MRF's graph is not linked to checkpoint 1386, and SEQ's has the wider scope, hence the parentheses.
   - The training checkpoint also holds the posterior encoder, discriminators and optimizer state.

6. **Piper (external)** was fed **en-us** phonemes, as every system here, but its released config names espeak voice "en" (British).
   - An exploratory check (`frontend-check/`) found its ASR errors concentrated on the words whose en and en-us phonemes differ: 34.2% against 7.5% on other words, while Piper_no_VITS2_cpn has 5.8% and 5.5%.
   - Until the prepared causal test (`docs/plan-frontend-causal-test.md`) is run, read every comparison with this column as unfair to Piper.

Sources:
- quality: `piper-seqir-no-vits2-cpn/comparison.json`, which has summaries of all seven result files;
- PyTorch RTF: `piper-ir-timing-session/` and the historical result files;
- deployment: `piper-ir-onnx-ptq/`;
- historical INT8 file sizes: `docs/evidence/journal-evidence-package-2026-10-04/evidence/E_onnx_ptq/onnx_inventory_inspected_now.json`;
- epochs and validation: each model's `selected_checkpoint.json` or `models/manifest.json`;
- size: `model_parameters.json` and the export and quantization records.

Every model has one training run and one synthesis draw per sentence.

## Model size

`model_parameters.json` records parameter counts and sizes for all seven Harvard-720 models.
- It was written by `scripts/evaluation/count_parameters.py`, which reads the checkpoints with `weights_only=True` without building or running a model.
- The three new checkpoints were counted on 2026-10-21.
- The four historical checkpoints were recounted on 2026-10-26. Each file matched its SHA-256 in `models/manifest.json`, and each recount equals the total in its stored Q05 report. File and ONNX sizes still come from `models/manifest.json`.
- The ONNX file sizes of the new models come from `piper-ir-onnx-ptq/*/export_record.json` and `quantize_record.json`.

| Model | Parameters (Q05 definition) | Used at inference | Decoder | FP32 weights | ONNX FP32 file | ONNX INT8 file | Training checkpoint |
|---|---:|---:|---:|---:|---:|---:|---:|
| Piper_no_VITS2_cpn (ResBlock2) | 16,425,776 | 15,663,771 | 1,664,768 | 65.7 MB | 63.5 MB | 43.3 MB | 845.9 MB |
| Piper_ParIR_no_VITS2_cpn | 15,918,032 | 15,156,027 | 1,157,024 | 63.7 MB | 61.6 MB | 41.3 MB | 843.4 MB |
| Piper_SeqIR_no_VITS2_cpn | 15,870,768 | 15,108,763 | 1,109,760 | 63.5 MB | 61.3 MB | 41.1 MB | 842.7 MB |
| HiFi-GAN ResBlock2 (baseline, with VITS2) | 17,620,784 | 16,858,779 | 1,664,768 | 70.5 MB | 68.5 MB | 24.1 MB | 867.0 MB |
| Parallel-IR (MRF) | 17,113,040 | 16,351,035 | 1,157,024 | 68.5 MB | 66.5 MB | (22.2 MB) | 864.5 MB |
| Sequential-IR (SEQ) | 17,065,776 | 16,303,771 | 1,109,760 | 68.3 MB | 66.3 MB | (22.0 MB) | 863.9 MB |
| Piper external checkpoint | 16,425,776 | 15,663,771 | 1,664,768 | 65.7 MB | 63.5 MB | — | 845.9 MB |

How to read the columns (MB = 10^6 bytes):
- **Parameters (Q05 definition):**
  - generator parameters without the posterior encoder `enc_q`, which is used only in training;
  - counted in weight-norm form, with buffers excluded;
  - the same definition as the Q05 export reports, so all seven rows are comparable.
  - It still includes 762,005 parameters of the stochastic duration predictor that inference does not use: the training-only posterior path `dp.post_*` (640,248) and the flow dropped in reverse mode (121,757).
- **Used at inference:** the Q05 count minus those 762,005. All seven models have the same stochastic duration predictor, so the subtraction is the same for each.
- **FP32 weights:** 4 bytes × the Q05 count. This is computed, not a file size.
- **ONNX FP32 file:** the measured size of the exported graph. It holds only the inference path, so it is smaller than the FP32-weights column.
  - New models: the 2026-10-21 export.
  - Historical models: the Q05 exports.
- **ONNX INT8 file:**
  - New models: the 2026-10-21 PTQ. They still store the flow in FP32 because its weight norm was not folded (see [the defect](#onnx-export-and-int8-ptq)).
  - Historical models: the benchmark INT8 graphs, with the caveats in note 5 above.
- **Training checkpoint:** the full Lightning state: generator, posterior encoder, discriminators and optimizer states. It is not a deployable model size.

What the counts show:
- **Shared part:** in the three new models, everything outside the decoder is identical, 21,999,024 parameters including `enc_q`.
- **Decoder:** ParIR and SeqIR have 30.5% and 33.3% fewer decoder parameters than ResBlock2. Overall that is 3.1% and 3.4% fewer, because the decoder is a small part of the generator.
- **Piper_no_VITS2_cpn and the external Piper checkpoint:** both have exactly 16,425,776 parameters. Both use the Piper (Config A) architecture.
- **Same decoders in both generations:** each decoder has the same parameter count with and without VITS2: ResBlock2 1,664,768, Parallel-IR 1,157,024 and Sequential-IR 1,109,760. So the Gen 1 and Gen 2 decoder comparisons involve the same decoder sizes.
- **What VITS2 adds:** every VITS2 model has 1,195,008 more generator parameters than its Config A counterpart. All of them are in the flow (8,297,088 against 7,102,080); the text encoder, duration predictor and decoder have the same counts.
- **Training only:**
  - the MPD (46,747,132), for the IR models the MRD (280,902), and for the VITS2 models the duration discriminator `model_d_dur` (555,841) are discriminators;
  - they are not part of the synthesis model.

Public redistribution of sentence collections, result records and generated audio remains an author decision. Local inclusion does not assert those rights.
