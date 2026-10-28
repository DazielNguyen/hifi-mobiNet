# ONNX/PTQ round 2: flow weight norm folded

*Tiếng Việt: [README.vi.md](README.vi.md)*

Round 2 repeats the ONNX export, static INT8 PTQ and Harvard-720 evaluation of the three Config A models with the one defect of round 1 fixed:
- Piper_no_VITS2_cpn (ResBlock2);
- Piper-ParIR;
- Piper-SeqIR.

Round 1 (`../piper-ir-onnx-ptq/`, 2026-10-21) left the flow's weight norm unfolded. In those graphs 32 of the 40 flow convolutions were quantized at run time, per tensor, as UINT8, instead of the declared static per-channel INT8.

Everything else is the round-1 protocol unchanged (`docs/evaluation-onnx-ptq.md`, plan `docs/plan-onnx-ptq-v2.md`):
- the same models, checkpoints, PTQ recipe and scope;
- the same train-only calibration set, matched-noise quality synthesis and scorer;
- the same timing design.

All numbers are **new measurements** of 2026-10-26.
- The round ran 03:06–07:09 +07 on repository commit `f495672`, with `scripts/evaluation/onnx/run_round.sh`.
- Every check passed.
- The only exception is the round-to-round comparison in section 4. It is a new analysis of stored per-sentence results.

## 1. What changed

- **Fold before export.** `fold_flow_weight_norm` calls `WN.remove_weight_norm()` on each of the 4 flow coupling layers before tracing.
  - The new gate `flow_weight_norm_fold_changes_only_rounding` compares the model output before and after the fold on 3 sentences. It found **no difference at all** (NRMSE 0, max abs error 0).
- **No weight norm left in the PTQ scope.** The new gate `no_weight_norm_ops_in_ptq_scope` requires no `ReduceL2` under `/enc_p/`, `/dp/` or `/flow/`. The graphs have none anywhere (`weight_norm_ops_in_graph` is empty).
- **Only constant weights.** The new gate `every_qlinearconv_weight_constant_per_channel_int8` requires every QLinearConv weight to be a constant per-channel INT8 initializer. All 109 per model are.
- **Not changed:**
  - **PTQ recipe:** QOperator, QInt8 per-channel weights, QUInt8 activations, MinMax calibration;
  - **scope:** `Conv` under enc_p, dp and flow, with the decoder kept in FP32;
  - **calibration:** the 60 train-only sentences;
  - **evaluation:** matched-noise synthesis, the scorer, and the timing rounds A and B at 2 and 12 threads.

## 2. Export and PTQ checks

| | Piper_no_VITS2_cpn | Piper-ParIR | Piper-SeqIR |
|---|---:|---:|---:|
| FP32 graph (SHA-256) | `b4a95a82…` | `506e76dc…` | `a8d18a9e…` |
| FP32 size | 63.5 MB | 61.5 MB | 61.3 MB |
| Flow `WN` layers folded; output change | 4; none | 4; none | 4; none |
| `ReduceL2` (weight norm) in the graph | 0 | 0 | 0 |
| Parity gate, 13 cases: max NRMSE | 8.71e-5 | 8.77e-5 | 7.86e-5 |
| Harvard-720 parity, FP32: passed | 720 / 720 | 720 / 720 | 720 / 720 |
| Harvard-720 parity, FP32: max NRMSE / max abs error | 3.3e-4 / 8.6e-4 | 6.0e-4 / 1.3e-3 | 1.0e-3 / 2.0e-3 |
| INT8 graph (SHA-256) | `f54bc09c…` | `a94703b7…` | `ac2353c1…` |
| **INT8 size (ratio to FP32)** | **22.4 MB (0.354)** | **20.5 MB (0.333)** | **20.3 MB (0.331)** |
| Round-1 INT8 size, for reference | 43.3 MB | 41.3 MB | 41.1 MB |
| QLinearConv (enc_p / dp / flow) | 37 / 32 / 40 | 37 / 32 / 40 | 37 / 32 / 40 |
| of which constant per-channel INT8 weights | 109 / 109 | 109 / 109 | 109 / 109 |
| Decoder initializers byte-identical to FP32 | 45 / 45 | 101 / 101 | 77 / 77 |
| Second quantization byte-identical | yes | yes | yes |

The parity thresholds are NRMSE ≤ 1e-3 and max abs error ≤ 5e-3. SeqIR's largest Harvard-720 NRMSE, 0.000996, is just inside its threshold, as in round 1 (0.00096).

Further checks:
- **PyTorch determinism.** In every quality pass the PyTorch reference WAVs were byte-identical to the stored PyTorch evaluations: 720/720 for each model.
- **INT8 waveform parity is not meaningful.** The quantized duration predictor changes the frame counts slightly. Only 116, 103 and 108 of 720 INT8 outputs have the PyTorch length, the same counts as in round 1. The quality of the INT8 graphs is judged by the scores in section 3.
- **Size.** The INT8 graphs fall in the 20–25 MB range the plan expected, because the 27.8 MB of FP32 flow weights became INT8. They are about as small as the historical INT8 graphs (22.0–24.1 MB).

## 3. Harvard-720 quality

| Model | Version | UTMOSv2 (predicted) | Macro WER | Substitutions / deletions / insertions | Sentences with WER 0 |
|---|---|---:|---:|---:|---:|
| Piper_no_VITS2_cpn | PyTorch (stored) | 3.517 | 0.0612 | 283 / 46 / 13 | 502 |
| | ONNX FP32 | 3.517 | 0.0612 | 283 / 46 / 13 | 502 |
| | **ONNX INT8** | **3.506** | **0.0584** | 260 / 54 / 11 | 505 |
| Piper-ParIR | PyTorch (stored) | 3.530 | 0.0726 | 321 / 71 / 14 | 464 |
| | ONNX FP32 | 3.526 | 0.0726 | 321 / 71 / 14 | 464 |
| | **ONNX INT8** | **3.520** | **0.0764** | 335 / 69 / 19 | 457 |
| Piper-SeqIR | PyTorch (stored) | 3.493 | 0.0696 | 292 / 76 / 14 | 476 |
| | ONNX FP32 | 3.497 | 0.0696 | 292 / 76 / 14 | 476 |
| | **ONNX INT8** | **3.474** | **0.0687** | 292 / 75 / 11 | 484 |

Paired by sentence (`*_int8_vs_fp32.json`, `*_fp32_vs_pytorch.json`):
- The difference is candidate − reference, and the CIs are paired bootstrap 95% intervals.
- Holm is over the two tests of each file.
- FP32 and INT8 were scored in the same session. The stored PyTorch scores come from earlier sessions.

| Comparison | ΔUTMOSv2 [95% CI] (Wilcoxon p) | ΔWER [95% CI] (Wilcoxon p) | WER better / worse |
|---|---|---|---|
| no_VITS2: INT8 vs FP32 | −0.011 [−0.031, +0.008] (0.27) | −0.003 [−0.009, +0.004] (0.19) | 78 / 67 |
| ParIR: INT8 vs FP32 | −0.007 [−0.025, +0.011] (0.50) | +0.004 [−0.004, +0.012] (0.49) | 91 / 99 |
| SeqIR: INT8 vs FP32 | −0.023 [−0.043, −0.002] (0.058; Holm 0.12) | −0.001 [−0.008, +0.006] (0.80) | 83 / 73 |
| no_VITS2: FP32 vs PyTorch | +0.000 [−0.012, +0.012] (0.91) | 0 on every sentence | 0 / 0 |
| ParIR: FP32 vs PyTorch | −0.004 [−0.014, +0.007] (0.64) | 0 on every sentence | 0 / 0 |
| SeqIR: FP32 vs PyTorch | +0.004 [−0.009, +0.016] (0.47) | 0 on every sentence | 0 / 0 |

`*_int8_vs_historical.json` compares each INT8 graph with the four stored systems (Holm over 8 tests). Every INT8 graph has lower WER than all four:
- −3.1 to −4.9 points against MRF;
- −4.6 to −6.4 points against SEQ;
- −7.1 to −8.9 points against baseline;
- −9.8 to −11.6 points against Piper. Read these with the frontend caveat in `../README.md`, note 6.

For UTMOSv2, every INT8 graph is above baseline (+0.45 to +0.49) and below MRF (−0.05 to −0.09) and SEQ (−0.03 to −0.07). Every one of these UTMOSv2 CIs excludes 0.

## 4. Round 2 against round 1 (INT8)

`*_int8_vs_round1_int8.json` pairs each round-2 INT8 result with the stored round-1 INT8 result of the same model. It was written with `harvard/compare.py --no-historical --metrics utmosv2,wer` after the round.
- WER compares the transcripts of two different graphs.
- UTMOSv2 also carries the scorer's session-to-session variation (±0.01–0.02), because the two rounds were scored in different sessions.

| Model | ΔUTMOSv2, round 2 − round 1 [95% CI] (p) | ΔWER [95% CI] (p) | WER better / worse |
|---|---|---|---|
| Piper_no_VITS2_cpn | +0.002 [−0.013, +0.017] (0.54) | −0.004 [−0.009, +0.001] (0.26) | 44 / 38 |
| Piper-ParIR | −0.004 [−0.019, +0.010] (0.72) | −0.001 [−0.006, +0.004] (0.84) | 50 / 54 |
| Piper-SeqIR | +0.015 [−0.001, +0.031] (0.052) | −0.005 [−0.011, +0.000] (0.13) | 59 / 35 |

No difference is significant. SeqIR's changes both point towards round 2 being better, by about as much as its round-1 INT8 loss shrank (−0.036 → −0.023 against FP32).

## 5. Speed (ONNX Runtime CPU, same session)

**Session:** an idle machine, 03:24–03:47; load average 0.94 at the start, no training, both GPUs idle.
- **Order:** a PyTorch control run opened the session. Then round A, and round B in the reverse order, for each thread setting.
- **Reading the table:** mean RTF of round A, with round B in brackets. "INT8 vs FP32" is the change of the mean in round A (round B).
- **Significance:** every paired INT8-vs-FP32 CI excludes 0 (`timing_summary.json`).

| Model | 2 threads: FP32 | 2 threads: INT8 | INT8 vs FP32 | 12 threads: FP32 | 12 threads: INT8 | INT8 vs FP32 |
|---|---:|---:|---:|---:|---:|---:|
| Piper_no_VITS2_cpn | 0.0323 (0.0321) | 0.0280 (0.0281) | −13.3% (−12.6%) | **0.0253** (0.0253) | **0.0226** (0.0226) | −10.6% (−10.8%) |
| Piper-ParIR | 0.0338 (0.0333) | 0.0306 (0.0289) | −9.5% (−13.2%) | 0.0318 (0.0321) | 0.0290 (0.0292) | −8.6% (−9.1%) |
| Piper-SeqIR | **0.0278** (0.0276) | **0.0238** (0.0232) | −14.5% (−15.8%) | 0.0267 (0.0282) | 0.0243 (0.0244) | −9.0% (−13.3%) |

**Against Piper_no_VITS2_cpn** (same precision and threads; round A, with round B in brackets; every round-A CI excludes 0):

| | 2 threads, FP32 | 2 threads, INT8 | 12 threads, FP32 | 12 threads, INT8 |
|---|---:|---:|---:|---:|
| Piper-ParIR | +4.6% (+3.6%) | +9.1% (+2.9%) | +25.6% (+27.1%) | +28.4% (+29.4%) |
| Piper-SeqIR | **−14.1% (−14.1%)** | **−15.3% (−17.3%)** | +5.4% (+11.4%) | +7.3% (+8.2%) |

**Controls:**
- **PyTorch control:** Piper_no_VITS2_cpn with 12 threads gave mean RTF 0.0388, against 0.0510 in the 2026-10-21 timing session (×0.76). Its WAVs were byte-identical (720/720).
  - The machine therefore ran about a quarter faster in this session.
  - **Compare RTF only within this session, not with round 1.**
- **ORT against PyTorch:** ORT FP32 with 12 threads ran Piper_no_VITS2_cpn at 0.65× the PyTorch RTF of the same session (0.59× in round 1).
- **Order effects:** round B / round A was 0.94–1.00 at 2 threads and 1.00–1.06 at 12 threads. Differences of a few percent between graphs are within that variation.

**What changed against round 1** (comparing within-session ratios only):
- **INT8 gain:** at 12 threads, INT8 now saves 9–13% against FP32, where round 1 saved 4–9%. At 2 threads the gain is similar (10–16%, against 10–17%). Round-1 INT8 graphs recomputed the flow weight norm and quantized the flow weights at every run.
- **Rankings stay the same:**
  - at 2 threads, SeqIR is fastest (−14% to −17% against ResBlock2);
  - at 12 threads, ResBlock2 is fastest, with SeqIR 5–11% and ParIR 26–29% slower;
  - the cause is unchanged: SnakeBeta's single-threaded `Sin` in ONNX Runtime 1.23, profiled in `../piper-ir-onnx-ptq/profile/`. The decoder is FP32 and identical in both rounds.
- **ParIR at 2 threads** is now slightly slower than ResBlock2 (+3% to +9%). Its 5–9% gain in round-1 round A had already failed to repeat in round B.

## 6. Reading the results

- **The fix worked as intended:**
  - every flow convolution is now static per-channel INT8;
  - the INT8 graphs are a third of the FP32 size (20.3–22.4 MB, against 41–43 MB in round 1);
  - INT8 is faster than FP32 for every model and thread setting.
- **The export stays lossless.** FP32 passes parity on all 720 sentences, the WER is identical to PyTorch on every sentence, and UTMOSv2 differs by less than 0.005.
- **INT8 quality:**
  - **WER:** no model shows a detectable change against its FP32 graph.
  - **Piper_no_VITS2_cpn INT8** has the lowest WER of all evaluated systems (0.0584). This is not a detectable gain over FP32 (CI includes 0).
  - **Piper_no_VITS2_cpn and ParIR:** no detectable UTMOSv2 change.
  - **SeqIR:**
    - UTMOSv2 is 0.023 lower than its FP32 graph. The bootstrap CI excludes 0, but the Wilcoxon test does not reach 0.05, before or after Holm. Treat it as a possible small loss, not an established one.
    - In round 1 the loss was 0.036 and significant after Holm. The round-to-round comparison (section 4) does not show a significant change. So the per-tensor flow weights may explain part of the loss at most; this round does not establish it.
- **Speed:** the decoder ranking still depends on the runtime and the thread count, as in round 1.

What this does **not** show:
- **Edge-device speed.** This is a workstation CPU under WSL, one timing pass per sentence, no warm-up, two orders.
- **Other PTQ scopes or methods.** The decoder was not quantized, MinMax calibration was used on 60 sentences, and QAT was not tried.
- **That SeqIR's remaining INT8 change is specific to its decoder.** The quantized parts (enc_p, dp, flow) have the same architecture in all three models but different trained weights.
- **Significance across training seeds,** or human MOS.

## 7. Files

Per model folder (`piper-no-vits2-cpn/`, `piper-parir-no-vits2-cpn/`, `piper-seqir-no-vits2-cpn/`):

| File | Content | Class |
|---|---|---|
| `export_record.json` | FP32 graph hashes and structure, the flow fold and its check, the `ReduceL2` count, wrapper equality, int64 cast, the 13 parity cases, native-random checks | new measurement record |
| `quantize_record.json` | Calibration check and IDs, recipe and scope, INT8 structure and QLinearConv weight kinds, decoder-initializer check, reproducibility, sizes | new measurement record |
| `harvard720_{fp32,int8}_synth_record.json` | Matched-noise synthesis: graph hash, per-sentence WAV SHA-256, PyTorch determinism and parity. Timing fields are not for use | new measurement |
| `harvard_<model>_{fp32,int8}_results.json` | Per-sentence UTMOSv2 / Whisper / WER, historical schema. `rtf` is not for use | new measurement |
| `{fp32,int8}_score_record.json` | Scorer identities and environment | new measurement record |
| `*_int8_vs_fp32.json`, `*_fp32_vs_pytorch.json`, `*_int8_vs_historical.json` | Paired comparisons (UTMOSv2, WER) | new analysis |
| `*_int8_vs_round1_int8.json` | Round 2 against round 1 INT8, paired (section 4) | new analysis of stored results |

`timing/`:

| File | Content | Class |
|---|---|---|
| `t{2,12}_{A,B}_<model>_{fp32,int8}.json` (24 files) | Per-sentence ORT timings of the deployment graphs | new measurement |
| `pytorch_ctrl_synth_record.json` | The PyTorch control of the session | new control |
| `timing_summary.json` | Means, order check, paired INT8-vs-FP32 and model-vs-control tests | new analysis |
| `machine_state_before.txt`, `machine_state_after.txt` | Load, GPU use and top processes | session record |

The graphs (`*.onnx`), WAVs and the round log stay outside Git in `training_output/onnx-ptq-v2/`. Their hashes are in the records.

## 8. Reproduction

From the training clone root, with the scorer environment (`SPY`: torch 2.13.0, onnx 1.22.0, onnxruntime 1.23.2) and the training venv (`TPY`):

```sh
SPY=<scorer python> TPY=<training venv python> OUT=training_output/onnx-ptq-v2 \
  RESULTS=<repository>/results/piper-ir-onnx-ptq-v2 \
  bash <repository>/scripts/evaluation/onnx/run_round.sh
```

`OUT` must be a new folder. An existing one resumes that round after its last finished stage. The round-to-round comparison of section 4 ran afterwards, for each model:

```sh
$SPY <repository>/scripts/evaluation/harvard/compare.py \
  --candidate results/piper-ir-onnx-ptq-v2/<model>/harvard_<id>_int8_results.json --label <id>_int8_round2 \
  --extra-reference <id>_int8_round1=results/piper-ir-onnx-ptq/<model>/harvard_<id>_int8_results.json \
  --no-historical --metrics utmosv2,wer --output results/piper-ir-onnx-ptq-v2/<model>/<model>_int8_vs_round1_int8.json
```

The tests are in `tests/test_onnx_ptq.py`; all 13 passed before the round.

**Expected outputs:**
- Export and quantization are deterministic, so the graph hashes should match.
- Synthesis is deterministic, so the WAV hashes should match.
- UTMOSv2 varies between scoring sessions.
- RTF varies between sessions; only within-session ratios are meaningful.
