# Piper-IR models as ONNX FP32 and INT8 (static PTQ): export, Harvard-720 quality and speed

*Tiếng Việt: [README.vi.md](README.vi.md)*

The three Config A models were exported to ONNX, quantized to INT8 with the historical PTQ
recipe, and evaluated on Harvard-720:
- Piper_no_VITS2_cpn (ResBlock2);
- Piper-ParIR;
- Piper-SeqIR.

The protocol, `docs/evaluation-onnx-ptq.md`, fixed the thresholds, the PTQ scope and the
calibration set before any graph existed. All numbers are **new measurements** of
2026-10-21, except the PyTorch results they are compared with. The round ran 16:35–21:47 +07
(`scripts/evaluation/onnx/run_round.sh`) and every check passed.

## 1. What was done

- **FP32 export.** The interface is the same as the Q05 graphs: `input`, `input_lengths`, `scales` → `output` [batch, 1, samples], opset 18. Two export obstacles in EdgeTTS were fixed at their source. Neither changes any number; the PyTorch output was bit-identical before and after, which was checked.
  - `infer()` takes the frame count with `.item()`, which tracing would freeze. The export wrapper keeps it as a tensor.
  - The flow's `WN` keeps its channel count in an int32 buffer used as a Slice bound. That gives an ONNX Slice with int32 `ends` and int64 `starts`, which the spec does not allow. The four buffers are cast to int64 (same value 192).
- **PTQ.** The recipe of the historical baseline/MRF INT8 graphs:
  - `quantize_static`, QOperator, weights QInt8 per channel, activations QUInt8, MinMax calibration;
  - `Conv` nodes under `/enc_p/`, `/dp/` and `/flow/` (109 per model); the decoder stays FP32.
  - Calibration: the 60-sentence train-only set designed on 2026-10-04 and never used before. All 60 are canonical train; 0 are in validation or test.
- **Quality.** Each FP32 and INT8 graph synthesized Harvard-720 through a *matched-noise* copy: the two random nodes were replaced by the PyTorch noise of each sentence (seed 1234). PyTorch, FP32-ONNX and INT8 therefore share the same random draws.
  - Scoring used `harvard/score.py` unchanged (UTMOSv2, Whisper-small, JiWER). All six sets were scored in one session.
- **Speed.** One idle-machine session, 16:55–17:24.
  - The deployment graphs were timed with native random noise, ORT CPU, intra-op 2 and 12 threads.
  - Each thread setting ran round A, then round B in reverse order.
  - A PyTorch control run of Piper_no_VITS2_cpn opened the session.
- **Profile (follow-up, 23:17–23:22).** ORT per-node profiles and single-op benchmarks were run to explain the thread-dependent ranking of the decoders (section 5).

## 2. Export and PTQ checks

| | Piper_no_VITS2_cpn | Piper-ParIR | Piper-SeqIR |
|---|---:|---:|---:|
| FP32 graph (SHA-256) | `9cd2f70a…` | `d8a70911…` | `2c04e820…` |
| FP32 size | 63.5 MB | 61.6 MB | 61.3 MB |
| Parity gate, 13 cases: max NRMSE | 1.07e-4 | 8.90e-5 | 8.59e-5 |
| Harvard-720 parity, FP32: passed | 720 / 720 | 720 / 720 | 720 / 720 |
| Harvard-720 parity, FP32: max NRMSE / max abs error | 3.3e-4 / 8.7e-4 | 5.9e-4 / 1.3e-3 | 9.6e-4 / 2.0e-3 |
| INT8 graph (SHA-256) | `7d4b35e4…` | `8ed64208…` | `b0a7ffc3…` |
| INT8 size (ratio to FP32) | 43.3 MB (0.681) | 41.3 MB (0.671) | 41.1 MB (0.670) |
| QLinearConv (enc_p / dp / flow) | 37 / 32 / 40 | 37 / 32 / 40 | 37 / 32 / 40 |
| Decoder initializers byte-identical to FP32 | 45 / 45 | 101 / 101 | 77 / 77 |
| Second quantization byte-identical | yes | yes | yes |

The parity thresholds are NRMSE ≤ 1e-3 and max abs error ≤ 5e-3.

Further checks:
- **SeqIR reproducibility.** The SeqIR FP32 and INT8 graphs have the same SHA-256 as a trial run of the same scripts three hours earlier.
- **PyTorch determinism.** In every quality synthesis pass, the PyTorch reference WAVs were byte-identical to the stored PyTorch evaluations: 720/720 for each model. For SeqIR this is the second synthesis that was still missing.
- **INT8 waveform parity is not meaningful.** The quantized duration predictor changes the frame counts slightly, so only 116, 103 and 108 of 720 INT8 outputs have the PyTorch length. The quality of the INT8 graphs is judged by the scores in section 3.

## 3. Harvard-720 quality

| Model | Version | UTMOSv2 (predicted) | Macro WER | Substitutions / deletions / insertions | Sentences with WER 0 |
|---|---|---:|---:|---:|---:|
| Piper_no_VITS2_cpn | PyTorch (stored) | 3.517 | 0.0612 | 283 / 46 / 13 | 502 |
| | ONNX FP32 | 3.508 | 0.0612 | 283 / 46 / 13 | 502 |
| | **ONNX INT8** | **3.503** | **0.0620** | 268 / 59 / 13 | 501 |
| Piper-ParIR | PyTorch (stored) | 3.530 | 0.0726 | 321 / 71 / 14 | 464 |
| | ONNX FP32 | 3.524 | 0.0726 | 321 / 71 / 14 | 464 |
| | **ONNX INT8** | **3.524** | **0.0772** | 327 / 81 / 18 | 456 |
| Piper-SeqIR | PyTorch (stored) | 3.493 | 0.0696 | 292 / 76 / 14 | 476 |
| | ONNX FP32 | 3.495 | 0.0696 | 292 / 76 / 14 | 476 |
| | **ONNX INT8** | **3.459** | **0.0739** | 316 / 78 / 12 | 469 |

Paired by sentence (`*_int8_vs_fp32.json`, `*_fp32_vs_pytorch.json`):
- The difference is candidate − reference, and the CIs are paired bootstrap 95% intervals.
- Holm is over the two tests of each file.
- FP32 and INT8 were scored in the same session. The stored PyTorch scores come from earlier sessions.

| Comparison | ΔUTMOSv2 [95% CI] (Wilcoxon p) | ΔWER [95% CI] (Wilcoxon p) | WER better / worse |
|---|---|---|---|
| no_VITS2: INT8 vs FP32 | −0.005 [−0.023, +0.013] (0.74) | +0.001 [−0.006, +0.008] (1.00) | 77 / 73 |
| ParIR: INT8 vs FP32 | +0.000 [−0.018, +0.019] (0.51) | +0.005 [−0.003, +0.012] (0.25) | 91 / 94 |
| SeqIR: INT8 vs FP32 | **−0.036 [−0.056, −0.016] (0.0007; Holm 0.0013)** | +0.004 [−0.003, +0.011] (0.33) | 71 / 85 |
| no_VITS2: FP32 vs PyTorch | −0.009 [−0.021, +0.003] (0.15) | 0 on every sentence | 0 / 0 |
| ParIR: FP32 vs PyTorch | −0.007 [−0.018, +0.005] (0.34) | 0 on every sentence | 0 / 0 |
| SeqIR: FP32 vs PyTorch | +0.002 [−0.010, +0.014] (0.85) | 0 on every sentence | 0 / 0 |

`*_int8_vs_historical.json` compares each INT8 graph with the four stored systems (Holm over 8 tests). Every INT8 graph has lower WER than all four:
- −3.0 to −4.5 points against MRF;
- −4.5 to −6.0 points against SEQ;
- −7.0 to −8.6 points against baseline;
- −9.7 to −11.3 points against Piper.

## 4. Speed (ONNX Runtime CPU, same session)

Mean RTF in round A, with round B (reverse order) in brackets. "INT8 vs FP32" is the change of the round-A mean (paired; every CI excludes 0, `timing_summary.json`).

| Model | 2 threads: FP32 | 2 threads: INT8 | INT8 vs FP32 | 12 threads: FP32 | 12 threads: INT8 | INT8 vs FP32 |
|---|---:|---:|---:|---:|---:|---:|
| Piper_no_VITS2_cpn | 0.0420 (0.0439) | 0.0376 (0.0394) | −10.5% (−10.4%) | 0.0292 (0.0271) | 0.0277 (0.0247) | −4.9% (−9.1%) |
| Piper-ParIR | 0.0397 (0.0453) | 0.0341 (0.0399) | −13.9% (−12.0%) | 0.0397 (0.0362) | 0.0366 (0.0348) | −7.8% (−4.0%) |
| Piper-SeqIR | 0.0323 (0.0323) | 0.0269 (0.0270) | −16.8% (−16.5%) | 0.0312 (0.0319) | 0.0297 (0.0291) | −4.8% (−8.5%) |

**Against Piper_no_VITS2_cpn** (same precision and threads, round A, paired):

| | 2 threads, FP32 | 2 threads, INT8 | 12 threads, FP32 | 12 threads, INT8 |
|---|---:|---:|---:|---:|
| Piper-ParIR | −5.5% | −9.1% | **+36.1%** | **+31.9%** |
| Piper-SeqIR | **−23.1%** | **−28.5%** | +6.9% | +7.1% |

**Controls:**
- **PyTorch control:** Piper_no_VITS2_cpn with 12 threads gave mean RTF 0.0495, against 0.0510 in the noon session (×0.97). Its WAVs were byte-identical (720/720).
- **ORT against PyTorch:** on the same model and session, ORT FP32 with 12 threads ran at 0.59× the PyTorch RTF.
- **Order effects:** round B / round A was 1.00–1.05 for most graphs at 2 threads, but 1.14–1.17 for ParIR. At 12 threads it was 0.89–1.02. Differences of a few percent between graphs are therefore within order and run-to-run variation.
  - In round B the SeqIR results and all 12-thread rankings are the same.
  - ParIR's small 2-thread gain is not: in round B, ParIR was slower than the control at 2 threads (FP32 0.0453 vs 0.0439, INT8 0.0399 vs 0.0394).

## 5. Why the decoders rank differently on ORT: profile

Run after the round on the same idle machine (23:17–23:22) with
`scripts/evaluation/onnx/profile_graphs.py` and `profile_ops.py`; records in `profile/`.
- **Graph profile.** ORT's per-node profiler on the six deployment graphs, at 2 and 12 threads, with ORT's default idle-thread spinning and with spinning off.
  - The input is 30 Harvard sentences (every 24th, about 75 s of audio), after one dropped warm-up run.
  - The profiler adds overhead, so compare the parts and the 2-vs-12 ratios, not the RTF values with section 4.
- **Op benchmark.** Single-op graphs on a tensor of the last decoder stage, [1, 32, 53,760], about 2.4 s of audio. Median of 50 runs.

**Where the FP32 graphs spend their kernel time** (ms over the 30 sentences, default spinning). "Conv" includes the fused convs.

| Model | Threads | Decoder total | of which Conv | of which ConvTranspose | of which Sin | Rest of the model |
|---|---:|---:|---:|---:|---:|---:|
| Piper_no_VITS2_cpn | 2 | 2,099 | 1,302 | 556 | — | 920 |
| | 12 | **1,466** | 698 | 658 | — | 990 |
| Piper-ParIR | 2 | 2,318 | 478 | 563 | 822 | 909 |
| | 12 | **2,264** | 302 | 650 | **1,069** | 895 |
| Piper-SeqIR | 2 | 1,693 | 308 | 555 | 561 | 903 |
| | 12 | **1,702** | 200 | 640 | **719** | 921 |

**Thread scaling of single ops** (median ms per call):

| Op | 1 thread | 2 threads | 4 threads | 12 threads | 12 threads, no spinning |
|---|---:|---:|---:|---:|---:|
| **Sin** | 2.13 | 2.27 | 2.33 | **3.13** | 2.39 |
| **Cos** | 2.30 | 2.31 | 2.43 | **3.08** | 2.25 |
| Mul | 0.28 | 0.15 | 0.08 | 0.04 | 0.14 |
| Add | 0.28 | 0.15 | 0.08 | 0.04 | 0.14 |
| Pow | 0.54 | 0.27 | 0.14 | 0.07 | 0.17 |
| LeakyRelu | 0.30 | 0.15 | 0.08 | 0.05 | 0.14 |
| Tanh | 0.45 | 0.22 | 0.11 | 0.07 | 0.17 |

**What this shows:**
1. **The decoder takes 60–80% of the kernel time** in every graph. The rest of the model (enc_p, dp, flow and glue ops) costs about the same in all three models, about 900 ms, as expected from the shared backbone.
2. **SnakeBeta's `Sin` does not use the thread pool.**
   - The IR decoders apply SnakeBeta, x + sin²(αx)/β, at every block. In the optimized graphs that is 22 `Sin` nodes for ParIR and 16 for SeqIR, all on full-resolution tensors.
   - In ONNX Runtime 1.23.2 on this CPU, `Sin` (and `Cos`) take the same time with 1, 2, 4 or 12 threads.
   - Every other elementwise op tested gets 6–8× faster with 12 threads.
   - Per element, `Sin` already costs about 5× `Tanh` and 8× `Mul` on one thread.
   - ResBlock2 uses LeakyRelu, which scales, and has no `Sin`.
3. **At 12 threads `Sin` becomes slower, not just flat:** +30% for ParIR (822 → 1,069 ms) and +28% for SeqIR.
   - The cause is ORT's idle-thread spinning. With spinning off, the single-op time returns to the 1-thread level (3.13 → 2.39 ms), and the ParIR graph's `Sin` time to 874 ms.
   - Turning spinning off is not a fix: the parallel ops then lose their speed-up, and every graph gets slower at 12 threads (ParIR RTF under profiling 0.0438 → 0.0512).
4. **`ConvTranspose` (the upsampling layers) does not scale either** (556 → 658 ms for ResBlock2). It is present in all three decoders with similar cost, so it does not explain the ranking.
5. **The IR decoders trade convolution time for `Sin` time:**
   - Their convolutions are 2.7× (ParIR) and 4.2× (SeqIR) cheaper than ResBlock2's (478 / 308 ms against 1,302 ms at 2 threads).
   - **At 2 threads** that saving outweighs the `Sin` cost for SeqIR (decoder 1,693 against 2,099 ms). For ParIR, with more `Sin` nodes, it nearly cancels out (2,318 ms).
   - **At 12 threads** ResBlock2's dense convolutions get 1.9× faster (1,302 → 698 ms), while the IR decoders' `Sin` does not get faster and even slows down. ResBlock2 then has the fastest decoder (1,466 against 1,702 and 2,264 ms). That is the reversal in section 4.
   - **INT8** graphs have the same decoder (FP32) and show the same pattern (`profile/*_int8*.json`).

So the ranking reversal is a property of the runtime's kernels, not of the decoders' arithmetic cost. With a multi-threaded `Sin`, the IR decoders' cheaper convolutions would be expected to keep them ahead, but that was not tested.
- **PyTorch:** its decoder kernels were not profiled here.
- **Possible remedies, not tested:**
  - fewer ORT threads (2–4, the usual edge setting);
  - a parallel polynomial or table approximation of `sin` in the exported graph (changes the numerics);
  - another runtime or execution provider.

## 6. Reading the results

- **The export is lossless.** All 720 Harvard sentences pass the FP32 parity thresholds, the Whisper WER is identical on every sentence, and UTMOSv2 differs by less than 0.01. That is within the scorer's run-to-run variation (±0.01–0.02).
- **INT8:**
  - **Size:** the graphs are 32–33% smaller.
  - **Speed:** INT8 is faster than FP32 for every model, by 10–17% with 2 threads and 4–9% with 12 threads. Only enc_p, dp and flow are quantized; the decoder stays FP32. How the time splits between these parts was not measured.
  - **Quality, Piper_no_VITS2_cpn and ParIR:** no detectable change. Both CIs include 0. ParIR's WER rises by 0.5 points, not significant.
  - **Quality, SeqIR:** UTMOSv2 drops by 0.036, which is significant after Holm, while WER does not change detectably.
  - **Ranking:** Piper_no_VITS2_cpn INT8 keeps the lowest WER of all evaluated systems (0.0620).
- **Decoder speed depends on the runtime and the thread count:**
  - **PyTorch, 12 threads** (earlier session): SeqIR −33% and ParIR −23% against ResBlock2.
  - **ORT, 2 threads:** SeqIR is still clearly fastest (−23% FP32, −28.5% INT8), in both rounds. ParIR's 5–9% gain in round A did not repeat in round B.
  - **ORT, 12 threads:** the order reverses. ResBlock2 is fastest, SeqIR is 7% slower and ParIR 32–36% slower.
  - **Cause** (section 5): SnakeBeta's `Sin` runs single-threaded in ONNX Runtime and gets slower under idle-thread spinning, while ResBlock2's dense convolutions scale with threads.
  - The claim "IR decoders are faster" therefore holds for PyTorch and for 2-thread ORT, not for 12-thread ORT on this CPU.

What this does **not** show:
- **Edge-device speed.** This is a workstation CPU under WSL with one timing pass per sentence, no warm-up and two orders.
- **Other PTQ scopes or methods.** The decoder was not quantized, MinMax calibration was used on 60 sentences, and QAT was not tried.
- **That SeqIR's INT8 loss is specific to its decoder.** The quantized parts (enc_p, dp, flow) have the same architecture in all three models but different trained weights.
- **Significance across training seeds,** or human MOS.

## 7. Files

Per model folder (`piper-no-vits2-cpn/`, `piper-parir-no-vits2-cpn/`, `piper-seqir-no-vits2-cpn/`):

| File | Content | Class |
|---|---|---|
| `export_record.json` | FP32 graph hashes and structure, wrapper equality, int64 cast, the 13 parity cases, native-random checks, ORT RNG behaviour | new measurement record |
| `quantize_record.json` | Calibration check and IDs, recipe and scope, INT8 structure, decoder-initializer check, reproducibility, sizes | new measurement record |
| `harvard720_{fp32,int8}_synth_record.json` | Matched-noise synthesis: graph hash, per-sentence WAV SHA-256, PyTorch determinism and parity. Timing fields are not for use | new measurement |
| `harvard_<model>_{fp32,int8}_results.json` | Per-sentence UTMOSv2 / Whisper / WER, historical schema. `rtf` is not for use | new measurement |
| `{fp32,int8}_score_record.json` | Scorer identities and environment | new measurement record |
| `*_int8_vs_fp32.json`, `*_fp32_vs_pytorch.json`, `*_int8_vs_historical.json` | Paired comparisons (UTMOSv2, WER) | new analysis |

`timing/`:

| File | Content | Class |
|---|---|---|
| `t{2,12}_{A,B}_<model>_{fp32,int8}.json` (24 files) | Per-sentence ORT timings of the deployment graphs | new measurement |
| `pytorch_ctrl_synth_record.json` | The PyTorch control of the session | new control |
| `timing_summary.json` | Means, order check, paired INT8-vs-FP32 and model-vs-control tests | new analysis |
| `machine_state_before.txt`, `machine_state_after.txt` | Load, GPU use and top processes | session record |

`profile/` (section 5):

| File | Content | Class |
|---|---|---|
| `profile_<model>_{fp32,int8}.json` | ORT per-node profile at 2 and 12 threads (default spinning): kernel time by module and op type, optimized-graph op counts | new measurement |
| `profile_<model>_{fp32,int8}_nospin.json` | The same with `session.intra_op.allow_spinning=0` | new measurement |
| `ops.json` | Single-op thread scaling (Sin, Cos, Mul, Add, Pow, LeakyRelu, Tanh) | new measurement |

The graphs (`*.onnx`), WAVs and the round log stay outside Git in `training_output/onnx-ptq/`. Their hashes are in the records.

## 8. Reproduction

From the training clone root, with a Python environment that has torch, onnx, onnxruntime and the scorers (`SPY`; the round used torch 2.13.0, onnx 1.22.0, onnxruntime 1.23.2) and the training venv (`TPY`):

```sh
SPY=<scorer python> TPY=<training venv python> RESULTS=<repository>/results/piper-ir-onnx-ptq \
  bash <repository>/scripts/evaluation/onnx/run_round.sh
```

The script stops at the first failed check and resumes after the last finished stage. The individual steps are `scripts/evaluation/onnx/{export,quantize,synthesize,time_graphs,summarize_timing}.py`, followed by `scripts/evaluation/harvard/{score,compare}.py`. The section 5 profile ran separately:

```sh
$SPY scripts/evaluation/onnx/profile_ops.py --output <dir>/ops.json
$SPY scripts/evaluation/onnx/profile_graphs.py --graph <graph.onnx> [--graph ...] --threads 2 12 [--no-spinning] --out-dir <dir>
```

In that command, `scripts/` stands for `<repository>/scripts/`. The tests are in `tests/test_onnx_ptq.py`.

**Expected outputs:**
- Export and quantization are deterministic, so the graph hashes should match.
- Synthesis is deterministic, so the WAV hashes should match.
- UTMOSv2 varies between scoring sessions.
- RTF varies between sessions; only within-session ratios are meaningful.
