# Intel N100: six-model OpenVINO CPU measurements, 2026-11-01

*Tiếng Việt: [README.vi.md](README.vi.md)*

The received session is complete: **6 models × 2 precisions × 3 rounds × 720 sentences = 25,920 timed observations**. Archive integrity, source hashes and per-sentence aggregates were checked on the PC. This is one N100 session with selective NNCF INT8, distinct from workstation PyTorch and ONNX Runtime PTQ measurements.

**Timing is complete; quality of these OpenVINO INT8 artifacts is not certified.** Two unusual duration observations are retained and described below. No new training, inference, export, quantization or scoring was performed to import this evidence.

## Results and interpretation

The primary RTF below is **sum(inference seconds) / sum(generated audio seconds)** over all three rounds. Lower is faster. Mean latency uses the same 2,160 observations per model/precision. Peak RSS is the maximum worker-process RSS, including runtime, inputs and model; it is not parameter memory alone.

| Model | RTF FP32 | RTF INT8 | Mean latency FP32 / INT8 (ms) | Peak RSS FP32 / INT8 (GiB) |
|---|---:|---:|---:|---:|
| `baseline-resblock2` | 0.07095 | 0.06880 | 198.36 / 211.17 | 11.08 / 3.83 |
| `parallel-ir` | 0.07053 | 0.06739 | 178.28 / 169.42 | 1.64 / 2.86 |
| `sequential-ir` | 0.06045 | 0.05511 | 154.63 / 146.34 | 1.55 / 3.02 |
| `piper-no-vits2-cpn` | 0.06457 | 0.05870 | 164.43 / 147.30 | 1.80 / 2.13 |
| `piper-parir-no-vits2-cpn` | 0.06322 | 0.05820 | 166.96 / 154.26 | 1.22 / 1.77 |
| `piper-seqir-no-vits2-cpn` | 0.05326 | 0.04788 | 137.25 / 123.52 | 1.27 / 1.80 |

Piper-SeqIR has the lowest RTF in this session. Its selective INT8 duration-weighted RTF is 10.09% lower than its FP32 RTF. All recorded per-utterance RTF values are below 1, but this establishes only the measured model-inference timing, not streaming latency or end-to-end text-to-WAV behavior.

Baseline INT8 has a lower RTF but **higher mean latency** than Baseline FP32. Its total generated audio is longer (6,629.73 versus 6,039.01 seconds). Native graph randomness is not matched between these precision runs, so an RTF reduction must not be presented as the same speedup for an identical waveform.

Distribution statistics use each sentence's `latency_s / audio_seconds`; they differ from the primary duration-weighted RTF. P95 uses the nearest-rank definition, `ceil(0.95 * n)`.

| Model | Mean RTF FP32 / INT8 | Median RTF FP32 / INT8 | P95 RTF FP32 / INT8 |
|---|---:|---:|---:|
| `baseline-resblock2` | 0.07018 / 0.06932 | 0.06745 / 0.07129 | 0.08476 / 0.08613 |
| `parallel-ir` | 0.07060 / 0.06753 | 0.06853 / 0.06381 | 0.08057 / 0.08380 |
| `sequential-ir` | 0.06056 / 0.05529 | 0.05876 / 0.05093 | 0.07111 / 0.07030 |
| `piper-no-vits2-cpn` | 0.06474 / 0.05887 | 0.06235 / 0.05502 | 0.07833 / 0.07548 |
| `piper-parir-no-vits2-cpn` | 0.06331 / 0.05830 | 0.06231 / 0.05618 | 0.06918 / 0.06762 |
| `piper-seqir-no-vits2-cpn` | 0.05339 / 0.04801 | 0.05228 / 0.04578 | 0.06001 / 0.05945 |

Duration-weighted RTF for each round:

| Model | FP32: r0 / r1 / r2 | INT8: r0 / r1 / r2 |
|---|---|---|
| `baseline-resblock2` | 0.06986 / 0.07313 / 0.06977 | 0.06843 / 0.06800 / 0.06998 |
| `parallel-ir` | 0.07070 / 0.07035 / 0.07054 | 0.06711 / 0.06853 / 0.06655 |
| `sequential-ir` | 0.06004 / 0.06005 / 0.06124 | 0.05455 / 0.05463 / 0.05614 |
| `piper-no-vits2-cpn` | 0.06492 / 0.06399 / 0.06481 | 0.05841 / 0.05831 / 0.05939 |
| `piper-parir-no-vits2-cpn` | 0.06274 / 0.06396 / 0.06297 | 0.05747 / 0.05878 / 0.05836 |
| `piper-seqir-no-vits2-cpn` | 0.05289 / 0.05275 / 0.05414 | 0.04810 / 0.04857 / 0.04698 |

The 720 sentences repeat three times; they are not 2,160 independent test sentences. These descriptive results do not establish significance across training seeds, devices, sessions, power profiles or decoder-only causation. Do not combine older quality scores with this NNCF INT8 experiment.

Sources: [original FP32 CSV](evidence/results-n100-t4-20261101T080423Z/summary_fp32.csv), [original INT8 CSV](evidence/results-n100-t4-20261101T080423Z/summary_int8.csv), [per-cell source hashes and tables](evidence/results-n100-t4-20261101T080423Z/summary.json), [PC recomputation](verification/received/rtf-recomputed.csv).

## Models and artifacts

| Model ID | Cohort | Selected checkpoint |
|---|---|---|
| `baseline-resblock2` | historical Q05 FP32 | `best-epoch=1489-val_loss_mel=19.7882.ckpt` |
| `parallel-ir` | historical Q05 FP32 | `best-epoch=1386-val_loss_mel=18.7681.ckpt` |
| `sequential-ir` | historical Q05 FP32 | `best-epoch=1442-val_loss_mel=19.0136.ckpt` |
| `piper-no-vits2-cpn` | Config A, round-2 FP32 | `best-epoch=1403-val_loss_mel=19.7719.ckpt` |
| `piper-parir-no-vits2-cpn` | Config A, round-2 FP32 | `best-epoch=1486-val_loss_mel=19.0069.ckpt` |
| `piper-seqir-no-vits2-cpn` | Config A, round-2 FP32 | `best-epoch=1455-val_loss_mel=19.0332.ckpt` |

The first three belong to the historical Q05 cohort; the last three are Config A models using the corrected round-2 FP32 export sources. External Piper, Low and High are not members of this six-model experiment. All six output 22,050 Hz audio. Full checkpoint, FP32 ONNX, configuration and export-record identities are in [models.json](evidence/models.json). Checkpoints themselves were not included or loaded.

**INT8 is selective, not full-model INT8.** NNCF targets eligible convolutions under `/enc_p/`, `/dp/` and `/flow/`, using CPU / MIXED preset / fast bias correction and a shared 60-sentence train-only calibration set. Decoder weighted operations remain FP32. The corrected policy also excludes masked duration-spline operations from activation quantization while keeping flow `pre/convs/proj` convolutions eligible. Executed runtime records identify 121 INT8 weighted kernels per historical model and 97 per Config A model.

Each model has 13 matched-noise FP32 OpenVINO-versus-ONNX Runtime parity records (78 total); all satisfy NRMSE ≤ 0.001 and max absolute error ≤ 0.005. These are FP32 engineering checks, not INT8 perceptual quality tests. Three pilot WAVs per precision/model were checked (36 total); pilot WAVs are individually peak-normalized for listening, unlike the waveform measured by the timer.

## Hardware, environment and timing protocol

| Setting | Recorded value |
|---|---|
| Device | Intel N100; 4 logical CPUs; native Ubuntu installation |
| Platform | `Linux-7.0.0-31-generic-x86_64-with-glibc2.39` |
| Python | 3.12.3 |
| OpenVINO / NNCF | 2026.4.1 / 3.4.0 |
| NumPy / ONNX / ONNX Runtime | 2.4.6 / 1.23.2 / 1.31.1 |
| Power before/after | `balanced`; `intel_pstate`; governor `powersave`; EPP `balance_performance` |
| Execution | CPU, batch 1, synchronous inference, 4 threads, 1 stream, `LATENCY`, precision hint `f32`, execution mode `ACCURACY` |
| Inputs | Harvard-720 phoneme IDs; scales `[0.667, 1.0, 0.8]`; each actual output length is the RTF denominator |
| Warm-up | 10 calls distributed over sentence lengths; not every dynamic shape is warmed separately |
| Repeats | 3 rounds; fresh worker process per model/precision/round |
| Ordering | Shuffled cells with seed 1234, reverse order in r1; sentence order seeded `1234 + round` |
| Recorded start/end UTC | 08:04:24.224763 / 09:18:46.163322 on 2026-11-01 |
| Local time, UTC+07 | 15:04:24 / 16:18:46; about 74 minutes 22 seconds |

The timer encloses `InferRequest.infer(share_outputs=True)`, including its Python/OpenVINO call overhead. Model load/compile, phonemization, input preparation/assignment, output validation/copy and file writing are outside it. First-use shape costs after the ten warm-up calls may still be timed. The `powersave` governor string under `intel_pstate` does not establish a fixed CPU clock; no continuous frequency or throttle trace was captured.

The archived environment lock, power records, command list and worker start/end environments are part of this package. Maximum recorded sensor value was 76 °C, sampled only at cell boundaries; this cannot rule out thermal throttling or swapping during a cell.

## Retained duration observations and missing quality evidence

1. **Baseline FP32, r1 (second round), zero-based Harvard idx 379:** “That move means the game is over.” Generated duration 87.0632 s; inference 12.5710 s; RTF 0.14439. Other FP32 rounds for the same sentence produced 2.3104 and 3.0302 s. This cell's peak RSS was 11.0809 GiB, versus 2.5242 and 2.4918 GiB in the other two FP32 cells. Co-occurrence does not establish causation.
2. **Piper-SeqIR FP32, r2, idx 332:** “The idea is to sew both edges straight.” Generated duration 7.3491 s. It remains in all summaries.

See [duration observations and matching rounds](verification/received/duration-observations.json). No observation was deleted, capped or silently repaired. A future sensitivity analysis must remain separate from these original aggregates.

The suite did not save the timed waveform for every sentence, so the two specific long outputs cannot be listened to from this package. Listening/WER/UTMOS for the new OpenVINO INT8 artifacts is still pending. The cause of the duration outliers and performance under other thread counts remain unestablished. No existing quality conclusion is extended to these artifacts.

## Calibration failure and corrected session

| Stage | Evidence | Interpretation |
|---|---|---|
| Initial diagnostic | [numeric-failure](diagnostics/numeric-failure/diagnostic-report.json), [log](diagnostics/numeric-failure/diagnostic.log), [numeric event](diagnostics/numeric-failure/numeric-event.json) | Calibration statistics overflowed; this stage is not accepted timing evidence. |
| SeqIR probe | [probe-report.json](diagnostics/spline-probe/probe-report.json), [empty-branch audit](diagnostics/spline-probe/empty-branch-audit.json), [code diff](diagnostics/spline-probe/candidate-code/suite-change.diff) | The selected spline branch was empty for 60/60 observed inputs; a probe excluding masked spline operations completed without intercepted numerics. |
| Final six-model preparation | `evidence/work-n100-clean-20261101T075458Z/<model>/preparation.json`, [prepare log](evidence/work-n100-clean-20261101T075458Z.prepare.log) | New work directory, all six models prepared under `n100-spline-fp32-v1-20261101`; numeric guard passed. |
| Final measurement | [schedule](evidence/results-n100-t4-20261101T080423Z/schedule.json), [run log](evidence/results-n100-t4-20261101T080423Z.run.log) | All 36 cells completed; final logs have no observed RuntimeWarning/overflow/invalid/divide-by-zero/Traceback. |

The probe and final preparation are distinct from the earlier ORT PTQ rounds. This change adjusts NNCF quantization boundaries; it does not repair a training checkpoint or alter the source FP32 weights. The final suite SHA-256 is `fb20086c7907a7730fd3e3446544c6c816809c88b2164e640e1b10882af902cf`. The original suite remains in `evidence/code-history/`.

## Package layout and provenance

| Path | Contents |
|---|---|
| `evidence/results-n100-t4-20261101T080423Z/` | 36 cell directories with `result.json`, 720-row `utterances.jsonl`, execution nodes; 36 worker logs; schedule and aggregate tables |
| `evidence/work-n100-clean-20261101T075458Z/` | Six preparation records, FP32 parity and FP32/INT8 execution-node records |
| `evidence/*.log`, `*.environment-lock.txt`, `*.power-*.txt` | Final preparation/run logs, environment locks and power settings |
| `evidence/metadata/`, `evidence/reference/`, `evidence/*.py` | Frozen inputs/split/configs, source/export provenance and exact executed source snapshot |
| `diagnostics/` | Earlier numerical failure and separate successful SeqIR probe; never merged into final timings |
| `provenance/` | Original archive member listing/checksum and the numbered 05–10 command record |
| `verification/received/` | PC archive audit, recomputed metrics, duration observations and original summary copies |
| [verification/import-check.json](verification/import-check.json) | Fresh import checks, unchanged pre-existing work and verifier tests for missing/changed files and a Git-only checkout |
| `manifest.json` | Source member, size, original SHA-256, repository path, derivative SHA-256 and explicit transformation/disposition for every imported item |
| `.local/originals/` | Ignored local originals, including 72 ONNX/XML/BIN/WAV artifacts; not present in a Git-only checkout |
| `SHA256SUMS`, `verify.py` | Checksums for the tracked package and a standard-library-only verifier |

240 imported text files are versioned: **232 byte-identical copies and 8 declared derivatives** that replace only personal home-directory prefixes. Original unredacted bytes remain local. The original preflight and PC checksum files still refer to original bytes; use this package's verifier for repository derivatives. One generated `.pyc` file is excluded from import and remains identified in the original archive and manifest. No dataset audio cache, virtual environment, checkpoint or transfer archive is added to Git.

The source archive remains outside the repo: `N100-RESULTS-20261101T092554Z/results-n100-t4-20261101T080423Z.tar.gz`, **1,178,279,698 bytes**, SHA-256 **`0655d3b9d4f146a0a4423f63c58e07b9c21234675f6f8f1e02e1f4f75a4b69b1`**. The received archive passed gzip CRC, member-list comparison, extraction and extracted-file hash verification (283 files, 345 entries). Import then checked copied bytes against those identities. The 120 original preflight identities include the excluded bytecode item; this is explicitly recorded, not treated as a copied file.

## Verify without rerunning models

From the repository root, using an existing Python 3.10+:

```sh
python results/edge-n100-openvino/2026-11-01/verify.py
python results/edge-n100-openvino/2026-11-01/verify.py --with-local-artifacts
```

The first command works in a Git-only checkout and checks hashes, provenance links, all 25,920 observations, every cell and the aggregate CSV/JSON statistics. The second also requires and hashes all retained local originals; missing local artifacts fail explicitly. Neither command imports OpenVINO or runs a model. A passed verification does not certify quality or explain the outliers.

For provenance, the already completed N100 commands were `openvino_suite.py prepare --work work-n100-clean-20261101T075458Z` and `openvino_suite.py run --work work-n100-clean-20261101T075458Z --threads 4 --rounds 3 --output results-n100-t4-20261101T080423Z`. Their shell guards, pinned versions, preflight and packaging steps are preserved in [LENH-05-DEN-10-N100.txt](provenance/commands/LENH-05-DEN-10-N100.txt). These are historical execution instructions, not an instruction to rerun them on the PC. Any future N100 experiment needs a new work/output directory and its own results.
