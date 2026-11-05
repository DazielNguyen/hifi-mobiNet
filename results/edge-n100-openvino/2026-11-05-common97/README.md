# Intel N100 common97 benchmark — November 5, 2026

*Tiếng Việt: [README.vi.md](README.vi.md)*

**Verified: 3 models × 2 precisions × 3 rounds × 720 sentences = 12,960 timed observations.**
Within this session, selective INT8 common97 reduces duration-weighted RTF by
**12.73–13.88%** relative to FP32. Piper-SeqIR has the lowest INT8 RTF, **0.04579**.
These are model-inference timings; independent WER/CER/UTMOS/human MOS for these
artifacts remains pending.

The user copied `20261105T075322Z-benchmark-common97.4SOO5I` from N100 to Downloads.
All **77 received files / 14,530,595 bytes** are preserved unchanged. This import
only checks files and recomputes statistics; it runs no inference, quantization,
scoring or new benchmark.

**Supplemental preparation log:** `20261105T073706Z-common97.6nBwhE` contains
only [prepare-common97.log](preparation-evidence/prepare-common97.log), 367,051
bytes, preserved unchanged and hash-checked. It records three completed 60/60
calibration blocks and ends with `COMMON97 READY: three Piper models prepared
and local catalog activated.` No Traceback/RuntimeWarning or arithmetic-error
message was found in the received log. Its success marker is supplemented by
the independently checked work artifacts below; READY alone is not the binding.

The log names this N100 work directory, now received:

```text
/home/capstone/benchmarks/hifi-mobiNet-n100-openvino-2026-10-31/hifi-mobiNet-n100-openvino-2026-10-31/work-common97-20261105T073706Z
```

It also names this catalog, received and hash-checked on November 6, 2026:

```text
/home/capstone/benchmarks/hifi-mobiNet-n100-common97-2026-11-05/.local/n100-common97-models.json
```

**Supplemental demo readiness records:** four files / **3,552 bytes** from
three directories are preserved unchanged in [demo-evidence/](demo-evidence/)
and hash-checked against Downloads:

| Received directory | Files | Recorded evidence |
|---|---|---|
| `20261105T074110Z-check.8oqXmf` | `check.log`, `doctor.json` | Byte-identical records; ready, en-us frontend matches 720/720 |
| `20261105T074122Z-server.CkuEHv` | `server.log` | Demo ready on port 8501, referring to the following session |
| `20261105T074123Z-a3c2e898` | `server.json` | Server initialized on N100 with matching environment/catalog hash |

Doctor and server records are dated **14:41:11 and 14:41:23 UTC+07 on November 5**,
before the benchmark. CPU, kernel, Python and all five recorded package versions
match the timing session. The server's source hash matches the repository copy
[n100_demo.py](reference/n100_demo.py); its preflight checks preparation/IR hashes
and the frontend before recording readiness.

Both records name catalog SHA-256
`d2f9895b90882430984ce9cb9751324a382bffbc4a60bf0fb0448db02c5e9f42`,
which exactly matches the received [n100-common97-models.json](catalog-evidence/n100-common97-models.json),
10,760 bytes. It contains six FP32 entries and three common97 INT8 selections
for Piper ResBlock2, ParIR and SeqIR. Both records report **`inference_performed: false`**.
The received server session contains only `server.json`, with no synthesis
WAV/JSON. These are startup checks, not independent graph or speech-quality checks.

**Work artifacts received:** `C:\Users\duyng\Downloads\work-common97-20261105T073706Z`
contains **84 files / 478,136,232 bytes**, 28 files per model. Every file was freshly
hashed and checked against original preparation/benchmark hashes where available.
The 15 JSON records are copied unchanged into [work-evidence/](work-evidence/).
The 69 XML/BIN/ONNX/WAV files remain local in Downloads, fully inventoried in
[work-manifest.json](work-manifest.json), outside Git.

| Check on received artifacts | ResBlock2 | ParIR | SeqIR |
|---|---:|---:|---:|
| Preparation matches benchmark cell hashes | 6/6 | 6/6 | 6/6 |
| FP32/INT8 XML/BIN matches preparation hashes | 4/4 | 4/4 | 4/4 |
| Actual INT8 FQ nodes restricted to selected Conv edges | 96 | 96 | 96 |
| INT8 kernels in runtime records | 97 | 97 | 97 |
| FP32 + INT8 smoke WAV format/sample checks | 6/6 | 6/6 | 6/6 |
| INT8 alignment WAV hash/sample checks | 12/12 | 12/12 | 12/12 |
| INT8 alignment WAVs longer than 10 s | 6 | 6 | 6 |
| Longest alignment WAV, measured from file | 23.452 s | 21.513 s | 22.291 s |

All 384 FQ range constants per INT8 graph were read from BIN: finite, low < high,
256 levels. Every FQ output edge feeds activation input port 0 of a selected Conv;
none leaks to duration/alignment or decoder. All 60 calibration token sequences
match dataset rows and belong exclusively to train. Policy, implementation,
numeric guard and regression config hashes match all three preparations.
See [work-verification.json](verification/work-verification.json).

FP32 parity records report 13/13 cases per model with consistent formulas and
thresholds. Alignment records report 12/12 cases with coverage 1 and zero missing
frames. These are **record-consistency checks**, not rerun inference on internal
tensors. All 54 WAVs were fully read as mono PCM16/22,050 Hz and checked for sample
count/duration. The 36 alignment WAVs have original hashes; 18 smoke WAVs have
fresh import hashes and original dimensions only. Listening and speech-quality
scoring were not performed in this audit.

**The INT8 catalog-to-benchmark chain is now hash-bound.** The catalog was activated
at `2026-11-05T07:40:35.325686+00:00`, after all three preparations and before
doctor/server. Policy and implementation match common97. All three
`int8_preparation` hashes match benchmark records, and all six selected INT8
XML/BIN hashes match actual work files. See `catalog` in
[verification.json](verification/verification.json).

FP32 selections retain `work-n100-clean-20261101T075458Z`; INT8 selections use
`work-common97-20261105T073706Z`. For the three Piper models the FP32 BIN hashes
match the new benchmark, but XML hashes differ. This does not establish identical
FP32 graphs or assign benchmark timing to the demo FP32 path. The preparation
code intentionally retained these catalog entries; this is not a copy defect.

No further catalog transfer or PTQ/benchmark rerun is required. A particular
demo output still needs its synthesis WAV/JSON; preparation WAVs are separate.
WER/CER/UTMOS/MOS remains pending. The catalog verifies recorded startup
selection, not the current live server state.

## Results

Primary RTF = total inference seconds / total generated audio seconds, pooled
across three rounds. Mean latency uses 2,160 observations/model/precision. Peak
RSS is the maximum of three worker-process peaks, including runtime/input/model.

| Model | FP32 RTF | common97 INT8 RTF | RTF reduction | Mean latency FP32 / INT8 (ms) | Peak RSS FP32 / INT8 (GiB) |
|---|---:|---:|---:|---:|---:|
| Piper ResBlock2 | 0.06473 | 0.05649 | 12.73% | 164.81 / 144.13 | 1.750 / 1.692 |
| Piper ParIR | 0.06549 | 0.05663 | 13.53% | 172.83 / 147.83 | 1.232 / 1.129 |
| Piper SeqIR | 0.05317 | 0.04579 | 13.88% | 136.63 / 118.08 | 1.166 / 1.056 |

Mean latency decreases by 12.55%, 14.47%, and 13.58%, respectively. Native RNG
is not matched between FP32 and INT8. Total INT8 audio duration differs from
FP32 by approximately +0.20%, −1.09%, +0.35%; RTF reduction is not identical
to latency reduction or a speedup on a fixed waveform.

| Model | Mean RTF FP32 / INT8 | Median RTF FP32 / INT8 | P95 RTF FP32 / INT8 | P95 latency FP32 / INT8 (ms) |
|---|---:|---:|---:|---:|
| Piper ResBlock2 | 0.06490 / 0.05664 | 0.06227 / 0.05394 | 0.07907 / 0.07068 | 220.48 / 198.26 |
| Piper ParIR | 0.06561 / 0.05671 | 0.06447 / 0.05523 | 0.07203 / 0.06383 | 219.41 / 189.47 |
| Piper SeqIR | 0.05330 / 0.04589 | 0.05219 / 0.04480 | 0.06040 / 0.05264 | 172.48 / 148.27 |

Distributional RTF uses each sentence's latency/duration. P95 is nearest-rank,
at `ceil(0.95 × n)`. The rounds repeat 720 texts; they are not 2,160 independent texts.

| Model | FP32 RTF r0 / r1 / r2 | INT8 RTF r0 / r1 / r2 |
|---|---|---|
| Piper ResBlock2 | 0.06384 / 0.06489 / 0.06545 | 0.05585 / 0.05719 / 0.05644 |
| Piper ParIR | 0.06487 / 0.06657 / 0.06504 | 0.05733 / 0.05670 / 0.05587 |
| Piper SeqIR | 0.05299 / 0.05283 / 0.05369 | 0.04641 / 0.04543 / 0.04552 |

Recomputed [CSV](verification/recomputed.csv), [JSON with per-cell statistics](verification/recomputed.json).
Received [FP32 CSV](evidence/results/summary_fp32.csv), [INT8 CSV](evidence/results/summary_int8.csv)
and [summary with source hashes](evidence/results/summary.json) agree with the
recomputed per-utterance values.

## Model and precision provenance

| Model ID | Selected checkpoint |
|---|---|
| `piper-no-vits2-cpn` | `best-epoch=1403-val_loss_mel=19.7719.ckpt` |
| `piper-parir-no-vits2-cpn` | `best-epoch=1486-val_loss_mel=19.0069.ckpt` |
| `piper-seqir-no-vits2-cpn` | `best-epoch=1455-val_loss_mel=19.0332.ckpt` |

All cells bind the same [models manifest](reference/models.json) by SHA-256.
These are Config A sources from the corrected round-2 FP32 export; historical
Q05 models, external Piper, Low and High are outside this session. Original
deployment ONNX/checkpoints are not in the work delivery. The three received
`fp32_matched_validation.onnx` files are auxiliary parity models, inventoried
by hash without loading them.

The recorded suite SHA-256 is
`7c064dcd88bd4727e781af34aa4c1a83365c878108dc3717540699c72a592323`,
matching the archived [source](reference/openvino_suite.py) copied from the repo.
This suite requires policy/IR identities and graph gates before measurement.
Independent inspection of all nine INT8 runtime records finds exactly **97
selected Convolution kernels: 37 encoder + 20 regular SDP + 40 main-flow**,
matching the [current policy](reference/piper-common97.json). Decoder weighted
kernels are FP32 (23 ResBlock2, 32 ParIR, 23 SeqIR). All weighted kernels in
the nine FP32 records are FP32.

**A fresh static XML/BIN audit of the received N100 IR is now complete.** Each
preparation matches six measured cell hashes, binding four IR files and 12
alignment WAVs per model. [verify_work.py](verify_work.py) reads actual graph
edges and binary ranges, independently of recorded kernel counts.
[verification.json](verification/verification.json) records the binding chain.
No PC/WSL or old N100 preparation is substituted for the new N100 records.

The unchanged `quality_status` text was inherited from preparation and includes
“N100 RTF not certified by preparation”. It describes the preparation stage's
scope; the new benchmark supplies measured timing separately.

## Hardware and timing protocol

| Setting | Recorded value |
|---|---|
| CPU | Intel N100; 4 logical CPUs |
| Platform | `Linux-7.0.0-42-generic-x86_64-with-glibc2.39` |
| Python | 3.12.3 |
| OpenVINO / NNCF | 2026.4.1 / 3.4.0 |
| NumPy / ONNX / ONNX Runtime | 2.4.6 / 1.23.2 / 1.31.1 |
| Runtime | CPU; batch 1; synchronous; 4 threads; 1 stream; `LATENCY`; `f32` hint; `ACCURACY` |
| Input | Harvard-720 phoneme IDs; scales `[0.667, 1.0, 0.8]`; 22,050-Hz output |
| Warm-up | 10 calls selected across input lengths, outside timing |
| Repetition/order | Three rounds; separate process/cell; seed 1234 shuffled cells, reversed in r1; sentence seed `1234 + round` |
| Governor | `powersave` for four CPUs at each worker boundary |
| Maximum recorded sensor | 72 °C; boundary snapshots only |
| UTC | November 5, 07:53:23–08:26:50 |
| UTC+07 | 14:53:23–15:26:50; approximately 33 min 27 s |

Only `InferRequest.infer(share_outputs=True)` is timed. Loading/compilation,
phonemization, input preparation/assignment, output copy/validation and writes
are outside the timer. First encounters with dynamic shapes after warm-up can
still affect timing. This is neither streaming latency nor end-to-end text-to-WAV latency.

No power-profile/EPP report, process inventory or continuous frequency/thermal
trace was received. Do not carry forward the November 1 `balanced` profile or
exclude throttling from sparse sensor snapshots. The kernel changed from
`7.0.0-31` to `7.0.0-42`; old/new recipe causality needs same-session controls.

## Observations and limitations

- All 12,960 rows contain finite positive measurements, the correct formulas,
  and the complete expected IDs/order. Maximum RTF is 0.12903; all are below 1.
- Timed audio spans **1.463–4.203 seconds**. No timed output exceeds 10 s;
  this benchmark does not certify the long-sentence tail fix.
- Recorded peaks range from 0.2357 to 0.8495, with none above 1. These summaries
  cannot establish intelligibility, absence of distortion, or complete text coverage.
- The parent log lists all 18 cells. All 18 worker logs are **empty files**, retained
  exactly; absence of logged warnings alone is not treated as a pass.
- N100 preparations, IR, parity, graph/alignment records and long WAVs have now
  been received and checked, as has the demo catalog. Complete package lock, frontend binaries
  and raw parity/coverage tensors remain absent. READY alone is not certification.
- Timed waveforms were not saved by the suite. WER/CER/UTMOS/human MOS is pending;
  no quality scores from PyTorch or ONNX Runtime are assigned to these artifacts.
- These are descriptive statistics from one session, not significance across
  sessions, hardware or training seeds.

The [November 1 session](../2026-11-01/README.md) remains unchanged. Its old INT8
graphs leaked activation quantization into duration/alignment; they are not pooled
with common97. The [integration report](../../../docs/quantization/piper-common97-integration-2026-11-05/README.vi.md)
documents the fix and separate PC/WSL regression evidence.

## Files and verification

`evidence/` preserves all 77 benchmark files. `preparation-evidence/` preserves
one supplemental preparation log, `demo-evidence/` four startup records, and
`work-evidence/` 15 JSON records, and `catalog-evidence/` the unchanged active
catalog: **98 received files archived in the repository**.
Both duplicate check files are retained. [work-manifest.json](work-manifest.json)
inventories all 84 work files, with 69 binary/audio files retained externally in
Downloads. None is omitted from the audit. `reference/` contains 11 hash-bound
repo source/config/manifest/metadata files. [manifest.json](manifest.json) records
copy provenance and remaining gaps. `verification/` contains derived reports;
`SHA256SUMS` covers the repository package except itself.

From the repository root, using Python 3.10+:

```sh
python results/edge-n100-openvino/2026-11-05-common97/verify.py
```

To compare the original received directory too:

```powershell
.venv/Scripts/python.exe results/edge-n100-openvino/2026-11-05-common97/verify.py --source "C:\Users\duyng\Downloads\20261105T075322Z-benchmark-common97.4SOO5I" --preparation-source "C:\Users\duyng\Downloads\20261105T073706Z-common97.6nBwhE" --demo-source-root "C:\Users\duyng\Downloads" --catalog-source "C:\Users\duyng\Downloads\n100-common97-models.json"
.venv/Scripts/python.exe -B results/edge-n100-openvino/2026-11-05-common97/verify_work.py --source "C:\Users\duyng\Downloads\work-common97-20261105T073706Z"
```

The verifier uses only the standard library, checks inventory/hashes, all 36
summary source hashes, schedule, warm-up, IDs/order, environments/timestamps,
numeric formulas, 18 runtime inventories and every aggregate against JSON/CSV.
It also checks the four demo files, identical doctor/check bytes, matching catalog
hashes, frontend, environment, chronology, demo source and preparation/benchmark
bindings. The actual catalog is checked against startup hashes, metadata,
recipe and all three INT8 preparations; historical FP32 selections are recorded
separately from new benchmark FP32. `verify.py` checks archived records; `verify_work.py` rereads all 84
external work files, graph XML/BIN and WAVs. Missing/modified files fail. No full
external transfer checksum was supplied; benchmark hashes bind preparations,
IR and alignment WAVs, while other bytes have import hashes. Retain Downloads
to repeat the external artifact audit. Neither command runs a model.
