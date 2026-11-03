# Piper Low — Harvard-720 results

*Tiếng Việt: [README.vi.md](README.vi.md)*

**PyTorch FP32 evaluation and ONNX FP32 scoring are complete on 720 utterances.**
The continuation finished at 2026-11-02 17:51 UTC (November 3, 00:51 UTC+7).
ONNX still fails one numerical parity case; its quality and timing observations
are diagnostic only. Final status:
[`completed_with_onnx_parity_failure`](fp32-evaluation/status.json).
Machine-generated summary: [RESULTS.txt](fp32-evaluation/RESULTS.txt).

## Quality

| Model / runtime | Sample rate | N | Predicted UTMOSv2 ↑ | Macro WER ↓ | Corpus WER ↓ | Macro CER ↓ | Corpus CER ↓ | WER-zero sentences |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Low / PyTorch FP32** | 16000 Hz | 720 | **3.5509** | **7.2147%** | 7.0185% | **3.2423%** | 3.1857% | 467 |
| Low / ONNX FP32 — diagnostic | 16000 Hz | 720 | 3.5535 | 7.2147% | 7.0185% | 3.2456% | 3.1893% | 467 |
| Medium / PyTorch FP32 — same-session control | 22050 Hz | 720 | 3.5102 | 6.1194% | 5.9561% | 2.7518% | 2.6850% | 502 |

Low has **+0.0408 predicted UTMOSv2**, but **+1.0954 percentage points macro WER**
and **+0.4905 percentage points macro CER** versus Medium. It is not uniformly
better. UTMOSv2 is a predictor and WER/CER use ASR; neither replaces listening
or human MOS. Paired bootstrap 95% intervals are [0.0181, 0.0638] for the UTMOS
difference and [+0.1546, +2.0691] percentage points for WER. These intervals
describe variation across utterances, not across training runs.

Evidence: [Low PyTorch results](fp32-evaluation/pytorch/results.json),
[Low ONNX results](fp32-evaluation/onnx-fp32/results.json),
[Medium control results](fp32-evaluation/medium-control/results.json),
[paired comparison and methods](fp32-evaluation/low-vs-medium.json).
Each runtime directory contains synthesis, validation, scorer and progress
records. Original stage logs are retained in [logs/](fp32-evaluation/logs/).

## Workstation timing

Host: **Intel Core i7-14700 / WSL2, 24 logical CPUs**;
[host record](fp32-evaluation/host.json). These are not N100 measurements.
RTF is inference time divided by audio duration; lower is faster.
PyTorch CPU, 12 threads, one quality-synthesis pass: Low **0.03291**, Medium
**0.04138** mean per-utterance RTF.

ONNX Runtime CPU, 720 utterances/pass, intra-op 2 or 12 threads, inter-op 1:

| Threads | Pass / order | Low FP32 — diagnostic | Medium FP32 |
|---|---|---:|---:|
| 2 | A: Low → Medium | 0.02450 | 0.03376 |
| 2 | B: Medium → Low | 0.02555 | 0.03357 |
| 12 | A: Low → Medium | 0.01988 | 0.02590 |
| 12 | B: Medium → Low | 0.02024 | 0.02613 |

These are arithmetic means of sentence RTF, not application latency or model
loading time. Native RNG, seed 1234, no warmup; reversed A/B ordering records
drift. Per-utterance records and protocol: [timing/](fp32-evaluation/timing/).
Do not substitute timings from the matched-noise parity loop, or extrapolate
this table to N100.

## Checkpoint and protocol

- Completed 1500 epochs; selected minimum validation mel loss checkpoint:
  `best-epoch=1351-val_loss_mel=19.1691.ckpt` (zero-based epoch).
- Checkpoint SHA-256: `7d34322f637e03eb1186fe8dd5681e1c5c0d81666d21398467f00ca441103e55`.
  Full training state: **845907083 bytes**, not a deployment artifact size.
- Inference parameters: **15663771**, excluding posterior encoder and
  training-only SDP paths; decoder: **1664768**. Definitions and module counts:
  [parameters.json](fp32-evaluation/parameters.json).
- ONNX FP32: **63469360 bytes**, opset 18, SHA-256
  `1afe4889d36086c1e39f1f0b6b1d1a4bc0b077404c00b3b6444cfc2568205a94`;
  [export record](graphs/export_record.json).
- Low retains Medium's network parameter shapes / ResBlock2 at 16 kHz;
  it is not the reduced x-low network. Raw mel validation losses across sample
  rates are not a quality ranking.
- Same Harvard IDs, texts/phonemes, per-utterance seed 1234 and scales
  0.667 / 1 / 0.8. WAVs retain native 16 kHz / 22.05 kHz rates; scorer input
  rates and required internal resampling are recorded in `score_record.json`.
- UTMOSv2 fusion-stage-3 fold 0 and Whisper small use verified cached weights.
  All **720/720** Medium control WAV hashes reproduce the historical session:
  [reproducibility record](fp32-evaluation/medium-reproducibility.json).

## ONNX numerical limitation

The initial export checks passed 13/13 cases. Full Harvard evaluation produced
finite outputs, equal lengths and reproduced PyTorch references for all 720,
but only **719/720 pass numerical parity**. For `idx=719`, “When you hear the
bell, come quickly.”, NRMSE **0.0010694584 exceeds 0.001**; max absolute error
0.0010539144 remains below 0.005.

The [probe](fp32-evaluation/parity_probe.json) also observes NRMSE 0.0014113898
between PyTorch at 2 and 12 threads; folding weight norm does not change the
output in that case. This suggests numerical sensitivity without establishing
that every cause is resolved. Thresholds were not relaxed, the failed session
was preserved, and **`deployment_approved: false`** remains in the final record.

## Session and evidence inventory

The original session in this directory stopped at parity. The continuation in
`fp32-evaluation/` reuses hash-verified WAVs/graphs and completes scoring,
timing and comparisons with diagnostic ONNX labeling.
[SESSION-INDEX.txt](SESSION-INDEX.txt) is the earlier continuation setup note;
this README and the continuation status record describe the final outcome.

Original [SHA256SUMS](SHA256SUMS) covers 41 files;
[continuation SHA256SUMS](fp32-evaluation/SHA256SUMS) covers 77 files.
All **118/118** entries were freshly read and verified on November 3, 2026.
This later README does not alter original evidence or hash manifests. On
Linux, run `sha256sum -c SHA256SUMS` in each manifest's directory.

Raw WAVs, graphs and checkpoints remain in Ubuntu WSL under:

```text
/home/capstone/hifimobinet-train/hifi-mobiNet/training_output/
  piper-low-no-vits2-cpn-2026-10-30/
  piper-low-eval-2026-11-02/
  piper-low-eval-2026-11-02-continued/
```

No Low INT8/PTQ, N100 benchmark or human MOS was collected in this session.
There is only one training seed. These observations do not replace the earlier
six-model N100 evidence.
