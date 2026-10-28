# Sequential-IR and Parallel-IR decoders: consolidated evidence and Q2-journal readiness

*Tiếng Việt: [q2-readiness-seq-par.vi.md](q2-readiness-seq-par.vi.md)*

Status on 2026-10-22. This document gathers, in one place, every stored result of the
HiFi-GAN ResBlock2 baseline, the Parallel-IR decoder (PAR, historical label MRF) and the
Sequential-IR decoder (SEQ). It then assesses what a journal paper on SEQ and PAR could
claim now, and what is missing.

- **Sources.** Every number comes from a record in this repository; the file is named under each table.
- **Not evidence.** Assessments and recommendations (sections 3–5) are the author-side reading of those records.
- **No new runs.** Nothing was trained, run or measured for this document.

## 1. Two generations of experiments

| | Generation 1 (historical, Aug–Sep 2026) | Generation 2 (controlled, Oct 2026) |
|---|---|---|
| Code base | BanhmiTTS, VITS2-based recipe | EdgeTTS Config A (Piper backbone, VITS2 components off) |
| Models | baseline-resblock2 (epoch 1489), Parallel-IR/MRF (1386), Sequential-IR/SEQ (1442), external Piper checkpoint | Piper_no_VITS2_cpn (ResBlock2, 1403), Piper-ParIR (1486), Piper-SeqIR (1455) |
| What changes between models | Decoder **and** split (MRF on another split, 95.4% shared training data), training code (SEQ adds non-finite-gradient skipping and SDP gating), gradient clipping (baseline 1.0, IR none), activation and MRD | Only the decoder bundle: IR blocks + SnakeBeta + MRD. Same data, split, seed, recipe and backbone initial weights |
| Records | `results/historical/`, `results/prior_audit/`, `docs/evidence/journal-evidence-package-2026-10-04/`, `docs/evidence/claim-evidence-2026-10-07/` | `results/piper-no-vits2-cpn/`, `results/piper-parir-no-vits2-cpn/`, `results/piper-seqir-no-vits2-cpn/`, `results/piper-ir-timing-session/`, `results/piper-ir-onnx-ptq/` |
| Weight as evidence | Confounded system-level comparison, one run per model | Controlled decoder-bundle comparison, one run per model |

**The decoder bundles** (both generations): inverted-residual blocks (pointwise expand → depthwise conv → pointwise project, expansion 1), SnakeBeta activations and an added MRD discriminator.
- **PAR:** three parallel branches per stage, depthwise kernels 3/5/7, averaged.
- **SEQ:** two serial blocks per stage, kernel 3.
- **What a comparison isolates:** the bundle changes blocks, activation and discriminator together. PAR versus SEQ also changes the block count and kernels, so neither comparison is topology-only.

## 2. Consolidated results

### 2.1 Harvard-720 (720 sentences, same protocol for all seven models)

Source: `results/README.md`, section "Seven-model comparison" (per-model files listed there).

| Model | Gen | Decoder parameters | UTMOSv2 (predicted) | Macro WER | Mean RTF (PyTorch CPU) | RTF vs ResBlock2 of the same group |
|---|---:|---:|---:|---:|---:|---:|
| Piper_no_VITS2_cpn (ResBlock2) | 2 | 1,664,768 | 3.517 | **0.0612** | 0.0510 | — |
| Piper-ParIR | 2 | 1,157,024 | **3.530** | 0.0726 | 0.0392 | −23.2% |
| Piper-SeqIR | 2 | **1,109,760** | 3.493 | 0.0696 | **0.0340** | **−33.3%** |
| baseline-resblock2 (with VITS2) | 1 | 1,664,768 | 3.026 | 0.1476 | 0.0418 | — |
| Parallel-IR (MRF) | 1 | 1,157,024 | **3.568** | 0.1073 | 0.0339 | −18.9% |
| Sequential-IR (SEQ) | 1 | **1,109,760** | 3.544 | 0.1222 | **0.0308** | **−26.2%** |
| Piper (external) | — | 1,664,768 | 3.451 | 0.1747 | 0.0392 | (−6.2%) |

Notes:
- **Decoder parameters.** All values are in weight-norm form, recounted from the checkpoints (`results/model_parameters.json`). Each decoder has the same size in both generations. J-E015 gives the same decoders with weight norm folded (ResBlock2 1,662,976, PAR 1,154,560, SEQ 1,107,968), about 0.1% fewer.
- **RTF groups.** The Gen 2 values come from one idle session on 2026-10-21. The Gen 1 values are the historical runs. The two groups cannot be compared with each other.
- **Decoder compute (Gen 1, J-E016).** SEQ 0.98, PAR 1.23, ResBlock2 and Piper 3.90 GFLOPs per second of audio.

**Paired results inside each generation** (sentence-level; CI = paired bootstrap 95%):

| Comparison | ΔUTMOSv2 | ΔWER (fraction) | ΔRTF | Source |
|---|---|---|---|---|
| Gen 2: ParIR − ResBlock2 | +0.013 [−0.011, +0.037]; same session +0.025 [+0.002, +0.049] | +0.011 [+0.002, +0.021] (Holm 0.11) | −23.2% (same session) | `results/piper-parir-no-vits2-cpn/`, `results/piper-ir-timing-session/` |
| Gen 2: SeqIR − ResBlock2 | −0.024 [−0.048, −0.00005]; same session −0.015 [−0.039, +0.009] | +0.008 [−0.0004, +0.018] (Holm 0.12) | −33.3% | `results/piper-seqir-no-vits2-cpn/` |
| Gen 2: SeqIR − ParIR | −0.037 [−0.060, −0.014]; same session −0.033 [−0.056, −0.010] | −0.003 [−0.013, +0.007] | −13.2% | same |
| Gen 1: SEQ − MRF | −0.024 (p 0.047; Holm 0.094) | +0.015 [+0.002, +0.028] | −9.1% (p 7e-37) | `results/prior_audit/`, ledger J-E003–J-E005 |
| Gen 1: SEQ − baseline | +0.518 | −0.026 | −26.2% | `results/prior_audit/` |
| Gen 1: SEQ − Piper | +0.094 | −0.053 | −21.3% | J-E007; Piper was fed mismatched (en-us) phonemes, see `results/frontend-check/` |

Validation mel loss (same 100 utterances, Gen 2 and the Gen 1 baseline only): ResBlock2 19.77, ParIR 19.01, SeqIR 19.03, baseline-resblock2 19.79.

### 2.2 LJ-500 tables (historical, Gen 1)

The historical LJ-500 tables use **three different 500-sentence sets**, so tables are comparable only within one set:
- **Baseline/canonical set** (`eval_true_testset_500_baseline.json`).
- **MRF set** (`eval_true_testset_500_mrf.json`): 21 sentences shared with the baseline set.
- **Overnight set:** 17 sentences shared with the baseline set.

Sources: `results/historical/{lj500,onnx_historical,lj500_development,qat}/`.

| Set | Model / table | Checkpoint | RTF | UTMOSv2 | WER |
|---|---|---|---:|---:|---:|
| baseline | baseline, PyTorch seeded (`eval_baseline_fp32_seeded_500`) | 1489 | 0.0434 | 3.125 | 0.1230 |
| baseline | baseline, PyTorch unseeded (`eval_baseline_best_500`) | 1489 | 0.0431 | 3.139 | 0.1197 |
| baseline | SEQ, PyTorch seeded (`eval_seq111_fp32_seeded_500`) | 1442 | 0.0324 | 3.581 | 0.1033 |
| baseline | Piper external (`eval_piper_original_500_full`) | 999 | 0.0386 | 3.534 | 0.1632 |
| MRF | MRF, PyTorch (`eval_mrf_final1500_500`) | 1386 | 0.0370 | 3.651 | 0.0983 |
| baseline | baseline, ONNX FP32 / INT8 (historical scope) | 1489 | 0.0388 / 0.0345 | 3.132 / 3.121 | 0.1230 / 0.1135 |
| MRF | MRF, ONNX FP32 / INT8 (**graph not linked to 1386**) | ? | 0.0416 / 0.0371 | 3.537 / 3.493 (rerun 3.490) | 0.1043 / 0.1030 (rerun 0.1124) |
| baseline | SEQ, ONNX FP32 / INT8 (**wider INT8 scope**) | 1442 | 0.0346 / 0.0289 | 3.571 / 3.586 | 0.1064 / 0.1081 |

**Training trajectory of MRF** (baseline set; `results/historical/lj500_development/`):

| Epoch | 74 | 216 | 353 | 480 | 836 | 836, evaluated again |
|---|---:|---:|---:|---:|---:|---:|
| UTMOSv2 | 2.187 | 3.169 | 3.297 | 3.348 | 3.505 | 3.518 |
| WER | 0.2552 | 0.1692 | 0.1013 | 0.1110 | 0.1024 | 0.1074 |

Two evaluations of the same MRF checkpoint differ by 0.013 in UTMOSv2 and 0.005 in WER. Two evaluations of the same baseline checkpoint differ by 0.003 WER. That is the size of the evaluation noise of these historical (partly unseeded) runs.

**SEQ precursors:**
- `mbconv_441` at epoch 12: WER 0.735 (an early, failed configuration).
- `mbconv_mrd` at epoch 74: WER 0.138.
- Their checkpoints no longer exist.

**QAT** (`results/historical/qat/`, LJ-500), fake-quantized flow, enc_p and dp, after further training:

| Model | Set | QAT epochs | RTF (PyTorch, fake-quant) | UTMOSv2 | WER |
|---|---|---:|---:|---:|---:|
| baseline QAT | baseline | 144 | 0.0497 / 0.0508 (seeded) | 3.569 / 3.549 | 0.0819 / 0.0786 |
| MRF QAT (from the 1000-epoch model) | baseline | 97 | 0.0456–0.0464 | 3.630–3.634 | 0.0996–0.1081 |
| MRF QAT (from the 1000-epoch model) | MRF | 97 | 0.0461 | 3.650 | 0.0977 |
| MRF QAT (from the 1500-epoch model) | MRF | 97 | 0.0456 | 3.639 | 0.0992 |

There is no QAT record for SEQ. The QAT gains cannot be attributed to quantization: QAT added 97–144 training epochs, and its RTF is PyTorch with fake-quant ops, not an INT8 runtime.

**Historical PTQ scope sweep** (`results/historical/ptq_sweep/`; the result files are derived copies without their Vocos entries):
- On the baseline, quantizing flow + enc_p + dp gave the lowest WER of the swept scopes (0.1215, 23.0 MB).
- **Caveat (J-E026):** that sweep's evaluator passed 22.05 kHz audio to Whisper (which expects 16 kHz) and used another WER normalization. Its WERs only rank scopes, and even that ranking is unvalidated.

### 2.3 ONNX FP32 and INT8, Gen 2 (clean protocol)

Source: `results/piper-ir-onnx-ptq-v2/` (round 2, flow weight norm folded; round 1 in `results/piper-ir-onnx-ptq/`). Harvard-720; matched noise; train-only calibration; the same INT8 scope for all models.

| Model | FP32 → INT8 size | ΔUTMOSv2 INT8 − FP32 | ΔWER INT8 − FP32 | ORT RTF, 2 threads (FP32 → INT8) | ORT RTF, 12 threads (FP32 → INT8) |
|---|---|---|---|---|---|
| ResBlock2 | 63.5 → 22.4 MB | −0.011 [−0.031, +0.008] | −0.003 [−0.009, +0.004] | 0.0323 → 0.0280 | 0.0253 → 0.0226 |
| ParIR | 61.5 → 20.5 MB | −0.007 [−0.025, +0.011] | +0.004 [−0.004, +0.012] | 0.0338 → 0.0306 | 0.0318 → 0.0290 |
| SeqIR | 61.3 → 20.3 MB | −0.023 [−0.043, −0.002] (Wilcoxon p 0.058) | −0.001 [−0.008, +0.006] | **0.0278 → 0.0238** | 0.0267 → 0.0243 |

- **FP32 export is lossless:** parity on 720/720 sentences, identical WER.
- **The speed ranking depends on the runtime and the thread count:**
  - ORT, 2 threads: SeqIR is fastest (−14% to −17% against ResBlock2 in round 2; −23% to −28% in round 1).
  - ORT, 12 threads: ResBlock2 is fastest; ParIR is 26–29% slower (round 1: 32–36%).
- **Cause (kernel profile):** SnakeBeta's `Sin` does not use ONNX Runtime 1.23's thread pool, while ResBlock2's dense convolutions do.

## 3. What the evidence supports today

| Claim | Status | Basis |
|---|---|---|
| IR decoders have 30–33% fewer decoder parameters and 3–4× fewer decoder FLOPs than ResBlock2; the whole model is about 3% smaller | **Supported** | parameter counts, J-E015/J-E016 |
| In PyTorch on CPU, SeqIR / ParIR cut RTF by 33% / 23% against ResBlock2 in a controlled setting | **Supported**, one session, one seed per model | Gen 2 timing session |
| SeqIR is faster than ParIR | **Supported** in PyTorch (Gen 1 −9%, Gen 2 −13%) and on ORT at 2 threads | both generations |
| IR decoders keep the quality of ResBlock2 | **Partial.** UTMOSv2 within 0.04 and WER within 1.1 points, but WER is slightly worse in both IR models, and no equivalence margin was set | Gen 2 paired tests |
| SEQ has better quality than PAR | **Contradicted.** SEQ has lower UTMOSv2 in Gen 1 and Gen 2; WER is not distinguishable | J-E004, J-E031, Gen 2 |
| The speed advantage holds after ONNX export and INT8 | **Only at low thread counts.** It holds at 2 ORT threads and reverses at 12 (cause identified) | `results/piper-ir-onnx-ptq-v2/`, `results/piper-ir-onnx-ptq/` |
| Selective INT8 PTQ (enc_p/dp/flow) costs no quality | **Mostly supported** (round 2, flow weight norm folded). No model changes WER detectably, and ResBlock2 and ParIR keep their UTMOSv2. SeqIR may lose 0.023 UTMOSv2: the bootstrap CI excludes 0, but Wilcoxon p is 0.058. Graphs are 65–67% smaller and 9–16% faster. One run per model; no equivalence margin | Gen 2 PTQ round 2, `results/piper-ir-onnx-ptq-v2/` |
| Removing the VITS2 components improves a Piper-size model | **Partial:** WER −8.6 points and UTMOSv2 +0.49, but the code bases also differ | `results/piper-no-vits2-cpn/` |
| Edge-device speed (N150, Raspberry Pi 5) | **Not available** | J-E028 |
| Human listening quality (MOS) | **Not available**; UTMOSv2 is a prediction | J-E029 |
| Robustness across training seeds | **Not available**: one run per model | — |

## 4. Gaps for a Q2 journal paper

Journals at this level usually expect, for a TTS or vocoder systems paper, human listening results, more than one training run or a variance estimate, controlled ablations for the claimed mechanism, comparisons with established lightweight systems, and measurements on the target hardware when deployment is the claim. Check the exact expectations against the chosen journal.

| Priority | Gap | What would close it | Rough cost |
|---|---|---|---|
| 1 | No human listening test | MOS or CMOS with confidence intervals on Gen 2 (ResBlock2, ParIR, SeqIR, INT8 variants, ground truth), pre-registered design, randomized and blind | 1–2 weeks of organization; ~20+ listeners |
| 1 | One training run per model | Two more seeds per Gen 2 model (6 runs) | ~5 days per 1500-epoch run on 2× RTX 4070 Ti: about 30 days on one machine, ~15 on two |
| 1 | Bundle not separated (blocks vs SnakeBeta vs MRD) | At least: IR blocks with LeakyReLU (no Snake), and ResBlock2 + SnakeBeta + MRD. The first also tests whether IR keeps its speed lead on ORT without `Sin` | 2 runs, ~10 days |
| 1 | No target-device measurements | N150 and Pi 5 (or the chosen devices): ORT 1–4 threads, warm-up, repetitions, peak memory, file size | 3–5 days once devices are available |
| 2 | No external lightweight baseline under the same protocol | e.g. Piper "low"/"x_low" voices, or an iSTFT-based VITS variant, scored with the same Harvard-720 pipeline | days, if checkpoints are public |
| 2 | Timing protocol | repeated, counterbalanced timing with warm-up; latency for short and long sentences | 1–2 days |
| 2 | In-domain test set for Gen 2 | canonical LJSpeech test-500 with the per-sentence seeding of Harvard-720 | ~1 day |
| 3 | Quality-parity wording | pre-specified equivalence (TOST) margins, or claims phrased as "within X with CI" | analysis only |
| 3 | Role of Gen 1 | supplementary history only, with its confounds stated; no main claim rests on it | writing |

## 5. Assessment

- **The data is not yet enough for a Q2 submission.** The strongest material is Gen 2: a controlled comparison, clean ONNX/INT8 protocol, determinism checks and an identified runtime mechanism. But it rests on one run per model and has no listening test or device data.
- **Gen 1 should not carry claims.** Its MRF/SEQ results are confounded (split, training code, clipping) and its ONNX/INT8 records have known defects. The ledger already marks the "SEQ better than PAR" quality claim and the historical "speed holds on INT8" claim as contradicted or unsupported.
- **A defensible paper story:**
  - lightweight inverted-residual decoders for Piper/VITS-class TTS on CPU;
  - −30% decoder parameters and −23% to −33% PyTorch RTF, at a quality cost of about 1 WER point;
  - selective INT8 PTQ adds −65% to −67% graph size and −9% to −16% ORT RTF, with no detectable WER change (round 2);
  - **deployment speed depends on runtime kernel coverage:** FLOPs and parameters did not predict ORT speed at 12 threads because SnakeBeta's `Sin` is single-threaded. This finding is concrete and measured, which makes it a contribution worth highlighting.
- **With the priority-1 items, a Q2-level systems or deployment paper is plausible, not guaranteed.** That means listening test, seeds, one or two ablations and device runs: about 4–6 weeks of compute plus the listening study. Novelty also has to be positioned against existing lightweight vocoder and TTS work through a literature review; this document does not assess it.

## 6. Where everything is

| Material | Location |
|---|---|
| Gen 1 per-sentence results, Harvard-720 and LJ-500 | `results/historical/harvard/`, `results/historical/lj500/` |
| Gen 1 development checkpoints, MRF trajectory, SEQ precursors, Snake/MRD variants | `results/historical/lj500_development/` (scripts in `records/`; logs referenced by hash in the import record) |
| Gen 1 QAT | `results/historical/qat/` |
| Gen 1 ONNX/INT8 and PTQ sweep | `results/historical/onnx_historical/`, `results/historical/ptq_sweep/` |
| Gen 1 statistics audit | `results/prior_audit/` |
| Gen 1 evidence package of 2026-10-04: claim ledger J-E001–J-E033, author notes including the earlier Q2 checklist and MRF/SEQ note, Q03 training history, Q05 export | `docs/evidence/journal-evidence-package-2026-10-04/` |
| Gen 1 Q01 archive catalog (27 result tables, provenance gaps) | `docs/evidence/q01-archive-2026-10-04/` |
| Claim evidence of 2026-10-07 (J-C claims) | `docs/evidence/claim-evidence-2026-10-07/` |
| Import record for the two folders above and `results/historical/{lj500_development,qat,ptq_sweep}` | `docs/evidence/historical-import-2026-10-22.json` |
| Gen 2 results, timing, ONNX/INT8, profile | `results/piper-*/`, `results/piper-ir-timing-session/`, `results/piper-ir-onnx-ptq-v2/`, `results/piper-ir-onnx-ptq/` |
| Parameter counts and sizes | `results/model_parameters.json` |
| Gen 2 protocols | `docs/training/piper-ir-protocol.md`, `docs/evaluation-onnx-ptq.md` |

Out of scope, because they concern other decoders:
- **Not imported:** the Vocos evaluation tables (`eval_true_testset_500`, `eval_last_ckpt_500`, `eval_full_500`) and the QAT report on BanhmiTTS_v1 (`QAT-Training/QAT_Quantization_Report.md`).
- **Vocos is not used anywhere.** Its PTQ-sweep logs are referenced by hash only, and the two PTQ-sweep result files are derived copies without their Vocos entries (`*.without_vocos.json`).
