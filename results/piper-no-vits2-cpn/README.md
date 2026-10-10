# Piper_no_VITS2_cpn: training outcome and Harvard-720 evaluation

The question this run serves (author, 2026-10-06) is whether the VITS2 components
make Piper better or worse. `Piper_no_VITS2_cpn` is the hifi-mobiNet baseline without
those components: EdgeTTS Config A, trained from scratch on the baseline's data and
split. The released `baseline-resblock2` is Piper with the VITS2 components.
Recipe and component audit: `docs/training/piper-component-comparison.md`.

All numbers below are **new measurements** (2026-10-10/11), except the four historical
Harvard result files they are compared with. Those files are unchanged.

## 1. Model evaluated

- **Run:** `training_output/piper-no-vits2-cpn` (outside Git).
  - Repository commit `d0c20d8`; 2× RTX 4070 Ti, DDP, bf16.
  - 1,500 epochs, finished 2026-10-10 21:58 +07.
  - `run_record.json` is reproduced in `training_summary.json`.
- **Selection rule:** fixed before training (`docs/training/next-run.md`): the lowest validation `val_loss_mel`.
  - Selected: **epoch 1403, `val_loss_mel` 19.7719**.
  - File: `best-epoch=1403-val_loss_mel=19.7719.ckpt`.
  - SHA-256 `2eeff5354c3320b12a288c0e52bdadbacce2c1ddaa14d5c739d1d4208b0f6003`.
  - Global step 1,100,736.
- **Loading:** strict, `weights_only=True`; 789 entries. The stored hyper-parameters equal `configs/training/piper-no-vits2-cpn.yaml`.
- **Test sets:** Harvard-720 and the test split were not used for selection.

## 2. Training curve against the baseline (validation `val_loss_mel`)

Both runs use the same 100 validation utterances and the same loss. Values come from the
TensorBoard scalars of each run (`training_summary.json`, per-epoch series included).

| Epochs | Piper_no_VITS2_cpn mean (SD) | baseline-resblock2 mean (SD) | Difference |
|---|---:|---:|---:|
| 1000–1099 | 20.364 (0.151) | 20.419 (0.216) | −0.056 |
| 1200–1299 | 20.224 (0.152) | 20.233 (0.220) | −0.009 |
| 1300–1399 | 20.160 (0.143) | 20.196 (0.178) | −0.036 |
| 1400–1499 | 20.105 (0.152) | 20.184 (0.216) | −0.078 |
| Best (epoch) | 19.7719 (1403) | 19.7882 (1489) | −0.016 |

The validation mel loss is almost the same for the two runs. Its epoch-to-epoch spread
(SD 0.15–0.22) is larger than the gap between them. The baseline numbers come from its
original run logs (four event files, resumed run); its selected checkpoint is epoch 1489.

## 3. Harvard-720 results

Protocol: the historical Harvard protocol, unchanged (`docs/evaluation.md`).
- PyTorch FP32 on CPU, 12 threads; scales 0.667/1.0/0.8; seed 1234 before every sentence.
- UTMOSv2 `fusion_stage3` fold 0 (weights SHA-256 `c8149d98…`), one prediction per WAV.
- Whisper `small` (`9ecf7799…`).
- JiWER macro WER.

The four other rows are the stored historical results (`results/historical/harvard/`).

| Model | UTMOSv2 (predicted) | Macro WER | Corpus WER | Sentences with WER 0 | Mean RTF |
|---|---:|---:|---:|---:|---:|
| **Piper_no_VITS2_cpn** (new) | **3.517** | **0.0612** | 0.0596 | 502 | 0.0421 * |
| baseline-resblock2 (with VITS2) | 3.026 | 0.1476 | 0.1437 | 318 | 0.0418 |
| Parallel-IR (MRF) | 3.568 | 0.1073 | 0.1033 | 381 | 0.0339 |
| Sequential-IR (SEQ) | 3.544 | 0.1222 | 0.1186 | 351 | 0.0308 |
| Piper (published checkpoint) | 3.451 | 0.1747 | 0.1696 | 228 | 0.0392 |

\* Measured in a different session from the historical rows; see section 4. Do not compare it directly.

Paired by sentence (`comparison.json`). The difference is new − reference. CIs are paired
bootstrap 95% intervals. Holm adjustment is over these 12 tests and exploratory.

| New vs | ΔUTMOSv2 [95% CI] | ΔWER [95% CI] | Sentences better / worse (WER) |
|---|---|---|---|
| baseline-resblock2 | **+0.491** [+0.465, +0.517] | **−0.086** [−0.099, −0.073] | 301 / 76 |
| Parallel-IR | −0.051 [−0.073, −0.029] | −0.046 [−0.057, −0.036] | 236 / 85 |
| Sequential-IR | −0.027 [−0.050, −0.005] (Holm p 0.13) | −0.061 [−0.073, −0.049] | 263 / 74 |
| Piper (published) | +0.066 [+0.042, +0.090] | −0.113 [−0.127, −0.100] | 399 / 67 |

WER is a fraction of words: −0.086 is 8.6 percentage points.

## 4. Controls run in the same session

- **Synthesis reproducibility (`session_control.json`).**
  - Re-running the original BanhmiTTS `synth_banhmi.py` (SHA-256 `23b2548d…`) on baseline epoch 1489 produced **720/720 WAVs byte-identical** to the historical Harvard WAVs.
  - The audio durations are also identical.
  - So the setup reproduces historical synthesis exactly.
  - The new model is deterministic too: sentence 0 had the same WAV hash in a smoke run and in the full run.
- **Timing drift.**
  - Today the same baseline took RTF 0.04524, against the stored 0.04177 (**+8.3%**).
  - RTF recorded on other days is therefore not comparable.
  - Back to back in this session, the new model's RTF was 0.04212 and the baseline's 0.04524.
  - That is **6.9% lower** for the new model: paired CI of the difference [−0.0037, −0.0025]; Wilcoxon p 2.7e-25; faster on 503/720 sentences.
- **Scorer reproducibility (`scorer_control.json`).** The baseline control WAVs were rescored with today's `score.py`.
  - Whisper gave **720/720 identical transcripts**, and macro WER 0.1476 equals the stored value.
  - UTMOSv2: mean 3.030 against 3.026 stored. Per-sentence differences have SD 0.22 and max 0.91, and r = 0.74, because random cropping is not seeded.
  - The WER and mean-UTMOSv2 differences in section 3 are therefore not scorer drift.

## 5. Reading the results

- **Against the baseline with VITS2:** the model without the VITS2 components scores clearly better on both quality measures.
  - Predicted UTMOSv2 +0.49; WER −8.6 points (0.148 → 0.061).
  - It is 6.9% faster in a same-session timing.
  - Validation mel loss is nearly equal.
  - In this experiment, adding the VITS2 package to Piper did **not** improve the model, and intelligibility (WER) was clearly worse with it.
- **Against Parallel-IR and Sequential-IR:** the new model has lower WER.
  - Its predicted UTMOSv2 is slightly lower than both. For SEQ the CI barely excludes 0, and the result is not significant after Holm.
  - Both decoders remain faster. For a cross-session comparison, scale the new model's 0.0421 by the 8.3% drift: that gives about 0.039, against 0.034 for MRF and 0.031 for SEQ.
  - MRF and SEQ were trained with the VITS2 baseline recipe: their exported flow carries the same attention, see `docs/evidence/claim-evidence-2026-10-07` (J-C025).
  - The new model also has the lowest WER of all five systems on Harvard-720.

What this does **not** show:
- **Which VITS2 component is responsible.** The two models also differ in codebase, as listed in the component audit: SDP guards, automatic versus manual optimization, and the fused AdamW kernel. This is not a single-component ablation. EdgeTTS Config A versus Config C would be the clean test.
- **That the result holds beyond this setup.** There is one training run per model, one synthesis draw per sentence, Whisper-small WER and predicted (not human) MOS.
- **That the difference is significant across training seeds.** Paired tests and CIs describe sentence-level variation only.

## 6. Not done in this round

- **Canonical test-500 evaluation:** not done.
  - The historical LJ500 protocol (`eval_*_seeded_500.py`) seeds once per run and scores inside the synthesis loop, so scorer randomness changes the next sentence's noise.
  - Reproducing it needs that loop, not the Harvard scripts.
  - A cleaner design: synthesize both checkpoints in one session with per-sentence seeding, then score both with the same scorer pass.
- **ONNX export, PTQ and edge timing:** not done.

## 7. Files

| File | Content | Class |
|---|---|---|
| `training_summary.json` | Run record, checkpoint hashes, per-epoch `val_loss_mel` for both runs, window statistics | new summary of run logs |
| `harvard720_synth_record.json` | Synthesis record: checkpoint/config/manifest hashes, protocol, environment, per-sentence timing and WAV SHA-256. It is the `synth.json` named in `score_record.json` (same SHA-256 `72bee8b4…`) | new measurement |
| `harvard_piper_no_vits2_cpn_results.json` | Per-sentence results, historical schema plus `wav_sha256` | new measurement |
| `score_record.json` | Scorer packages, weight hashes, environment, results hash | new measurement record |
| `comparison.json` | Summaries of all five systems and the 12 paired tests | new analysis |
| `session_control.json` | Synthesis reproducibility, timing drift, same-session RTF comparison | new control |
| `harvard_baseline_rescored_results.json`, `baseline_rescore_record.json` | Today's rescoring of the baseline control WAVs. The synthesis record they cite is kept with the run because it contains host paths | new control |
| `scorer_control.json` | Rescored versus stored baseline scores | new control |

The WAVs (720 new and 720 control) are outside Git, under the run directory and the
workstation's historical WAV area. Their hashes are in the records.

## 8. Reproduction

From the training clone root (training venv for synthesis; scorer environment with
UTMOSv2, Whisper and JiWER for scoring; any environment with SciPy/JiWER for analysis):

```sh
python scripts/evaluation/harvard/synthesize.py --config configs/training/piper-no-vits2-cpn.yaml \
  --checkpoint "training_output/piper-no-vits2-cpn/checkpoints/best-epoch=1403-val_loss_mel=19.7719.ckpt" \
  --manifest results/manifests/harvard720.json \
  --wav-dir training_output/piper-no-vits2-cpn/eval/harvard720/wavs --output training_output/piper-no-vits2-cpn/eval/harvard720/synth.json
python scripts/evaluation/harvard/score.py --synth training_output/piper-no-vits2-cpn/eval/harvard720/synth.json \
  --output <dir>/harvard_piper_no_vits2_cpn_results.json --record <dir>/score_record.json
python scripts/evaluation/harvard/compare.py --candidate results/piper-no-vits2-cpn/harvard_piper_no_vits2_cpn_results.json \
  --label piper_no_vits2_cpn --reference-dir results/historical/harvard --output <dir>/comparison.json
python scripts/evaluation/harvard/scorer_control.py --rescored results/piper-no-vits2-cpn/harvard_baseline_rescored_results.json \
  --historical results/historical/harvard/harvard_baseline_results.json --output <dir>/scorer_control.json
python scripts/training/summarize_run.py --run-dir training_output/piper-no-vits2-cpn \
  --reference-logdir <baseline run>/lightning_logs --reference-label baseline-resblock2 \
  --reference-best-checkpoint "best-epoch=1489-val_loss_mel=19.7882.ckpt" --output <dir>/training_summary.json
```

**Expected outputs:**
- Synthesis is deterministic, so the WAV hashes should match.
- Whisper transcripts reproduced exactly in the control.
- UTMOSv2 per-sentence scores vary between scoring sessions.

**Session control:** the original `synth_banhmi.py` was run with label `baseline_ctrl_20261010`. Then `session_control.py` compared its WAVs and timing with the historical baseline.
