# Piper_ParIR_no_VITS2_cpn: training outcome and Harvard-720 quality evaluation

*Tiếng Việt: [README.vi.md](README.vi.md)*

The question (protocol: `docs/training/piper-ir-protocol.md`, set up 2026-10-11) is whether
replacing the ResBlock2 decoder of `Piper_no_VITS2_cpn` (Piper-ConfigA) with the released
Parallel-IR decoder bundle gives a better model. **Piper-ParIR** keeps the
Config A backbone, data, split and recipe. It changes three things:

- three parallel inverted-residual branches per stage (depthwise kernels 3/5/7, expansion 1);
- SnakeBeta activations;
- an added MRD discriminator.

This is a decoder-bundle comparison, not a topology-only one.

All numbers below are **new measurements** (2026-10-16), except the stored results they
are compared with. **RTF was not measured in this round**; see section 6.

## 1. Model evaluated

- **Run:** `training_output/piper-parir-no-vits2-cpn` (outside Git).
  - Repository commit `9f0988b` (clean); 2× RTX 4070 Ti, DDP, bf16.
  - 1,500 epochs from scratch, seed 1234, no gradient clipping.
  - Started 2026-10-11 20:30 +07; finished 2026-10-16 13:52 +07.
- **One interruption.** On 2026-10-15 ~13:57 +07 the Windows host reset without a clean shutdown (Kernel-Power 41, bugcheck 0) during epoch 1222.
  - Before the resume, `last.ckpt` (epoch 1221, global step 958,048) was verified: zip CRC, finite tensors, readable events.
  - The run was resumed from it by the controller (`run_sequence.py resume`, `resume_record_*.json`).
  - Under DDP the RNG state is not restored, by design. From epoch 1222 on, data order and noise therefore differ from an uninterrupted run.
- **Validation by the controller:** all 8 checks passed (`validation_parir.json`):
  - exit code 0, budget reached, no traceback;
  - consistent run record;
  - 1,500 `val_loss_mel` points, all scalars finite;
  - `last.ckpt` at epoch 1,499;
  - selected checkpoint equals the minimum.
- **Selection rule** (fixed before training): the lowest validation `val_loss_mel`.
  - Selected: **epoch 1486, `val_loss_mel` 19.0069**.
  - File: `best-epoch=1486-val_loss_mel=19.0069.ckpt`.
  - SHA-256 `d57f6abb589a28db83dc60d4f2b7d0c0c4a5a16db9ae124e12f40870d31c97ee`.
  - Global step 1,165,808.
- **Loading:** strict, `weights_only=True`; 917 entries. The stored hyper-parameters equal `configs/training/piper-parir-no-vits2-cpn.yaml`.
- **Held-out data:** Harvard-720 and the test split were not used for selection.
  - One informal listening session on 2026-10-14 used Harvard list 1 (idx 0–9) with intermediate checkpoints (epochs 645/664/717). It is recorded outside Git in `outputs/Piper_ParIR_no_VITS2_cpn/`.
  - No training or selection decision was based on it.

## 2. Training curve against Piper_no_VITS2_cpn (validation `val_loss_mel`)

Same 100 validation utterances and the same loss. Values come from the TensorBoard scalars
(`training_summary.json`, per-epoch series included; the two event files of the resumed run
are merged by global step).

| Epochs | Piper-ParIR mean (SD) | Piper_no_VITS2_cpn mean (SD) | Difference |
|---|---:|---:|---:|
| 500–599 | 19.925 (0.177) | 20.796 (0.187) | −0.872 |
| 1000–1099 | 19.828 (0.181) | 20.364 (0.151) | −0.535 |
| 1200–1299 | 19.466 (0.164) | 20.224 (0.152) | −0.758 |
| 1300–1399 | 19.404 (0.162) | 20.160 (0.143) | −0.756 |
| 1400–1499 | 19.317 (0.143) | 20.105 (0.152) | −0.788 |
| Best (epoch) | 19.0069 (1486) | 19.7719 (1403) | −0.765 |

ParIR's validation mel loss is lower by about 0.75–0.79 from epoch 1200 on. That is about
five times the epoch-to-epoch SD.

From epoch ~800 its train and validation mel loss rose by about 0.3–0.4:
- the 20-epoch mean went from 19.71 (780–799) to about 20.1 (800–860);
- the 900–999 mean was still 19.97;
- it fell below the pre-800 level only around epochs 1100–1200.

At the start of the rise, the MRD discriminator loss fell (from ~1.02 to ~0.96 over epochs
800–860), and the generator–discriminator balance shifted. The ResBlock2 run has no MRD
and shows no such step.

The mel loss is a reconstruction loss. Section 3 shows that the gap does not carry over
to the Harvard quality measures.

## 3. Harvard-720 results

Protocol: the historical Harvard protocol, unchanged (`docs/evaluation.md`), same as for Piper_no_VITS2_cpn.
- PyTorch FP32 on CPU, 12 threads; scales 0.667/1.0/0.8; seed 1234 before every sentence.
- UTMOSv2 `fusion_stage3` fold 0 (weights SHA-256 `c8149d98…`), one prediction per WAV.
- Whisper `small` (`9ecf7799…`).
- JiWER macro WER.
- Scorer environment equal to the 2026-10-10 one: Python 3.10.20, torch 2.13.0+cu130, NumPy 1.26.4.

Synthesis and scoring ran on CPU while Piper-SeqIR was training on the GPUs. Synthesis is
deterministic, and the scores do not depend on machine load, but the timing does. **RTF
from this session is therefore not reported.**

The other five rows are stored results:
- Piper_no_VITS2_cpn from `results/piper-no-vits2-cpn/` (2026-10-10);
- four historical systems from `results/historical/harvard/`.

| Model | UTMOSv2 (predicted) | Macro WER | Corpus WER | Sentences with WER 0 | Mean RTF |
|---|---:|---:|---:|---:|---:|
| **Piper-ParIR** (new) | **3.530** | **0.0726** | 0.0707 | 464 | not measured |
| Piper_no_VITS2_cpn (ResBlock2, 2026-10-10) | 3.517 | 0.0612 | 0.0596 | 502 | 0.0421 * |
| baseline-resblock2 (with VITS2) | 3.026 | 0.1476 | 0.1437 | 318 | 0.0418 |
| Parallel-IR (MRF, released) | 3.568 | 0.1073 | 0.1033 | 381 | 0.0339 |
| Sequential-IR (SEQ, released) | 3.544 | 0.1222 | 0.1186 | 351 | 0.0308 |
| Piper (published checkpoint) | 3.451 | 0.1747 | 0.1696 | 228 | 0.0392 |

\* Different session from the historical rows; see `results/piper-no-vits2-cpn/README.md`, section 4.

Word errors over the 5,742 reference words (substitutions / deletions / insertions):
- Piper-ParIR: 321 / 71 / 14;
- Piper_no_VITS2_cpn: 283 / 46 / 13.

Paired by sentence (`comparison.json`). The difference is ParIR − reference, and the CIs
are paired bootstrap 95% intervals. The Holm adjustment is over these 10 tests (UTMOSv2
and WER only) and is exploratory.

| ParIR vs | ΔUTMOSv2 [95% CI] (Holm p) | ΔWER [95% CI] (Holm p) | Sentences better / worse (WER) |
|---|---|---|---|
| **Piper_no_VITS2_cpn** | +0.013 [−0.011, +0.037] (0.43) | **+0.011** [+0.002, +0.021] (0.11) | 108 / 159 |
| baseline-resblock2 | **+0.504** [+0.479, +0.530] | **−0.075** [−0.088, −0.062] | 286 / 99 |
| Parallel-IR (MRF) | −0.038 [−0.060, −0.016] (0.003) | −0.035 [−0.046, −0.024] | 225 / 122 |
| Sequential-IR (SEQ) | −0.014 [−0.037, +0.008] (0.43) | −0.050 [−0.062, −0.037] | 252 / 112 |
| Piper (published) | +0.079 [+0.056, +0.103] | −0.102 [−0.116, −0.089] | 382 / 84 |

Holm p is shown where it is above 0.001. WER is a fraction of words: +0.011 is
1.1 percentage points.

**Same-session UTMOSv2 against Piper_no_VITS2_cpn** (supplementary, `comparison_same_session_utmosv2.json`):
- UTMOSv2 varies between scoring sessions (section 4), so the Piper_no_VITS2_cpn WAVs were rescored in this session.
- Against those scores, ParIR is **+0.025 [+0.002, +0.049]** (Wilcoxon p 0.097), higher on 368 and lower on 352 sentences.
- WER is the same in both scorings, so the WER comparison above is unaffected.

## 4. Controls

- **Scorer reproducibility (`scorer_control.json`).** The 2026-10-10 Piper_no_VITS2_cpn WAVs were rescored with the same `score.py` in this session. `score.py` checks every WAV's SHA-256 before scoring.
  - Whisper gave **720/720 identical transcripts**; macro WER 0.0612 equals the stored value.
  - UTMOSv2: mean 3.505 against 3.517 stored (−0.012). Per-sentence differences have SD 0.16 and max 0.62, and r = 0.80, because random cropping is not seeded.
  - The ΔWER above is therefore not scorer drift. A ΔUTMOSv2 of about ±0.01–0.02 between sessions is within scorer variation.
- **Synthesis:** 720/720 WAVs finite. WAV SHA-256 values are in `harvard720_synth_record.json`. A determinism re-run of this checkpoint is planned for the RTF session (section 6).

## 5. Reading the results

- **Against Piper_no_VITS2_cpn** (same backbone, ResBlock2 decoder): on Harvard-720 the two models are close.
  - **UTMOSv2:** ParIR is slightly higher. Cross-session the difference is +0.013 and its CI includes 0. In the same session it is +0.025 and the CI just excludes 0, but Wilcoxon p is 0.097.
  - **WER:** ParIR is slightly worse, 0.073 against 0.061 (+1.1 points). The CI excludes 0 (Wilcoxon p 0.037), but the result is not significant after Holm (0.11). ParIR is worse on 159 sentences and better on 108. Most of the extra errors are deletions (71 vs 46).
  - **Validation mel loss:** ParIR is clearly lower (−0.77). That gain did not turn into better intelligibility or clearly better predicted quality.
  - **Size:** the ParIR decoder has 1.16 M parameters against 1.66 M (weight-norm form), 30% fewer.
  - **Speed:** whether ParIR is faster is not known yet (section 6).
- **Against the four stored systems:**
  - ParIR is clearly better than baseline-resblock2 (with VITS2) and the published Piper checkpoint on both measures.
  - Against the released Parallel-IR (MRF), which has the same decoder bundle on the VITS2 baseline recipe, ParIR has lower WER (−3.5 points) and slightly lower UTMOSv2 (−0.038).
  - Against Sequential-IR, ParIR has lower WER, and UTMOSv2 is not distinguishable.
- **Ranking by WER:** Piper_no_VITS2_cpn has the lowest WER of the six systems, and Piper-ParIR is second.

What this does **not** show:
- **Which part of the bundle matters.** The decoder blocks, SnakeBeta and the added MRD cannot be separated in this run (protocol, section 4).
- **That the result holds beyond this setup.** There is one training run per model with seed 1234, one synthesis draw per sentence, Whisper-small WER and predicted (not human) MOS. The run also had one DDP resume without RNG restore.
- **Significance across training seeds.** Paired tests describe sentence-level variation only.

## 6. Not done in this round

- **RTF.** Planned on an idle machine after Piper-SeqIR finishes. ParIR, SeqIR and a Piper_no_VITS2_cpn control will be timed back to back in one session (`session_control.py`), and the ParIR WAVs will be re-synthesized to confirm determinism.
- **Canonical test-500, ONNX export, PTQ and edge timing:** not done (same reasons as in `results/piper-no-vits2-cpn/README.md`, section 6).

## 7. Files

| File | Content | Class |
|---|---|---|
| `selected_checkpoint.json` | Controller's selection: file, bytes, SHA-256, epoch, global step, `val_loss_mel`, rule | run record |
| `training_summary.json` | Run record, checkpoint hashes, per-epoch `val_loss_mel` for ParIR and Piper_no_VITS2_cpn, window statistics | new summary of run logs |
| `harvard720_synth_record.json` | Synthesis record: checkpoint/config/manifest hashes, protocol, environment, per-sentence WAV SHA-256. Its timing fields were recorded under training load and are not used | new measurement |
| `harvard_piper_parir_no_vits2_cpn_results.json` | Per-sentence results, historical schema plus `wav_sha256`. The `rtf` field is from the loaded session; do not use it | new measurement |
| `score_record.json` | Scorer packages, weight hashes, environment, results hash | new measurement record |
| `comparison.json` | Summaries of all six systems and the 10 paired tests (UTMOSv2, WER) | new analysis |
| `comparison_same_session_utmosv2.json` | Supplementary UTMOSv2 tests including the same-session rescoring of Piper_no_VITS2_cpn | new analysis |
| `harvard_piper_no_vits2_cpn_rescored_results.json`, `piper_no_vits2_cpn_rescore_record.json` | Rescoring of the 2026-10-10 Piper_no_VITS2_cpn WAVs in this session | new control |
| `scorer_control.json` | Rescored versus stored Piper_no_VITS2_cpn scores | new control |

The 720 WAVs are outside Git, in `training_output/piper-parir-no-vits2-cpn/eval/harvard720/wavs`.
Their hashes are in the records.

## 8. Reproduction

Run from the training clone root:
- synthesis with the training venv;
- scoring with the scorer environment used on 2026-10-10 (UTMOSv2, Whisper, JiWER; versions in `score_record.json`);
- analysis in any environment with SciPy.

`compare.py --extra-reference/--metrics` were added on 2026-10-16 for this comparison.

```sh
python scripts/evaluation/harvard/synthesize.py --config configs/training/piper-parir-no-vits2-cpn.yaml \
  --checkpoint "training_output/piper-parir-no-vits2-cpn/checkpoints/best-epoch=1486-val_loss_mel=19.0069.ckpt" \
  --manifest results/manifests/harvard720.json \
  --wav-dir training_output/piper-parir-no-vits2-cpn/eval/harvard720/wavs \
  --output training_output/piper-parir-no-vits2-cpn/eval/harvard720/synth.json
python scripts/evaluation/harvard/score.py --synth training_output/piper-parir-no-vits2-cpn/eval/harvard720/synth.json \
  --output <dir>/harvard_piper_parir_no_vits2_cpn_results.json --record <dir>/score_record.json
python scripts/evaluation/harvard/score.py --synth training_output/piper-no-vits2-cpn/eval/harvard720/synth.json \
  --output <dir>/harvard_piper_no_vits2_cpn_rescored_results.json --record <dir>/piper_no_vits2_cpn_rescore_record.json
python scripts/evaluation/harvard/scorer_control.py --rescored <dir>/harvard_piper_no_vits2_cpn_rescored_results.json \
  --historical results/piper-no-vits2-cpn/harvard_piper_no_vits2_cpn_results.json --output <dir>/scorer_control.json
python scripts/evaluation/harvard/compare.py --candidate <dir>/harvard_piper_parir_no_vits2_cpn_results.json \
  --label piper_parir_no_vits2_cpn --reference-dir results/historical/harvard \
  --extra-reference piper_no_vits2_cpn=results/piper-no-vits2-cpn/harvard_piper_no_vits2_cpn_results.json \
  --metrics utmosv2,wer --output <dir>/comparison.json
python scripts/evaluation/harvard/compare.py --candidate <dir>/harvard_piper_parir_no_vits2_cpn_results.json \
  --label piper_parir_no_vits2_cpn --reference-dir results/historical/harvard \
  --extra-reference piper_no_vits2_cpn_rescored=<dir>/harvard_piper_no_vits2_cpn_rescored_results.json \
  --metrics utmosv2 --output <dir>/comparison_same_session_utmosv2.json
python scripts/training/summarize_run.py --run-dir training_output/piper-parir-no-vits2-cpn \
  --reference-logdir training_output/piper-no-vits2-cpn/lightning_logs --reference-label piper-no-vits2-cpn \
  --reference-best-checkpoint "best-epoch=1403-val_loss_mel=19.7719.ckpt" --output <dir>/training_summary.json
```

**Expected outputs:**
- Synthesis is deterministic, so the WAV hashes should match.
- Whisper transcripts reproduced exactly in the control.
- UTMOSv2 per-sentence scores vary between scoring sessions.

The pipeline that ran these steps is kept with the run (`eval/pipeline.sh`, `eval/pipeline.log`).
