# Piper_SeqIR_no_VITS2_cpn: training outcome, Harvard-720 quality and same-session RTF

*Tiếng Việt: [README.vi.md](README.vi.md)*

The question (protocol: `docs/training/piper-ir-protocol.md`, set up 2026-10-11) is whether
replacing the ResBlock2 decoder of `Piper_no_VITS2_cpn` (Piper-ConfigA) with the released
Sequential-IR decoder bundle gives a better model. **Piper-SeqIR** keeps the Config A
backbone, data, split and recipe. It changes three things:

- two sequential inverted-residual blocks per stage (depthwise kernel 3, expansion 1);
- SnakeBeta activations;
- an added MRD discriminator.

This is a decoder-bundle comparison, not a topology-only one. Against Piper-ParIR it is not
topology-only either: besides sequential versus parallel, the number of blocks per stage
(2 vs 3) and the depthwise kernels (3 vs 3/5/7) differ.

All numbers below are **new measurements** (2026-10-21), except the stored results they are
compared with. This round also contains the **idle-machine timing session** that was
deferred in the ParIR report: Piper_no_VITS2_cpn (control), Piper-ParIR and Piper-SeqIR were
timed back to back (section 4). Its records are in
[`../piper-ir-timing-session/`](../piper-ir-timing-session/).

## 1. Model evaluated

- **Run:** `training_output/piper-seqir-no-vits2-cpn` (outside Git).
  - Repository commit `9f0988b` (clean); 2× RTX 4070 Ti, DDP, bf16.
  - 1,500 epochs from scratch, seed 1234, no gradient clipping.
  - Started 2026-10-16 13:55 +07; the trainer exited on 2026-10-21 07:55 +07, and the controller marked the run completed at 07:58.
- **No interruption.** The controller status has one start attempt and no resume record; the run wrote one TensorBoard event file.
- **Validation by the controller:** all 8 checks passed (`validation_seqir.json`, kept with the controller state):
  - exit code 0, budget reached, no traceback;
  - consistent run record, no resume records;
  - 1,500 `val_loss_mel` points, all scalars finite;
  - `last.ckpt` at epoch 1,499, global step 1,176,000 (as expected);
  - selected checkpoint equals the minimum.
- **Selection rule** (fixed before training): the lowest validation `val_loss_mel`.
  - Selected: **epoch 1455, `val_loss_mel` 19.0332**.
  - File: `best-epoch=1455-val_loss_mel=19.0332.ckpt`.
  - SHA-256 `cc1ba70d20471fe587d0099fc9d1c739bc0d8d3b08541450629eb8bb068c6637`.
  - Global step 1,141,504.
  - Also kept: epochs 1382 (19.1521) and 1491 (19.0770), and `last.ckpt`.
- **Loading:** strict, `weights_only=True`; 878 entries. `synthesize.py` stops if the stored hyper-parameters differ from `configs/training/piper-seqir-no-vits2-cpn.yaml`; they did not.
- **Held-out data:** Harvard-720 and the test split were not used for selection.

## 2. Training curve (validation `val_loss_mel`)

Same 100 validation utterances and the same loss for all three runs. Values come from the
TensorBoard scalars (`training_summary.json` for SeqIR and Piper_no_VITS2_cpn;
`../piper-parir-no-vits2-cpn/training_summary.json` for ParIR).

| Epochs | Piper-SeqIR mean (SD) | Piper-ParIR mean (SD) | Piper_no_VITS2_cpn mean (SD) | SeqIR − no_VITS2 | SeqIR − ParIR |
|---|---:|---:|---:|---:|---:|
| 500–599 | 20.318 (0.172) | 19.925 (0.177) | 20.796 (0.187) | −0.479 | +0.393 |
| 1000–1099 | 19.753 (0.180) | 19.828 (0.181) | 20.364 (0.151) | −0.610 | −0.075 |
| 1200–1299 | 19.553 (0.155) | 19.466 (0.164) | 20.224 (0.152) | −0.671 | +0.087 |
| 1300–1399 | 19.507 (0.145) | 19.404 (0.162) | 20.160 (0.143) | −0.652 | +0.103 |
| 1400–1499 | 19.441 (0.147) | 19.317 (0.143) | 20.105 (0.152) | −0.664 | +0.124 |
| Best (epoch) | 19.0332 (1455) | 19.0069 (1486) | 19.7719 (1403) | −0.739 | +0.026 |

- **Against Piper_no_VITS2_cpn:** from epoch 1200 on, SeqIR's validation mel loss is lower by about 0.65–0.67, about four times the epoch-to-epoch SD.
- **Against ParIR:** SeqIR is higher by 0.09–0.12 over the last 300 epochs, less than one epoch-to-epoch SD.
- **No step around epoch 800.** ParIR's loss rose by about 0.3 from epoch ~800 (ParIR README, section 2). SeqIR, which also has the MRD, kept falling. Its 40-epoch means were:

  | Epochs | 760–799 | 800–839 | 840–879 | 880–919 |
  |---|---:|---:|---:|---:|
  | SeqIR | 20.03 | 20.02 | 19.98 | 19.89 |
  | ParIR | 19.74 | 20.05 | 20.07 | 20.04 |

  The 1000–1099 window, where SeqIR is below ParIR, falls inside ParIR's step. The presence of the MRD alone therefore does not explain that step. Its cause is not established, and SeqIR's discriminator losses were not inspected for this report.

## 3. Harvard-720 results

Protocol: the historical Harvard protocol, unchanged (`docs/evaluation.md`), as for the
other two Config A models.
- PyTorch FP32 on CPU, 12 threads; scales 0.667/1.0/0.8; seed 1234 before every sentence.
- UTMOSv2 `fusion_stage3` fold 0 (weights SHA-256 `c8149d98…`), one prediction per WAV.
- Whisper `small` (`9ecf7799…`).
- JiWER macro WER.
- Scorer environment equal to the earlier rounds: Python 3.10.20, torch 2.13.0+cu130, NumPy 1.26.4.

SeqIR was synthesized in the idle timing session (section 4) and scored afterwards on CPU,
with no training running.

The other six rows are stored results:
- Piper-ParIR from `results/piper-parir-no-vits2-cpn/` (2026-10-16);
- Piper_no_VITS2_cpn from `results/piper-no-vits2-cpn/` (2026-10-10);
- four historical systems from `results/historical/harvard/`.

RTF is given only for the three Config A models, from the 2026-10-21 session. The
historical RTF values come from other sessions and are not comparable (section 4).

| Model | UTMOSv2 (predicted) | Macro WER | Corpus WER | Sentences with WER 0 | Mean RTF (2026-10-21 session) |
|---|---:|---:|---:|---:|---:|
| **Piper-SeqIR** (new) | **3.493** | **0.0696** | 0.0665 | 476 | **0.0340** |
| Piper-ParIR (2026-10-16) | 3.530 | 0.0726 | 0.0707 | 464 | 0.0392 |
| Piper_no_VITS2_cpn (ResBlock2, 2026-10-10) | 3.517 | 0.0612 | 0.0596 | 502 | 0.0510 |
| baseline-resblock2 (with VITS2) | 3.026 | 0.1476 | 0.1437 | 318 | — |
| Parallel-IR (MRF, released) | 3.568 | 0.1073 | 0.1033 | 381 | — |
| Sequential-IR (SEQ, released) | 3.544 | 0.1222 | 0.1186 | 351 | — |
| Piper (published checkpoint) | 3.451 | 0.1747 | 0.1696 | 228 | — |

Word errors over the 5,742 reference words (substitutions / deletions / insertions):
- Piper-SeqIR: 292 / 76 / 14;
- Piper-ParIR: 321 / 71 / 14;
- Piper_no_VITS2_cpn: 283 / 46 / 13.

Paired by sentence (`comparison.json`). The difference is SeqIR − reference, and the CIs
are paired bootstrap 95% intervals. The Holm adjustment is over these 12 tests (UTMOSv2
and WER only) and is exploratory.

| SeqIR vs | ΔUTMOSv2 [95% CI] (Holm p) | ΔWER [95% CI] (Holm p) | Sentences better / worse (WER) |
|---|---|---|---|
| **Piper_no_VITS2_cpn** | −0.024 [−0.048, −0.00005] (0.12) | +0.008 [−0.0004, +0.018] (0.12) | 116 / 140 |
| **Piper-ParIR** | −0.037 [−0.060, −0.014] (0.021) | −0.003 [−0.013, +0.007] (0.71) | 144 / 128 |
| baseline-resblock2 | **+0.467** [+0.439, +0.495] | **−0.078** [−0.091, −0.065] | 290 / 84 |
| Parallel-IR (MRF) | −0.075 [−0.097, −0.052] | −0.038 [−0.049, −0.026] | 220 / 113 |
| Sequential-IR (SEQ) | −0.051 [−0.075, −0.028] | −0.053 [−0.064, −0.041] | 248 / 96 |
| Piper (published) | +0.042 [+0.016, +0.068] (0.003) | −0.105 [−0.119, −0.091] | 380 / 80 |

Holm p is shown where it is above 0.001. WER is a fraction of words: +0.008 is
0.8 percentage points.

**Same-session UTMOSv2** (supplementary, `comparison_same_session_utmosv2.json`, Holm over its 6 tests):
- UTMOSv2 varies between scoring sessions (section 5). The stored Piper_no_VITS2_cpn and ParIR WAVs were therefore rescored in this session, right after SeqIR.
- Against Piper_no_VITS2_cpn rescored (3.508): **−0.015 [−0.039, +0.009]**, Wilcoxon p 0.23; higher on 344 and lower on 376 sentences.
- Against ParIR rescored (3.526): **−0.033 [−0.056, −0.010]**, Wilcoxon p 0.013 (Holm 0.025); higher on 330 and lower on 390 sentences.
- WER is the same in both scorings, so the WER comparisons above are unaffected.

## 4. Same-session RTF (idle machine)

**Setup.**
- **When:** the session started at 09:05 +07 on 2026-10-21, after the controller had marked SeqIR completed and no trainer process was left. It ended at 09:09:36.
- **What:** three models were synthesized back to back, each in a fresh process of `synthesize.py`, through the same harness and EdgeTTS code:
  1. Piper_no_VITS2_cpn (control, epoch 1403);
  2. Piper-ParIR (epoch 1486);
  3. Piper-SeqIR (epoch 1455).
- **How:** CPU FP32, 12 threads, normal priority, CUDA hidden. The timed region is the forward call only. There was one pass per sentence and no warm-up.
- **Machine state before** (`machine_state_before.txt`): load average 0.03; GPU utilization 6% and 0%; the busiest process was TensorBoard at 0.2% CPU.

| Model | Mean RTF | Median RTF | Δ mean vs control [95% CI] | Relative | Faster than control | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
| Piper_no_VITS2_cpn (control) | 0.0510 | 0.0490 | — | — | — | — |
| Piper-ParIR | 0.0392 | 0.0374 | −0.0118 [−0.0126, −0.0110] | **−23.2%** | 653 / 720 | 4e-99 |
| Piper-SeqIR | 0.0340 | 0.0320 | −0.0170 [−0.0178, −0.0162] | **−33.3%** | 690 / 720 | 2e-113 |

The control comparisons are in `session_control_parir.json` and `session_control_seqir.json`.

**SeqIR against ParIR** in the same session: the mean RTF is 13.2% lower (0.0340 / 0.0392 = 0.868), and SeqIR is faster on 561 of 720 sentences. This is plain arithmetic on the two synthesis records; no paired test file was produced for this pair.

**Checks in the same session:**
- **Control synthesis:** 720/720 WAVs byte-identical to the 2026-10-10 Piper_no_VITS2_cpn WAVs, and the durations are equal.
- **ParIR re-synthesis:** 720/720 WAV SHA-256 identical to the 2026-10-16 evaluation, and the durations are identical (`parir_determinism.json`). The ParIR quality scores of 2026-10-16 therefore describe the same audio that was timed here.
- **SeqIR synthesis:** done once; 720/720 WAVs are finite. Determinism was not checked by a second synthesis.
- **Session drift:** the control's mean RTF was 0.0510 here, against 0.0421 in its own 2026-10-10 session (×1.21), although the machine was idle. The cause was not recorded: the Windows host's power and thermal state and its load outside WSL were not logged.

**Limits.**
- The RTF values are comparable only within this session. The historical values (MRF 0.0339, SEQ 0.0308, baseline 0.0418, Piper 0.0392) and the 2026-10-10 value 0.0421 come from other sessions; do not put them beside 0.0340.
- The order (control → ParIR → SeqIR) was fixed, not counterbalanced. There is one pass per sentence and no repetition.
- This is PyTorch on CPU under WSL on a workstation. It is not ONNX and not edge-device timing.
- The models generate different audio lengths (total 1,749 s control, 1,798 s ParIR, 1,744 s SeqIR); RTF normalizes per sentence.

## 5. Scorer controls

- **Piper_no_VITS2_cpn WAVs rescored (`scorer_control.json`):**
  - Whisper gave **720/720 identical transcripts**; macro WER 0.0612 equals the stored value.
  - UTMOSv2: mean 3.508 against 3.517 stored (−0.009). Per-sentence differences have SD 0.17 and max 0.88, and r = 0.79.
- **ParIR WAVs rescored (`scorer_control_parir.json`):**
  - Whisper gave **720/720 identical transcripts**; macro WER 0.0726 equals the stored value.
  - UTMOSv2: mean 3.526 against 3.530 stored (−0.004). Per-sentence differences have SD 0.15 and max 0.48, and r = 0.82.
- **What this means:**
  - The WER differences in section 3 are not scorer drift.
  - A ΔUTMOSv2 of about ±0.01–0.02 between sessions is within scorer variation, because UTMOSv2 random cropping is not seeded.
  - Same-session tests remove the session effect on the means, but not this per-WAV noise.

## 6. Reading the results

- **Against Piper_no_VITS2_cpn** (same backbone, ResBlock2 decoder):
  - **Speed:** SeqIR has 33% lower RTF in the same session and is faster on 690/720 sentences.
  - **WER:** SeqIR is slightly worse, 0.070 against 0.061 (+0.8 points). The CI just includes 0, and Holm p is 0.12. Most of the extra errors are deletions (76 vs 46).
  - **UTMOSv2:** SeqIR is slightly lower. Cross-session the difference is −0.024, with the CI upper bound just below 0 (Holm 0.12). In the same session it is −0.015, and the CI includes 0 (p 0.23).
  - **Validation mel loss:** SeqIR is clearly lower (−0.66 over the last 300 epochs). As with ParIR, that gain did not turn into better WER or predicted quality.
  - **Size:** the SeqIR decoder has 1.11 M parameters against 1.66 M (weight-norm form), 33% fewer.
- **Against Piper-ParIR:**
  - **Speed:** SeqIR has 13% lower RTF in the same session.
  - **WER:** not distinguishable (−0.003 [−0.013, +0.007]).
  - **UTMOSv2:** SeqIR is lower by 0.033–0.037, both cross-session and in the same session (Holm 0.021 and 0.025).
  - **Validation mel loss:** SeqIR is slightly higher (+0.09 to +0.12).
  - **Size:** 1.11 M against 1.16 M decoder parameters.
- **Against the four stored systems:**
  - SeqIR is clearly better than baseline-resblock2 (with VITS2) and the published Piper checkpoint on both measures.
  - Against the released Parallel-IR and Sequential-IR (VITS2 baseline recipe), SeqIR has lower WER (−3.8 and −5.3 points) and lower UTMOSv2 (−0.075 and −0.051).
- **Ranking by WER (seven systems):** Piper_no_VITS2_cpn 0.061, Piper-SeqIR 0.070, Piper-ParIR 0.073, then MRF 0.107, SEQ 0.122, baseline-resblock2 0.148 and Piper 0.175.
- **Among the three Config A models:**
  - **Speed:** SeqIR is fastest, then ParIR, then the control.
  - **WER:** Piper_no_VITS2_cpn is best; SeqIR and ParIR are close to each other.
  - **UTMOSv2:** the three are within 0.04, with ParIR highest and SeqIR lowest.

What this does **not** show:
- **Which part of the bundle matters.** The decoder blocks, SnakeBeta and the added MRD cannot be separated in this run. ParIR versus SeqIR also changes the block count and the kernels (protocol, section 4).
- **That the result holds beyond this setup.** There is one training run per model with seed 1234, one synthesis draw per sentence, Whisper-small WER and predicted (not human) MOS.
- **Significance across training seeds.** Paired tests describe sentence-level variation only.
- **Equivalence.** A non-significant difference does not establish equivalence; no margin was pre-specified.
- **Edge-device speed.** The timing is one fixed-order PyTorch CPU session on a workstation.

## 7. Not done in this round

- **Canonical test-500, ONNX export, PTQ and edge timing:** not done (same reasons as in `results/piper-no-vits2-cpn/README.md`, section 6).
- **A second SeqIR synthesis** to confirm determinism.
- **Repeated or counterbalanced timing,** and timing with a warm-up.

## 8. Files

In this folder:

| File | Content | Class |
|---|---|---|
| `selected_checkpoint.json` | Controller's selection: file, bytes, SHA-256, epoch, global step, `val_loss_mel`, rule | run record |
| `training_summary.json` | Run record, checkpoint hashes, per-epoch `val_loss_mel` for SeqIR and Piper_no_VITS2_cpn, window statistics | new summary of run logs |
| `harvard720_synth_record.json` | Synthesis record from the idle timing session: checkpoint/config/manifest hashes, protocol, environment, per-sentence timing and WAV SHA-256 | new measurement |
| `harvard_piper_seqir_no_vits2_cpn_results.json` | Per-sentence results, historical schema plus `wav_sha256`. The `rtf` field is from the idle session | new measurement |
| `score_record.json` | Scorer packages, weight hashes, environment, results hash | new measurement record |
| `comparison.json` | Summaries of all seven systems and the 12 paired tests (UTMOSv2, WER) | new analysis |
| `comparison_same_session_utmosv2.json` | Supplementary UTMOSv2 tests against the same-session rescorings | new analysis |
| `harvard_piper_no_vits2_cpn_rescored_results.json`, `piper_no_vits2_cpn_rescore_record.json` | Rescoring of the 2026-10-10 Piper_no_VITS2_cpn WAVs in this session. The `rtf` field is from 2026-10-10 | new control |
| `harvard_piper_parir_no_vits2_cpn_rescored_results.json`, `parir_rescore_record.json` | Rescoring of the 2026-10-16 ParIR WAVs in this session. The `rtf` field is from the loaded 2026-10-16 session; do not use it | new control |
| `scorer_control.json`, `scorer_control_parir.json` | Rescored versus stored scores for Piper_no_VITS2_cpn and ParIR | new control |

In [`../piper-ir-timing-session/`](../piper-ir-timing-session/):

| File | Content | Class |
|---|---|---|
| `piper_no_vits2_cpn_ctrl_synth_record.json` | Control synthesis of Piper_no_VITS2_cpn in the session, per-sentence timing and WAV SHA-256 | new measurement |
| `piper_parir_timing_synth_record.json` | ParIR synthesis in the session, per-sentence timing and WAV SHA-256 | new measurement |
| `session_control_parir.json`, `session_control_seqir.json` | Control reproducibility, session drift and paired RTF comparison against the control | new control |
| `parir_determinism.json` | ParIR WAV hashes and durations against the 2026-10-16 evaluation | new control |
| `machine_state_before.txt`, `machine_state_after.txt` | Load average, GPU use and top processes before and after the timing | session record |

The SeqIR record above (`harvard720_synth_record.json`) is the third synthesis of the same session.

The WAVs are outside Git:
- SeqIR: `training_output/piper-seqir-no-vits2-cpn/eval/harvard720/wavs`;
- timing session: `training_output/piper-ir-post-eval/timing/{ctrl,parir}/wavs`.

Their hashes are in the records.

## 9. Reproduction

Run from the training clone root:
- synthesis with the training venv;
- scoring with the scorer environment (versions in `score_record.json`);
- analysis in any environment with SciPy.

`<post>` is `training_output/piper-ir-post-eval`, and `<dir>` is any output directory.

```sh
# idle timing session, back to back: control, ParIR, SeqIR
python scripts/evaluation/harvard/synthesize.py --config configs/training/piper-no-vits2-cpn.yaml \
  --checkpoint "training_output/piper-no-vits2-cpn/checkpoints/best-epoch=1403-val_loss_mel=19.7719.ckpt" \
  --manifest results/manifests/harvard720.json --wav-dir <post>/timing/ctrl/wavs --output <post>/timing/ctrl/synth.json
python scripts/evaluation/harvard/synthesize.py --config configs/training/piper-parir-no-vits2-cpn.yaml \
  --checkpoint "training_output/piper-parir-no-vits2-cpn/checkpoints/best-epoch=1486-val_loss_mel=19.0069.ckpt" \
  --manifest results/manifests/harvard720.json --wav-dir <post>/timing/parir/wavs --output <post>/timing/parir/synth.json
python scripts/evaluation/harvard/synthesize.py --config configs/training/piper-seqir-no-vits2-cpn.yaml \
  --checkpoint "training_output/piper-seqir-no-vits2-cpn/checkpoints/best-epoch=1455-val_loss_mel=19.0332.ckpt" \
  --manifest results/manifests/harvard720.json \
  --wav-dir training_output/piper-seqir-no-vits2-cpn/eval/harvard720/wavs \
  --output training_output/piper-seqir-no-vits2-cpn/eval/harvard720/synth.json
# session control (the control rows are the 'rows' list of <post>/timing/ctrl/synth.json); repeat with the ParIR synth.json
python scripts/evaluation/harvard/session_control.py \
  --candidate-synth training_output/piper-seqir-no-vits2-cpn/eval/harvard720/synth.json \
  --control-synth <post>/timing/ctrl/synth_rows.json --control-wav-dir <post>/timing/ctrl/wavs \
  --historical-wav-dir training_output/piper-no-vits2-cpn/eval/harvard720/wavs \
  --historical-results results/piper-no-vits2-cpn/harvard_piper_no_vits2_cpn_results.json \
  --control-script scripts/evaluation/harvard/synthesize.py --output <dir>/session_control_seqir.json
# scoring: SeqIR, then the stored ParIR and Piper_no_VITS2_cpn WAVs
python scripts/evaluation/harvard/score.py --synth training_output/piper-seqir-no-vits2-cpn/eval/harvard720/synth.json \
  --output <dir>/harvard_piper_seqir_no_vits2_cpn_results.json --record <dir>/score_record.json
python scripts/evaluation/harvard/score.py --synth training_output/piper-parir-no-vits2-cpn/eval/harvard720/synth.json \
  --output <dir>/harvard_piper_parir_no_vits2_cpn_rescored_results.json --record <dir>/parir_rescore_record.json
python scripts/evaluation/harvard/score.py --synth training_output/piper-no-vits2-cpn/eval/harvard720/synth.json \
  --output <dir>/harvard_piper_no_vits2_cpn_rescored_results.json --record <dir>/piper_no_vits2_cpn_rescore_record.json
# analysis
python scripts/evaluation/harvard/compare.py --candidate <dir>/harvard_piper_seqir_no_vits2_cpn_results.json \
  --label piper_seqir_no_vits2_cpn --reference-dir results/historical/harvard \
  --extra-reference piper_no_vits2_cpn=results/piper-no-vits2-cpn/harvard_piper_no_vits2_cpn_results.json \
  --extra-reference piper_parir_no_vits2_cpn=results/piper-parir-no-vits2-cpn/harvard_piper_parir_no_vits2_cpn_results.json \
  --metrics utmosv2,wer --output <dir>/comparison.json
python scripts/evaluation/harvard/compare.py --candidate <dir>/harvard_piper_seqir_no_vits2_cpn_results.json \
  --label piper_seqir_no_vits2_cpn --reference-dir results/historical/harvard \
  --extra-reference piper_no_vits2_cpn_rescored=<dir>/harvard_piper_no_vits2_cpn_rescored_results.json \
  --extra-reference piper_parir_no_vits2_cpn_rescored=<dir>/harvard_piper_parir_no_vits2_cpn_rescored_results.json \
  --metrics utmosv2 --output <dir>/comparison_same_session_utmosv2.json
python scripts/evaluation/harvard/scorer_control.py --rescored <dir>/harvard_piper_no_vits2_cpn_rescored_results.json \
  --historical results/piper-no-vits2-cpn/harvard_piper_no_vits2_cpn_results.json --output <dir>/scorer_control.json
python scripts/evaluation/harvard/scorer_control.py --rescored <dir>/harvard_piper_parir_no_vits2_cpn_rescored_results.json \
  --historical results/piper-parir-no-vits2-cpn/harvard_piper_parir_no_vits2_cpn_results.json --output <dir>/scorer_control_parir.json
python scripts/training/summarize_run.py --run-dir training_output/piper-seqir-no-vits2-cpn \
  --reference-logdir training_output/piper-no-vits2-cpn/lightning_logs --reference-label piper-no-vits2-cpn \
  --reference-best-checkpoint "best-epoch=1403-val_loss_mel=19.7719.ckpt" --output <dir>/training_summary.json
```

**Expected outputs:**
- Synthesis is deterministic, so the WAV hashes should match.
- Whisper transcripts reproduced exactly in both controls.
- UTMOSv2 per-sentence scores vary between scoring sessions.
- Timing varies between sessions; only within-session ratios are meaningful.

The pipeline that ran these steps is kept with the runs, outside Git (`training_output/piper-ir-post-eval/post_seqir.sh`, `post.log`).
