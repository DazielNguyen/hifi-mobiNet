# Plan: causal test of the frontend mismatch (Piper external on Harvard-720)

*Tiếng Việt: [plan-frontend-causal-test.vi.md](plan-frontend-causal-test.vi.md)*

Status: **prepared, not run.** Written 2026-10-23, before any result of this test exists.

## 1. Question and evidence so far

- **The setup.** Harvard-720 feeds **en-us** espeak phonemes to every system. The released config of the external Piper checkpoint (`lj-med_1000.ckpt`) names espeak voice **"en"** (British), while every hifi-mobiNet model was trained on en-us.
- **The exploratory check** (`results/frontend-check/`) supports a mismatch, but only by correlation:
  - Piper's WER rises with the per-sentence en/en-us phoneme distance (ρ = +0.36), while Piper_no_VITS2_cpn's barely does (ρ = +0.07).
  - Piper misses 34.2% of the words whose phonemes differ between the two voices, against 7.5% of the other words. Piper_no_VITS2_cpn misses 5.8% and 5.5%.
  - The difference-in-differences is +26.4 points [23.9, 29.1].
  - About 88% of Piper's excess errors are on differing words.

**This test:** does feeding Piper its own voice ("en") remove most of its WER gap? And does feeding en to the team's model, which was trained on en-us, hurt it?

## 2. Design

All on CPU, Harvard-720, the unchanged protocol of `docs/evaluation.md` (seed 1234 per sentence, scales 0.667/1.0/0.8, `harvard/score.py`):

| Set | System | Phonemes | Role |
|---|---|---|---|
| `piper_enus` | Piper external | en-us (`results/manifests/harvard720.json`) | same-session control; must reproduce the historical Piper WAVs |
| `piper_en` | Piper external | en (`results/manifests/harvard720_espeak_en.json`) | the test |
| `team_en` | Piper_no_VITS2_cpn (epoch 1403) | en | reverse test (mismatch for a model trained on en-us) |
| (stored) | Piper_no_VITS2_cpn | en-us | `results/piper-no-vits2-cpn/` |

- **The en manifest:** built by `scripts/evaluation/harvard/build_manifest.py`, which first rebuilt the en-us manifest and found it identical.
  - It differs in 691 of 720 sentences.
  - Every en symbol exists in Piper's phoneme-ID map, and the ID maps of Piper and the project agree on all 157 shared symbols.
- **Piper synthesis:** `scripts/evaluation/harvard/synthesize_piper.py`, with Piper's unmodified VITS code (`vendor/piper`) and the procedure of the historical `synth_piper.py`.
- **Scoring:** all three new sets are scored in one session, so the Piper en / en-us UTMOSv2 comparison is same-session. WER reproduces exactly across sessions.

## 3. Predictions (declared before the run)

| | Prediction if the frontend mismatch causes most of the gap | Read as |
|---|---|---|
| **P1 (primary)** | Piper `piper_en` macro WER < `piper_enus`; paired bootstrap 95% CI of the difference entirely below 0 | the mismatch hurts Piper |
| **P2** | `dialect_check` of `piper_en` (control: Piper_no_VITS2_cpn): difference-in-differences ≤ one third of round 1's (≤ +0.088), or its CI includes 0 | the dialect-specific error excess disappears |
| **P3 (reverse)** | `team_en` macro WER > stored en-us WER; CI entirely above 0 | the same mismatch hurts a model trained on the other voice |
| **P4 (descriptive)** | The WER gap between Piper_no_VITS2_cpn (en-us) and Piper (en) is smaller than the current 11.35 points | how much of the gap remains |

- **P1 and P2 hold:** the frontend mismatch is shown to cause the dialect-specific errors, and the remaining gap (P4) is what training, checkpoint and code differences can explain.
- **P1 fails:** the hypothesis is rejected for Piper, and the gap has another cause.
- **P3** is supporting evidence, not required.
- **UTMOSv2** is reported for every set but is not part of the predictions.

## 4. Run

Before starting:
- commit the prepared scripts and the en manifest (the launcher checks this);
- make sure the machine is idle and a WSL keep-alive is running.

In WSL:

```sh
bash <WSL_PROJECT_HOME>/hifimobinet-private/launch_frontend_test.sh
```

It starts the tmux session `frontend-test`, which runs `scripts/evaluation/harvard/run_frontend_test.sh` with:
- `OUT=training_output/frontend-test`;
- `RESULTS=<repository>/results/frontend-test`.

| Stage | Expected time |
|---|---|
| Piper en-us and en synthesis, team en synthesis (3 × 720 sentences) | about 6 min |
| Scoring, 3 WAV sets (about 45 min each) | about 2 h 15 min |
| Comparisons and dialect checks | about 2 min |
| **Total** | **about 2 h 25 min** |

**Following the run:**
- `tail -f <training clone>/training_output/frontend-test/run.log` shows progress.
- A pop-up reports DONE or FAILED.
- After an interruption, re-run `run_frontend_test.sh` with the same `OUT`: it resumes after the last finished stage.

## 5. After the run

1. Check `piper_enus_synth_record.json → reference_inventory.wavs_identical`.
   - **720/720:** the environment reproduces the historical Piper synthesis.
   - **Otherwise:** say so. The en versus en-us comparison stays valid because both sets were made in this run.
2. Evaluate P1–P4 from `piper_en_vs_piper_enus.json`, `dialect_check_piper_en.json`, `team_en_vs_team_enus.json` and `team_enus_vs_piper_en.json`. Report every prediction, including failed ones.
3. Write `results/frontend-test/README.md` and `README.vi.md`, and add `results/manifest.json` entries.
4. **If P1 holds, correct every comparison with Piper:**
   - in `results/README(.vi).md`, the Piper row and its notes;
   - in the model READMEs, every "vs Piper" result;
   - in `docs/q2-readiness-seq-par(.vi).md`, the SEQ/MRF/baseline-versus-Piper claims, including the Gen 1 ones.

   The historical Piper rows stay as records, labelled as measured with mismatched phonemes.
5. Commit in separate scopes, without attribution. The author pushes.

## 6. Prompt for the session that runs it

> Run the causal frontend test as planned in `hifi-mobiNet/docs/plan-frontend-causal-test.md`.
> - First check section 4 (committed scripts and en manifest, idle machine, WSL keep-alive). Then start `<WSL_PROJECT_HOME>/hifimobinet-private/launch_frontend_test.sh` and follow `run.log`.
> - Do not change the predictions of section 3, the protocol or the scorer.
> - When ALL DONE is logged, do section 5. Report every prediction, including failed ones.
> - Commit only when I ask, without attribution. Discuss in Vietnamese; documents in English with a Vietnamese README.vi.md.
