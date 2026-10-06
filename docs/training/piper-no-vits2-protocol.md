# Protocol: Piper_no_VITS2_cpn (EdgeTTS Config A) versus the internal baseline

Status: settings fixed by the author on 2026-10-06; written before any long run.
Nothing has been trained beyond functional smoke tests. Where the comparison
goes in the paper is not decided.

## 1. Question

Does adding the VITS2 components to Piper move quality or cost in a positive
or negative direction? `Piper_no_VITS2_cpn` (EdgeTTS Config A: vanilla Piper)
is compared with the internal ResBlock2 baseline that carries the VITS2
package. This is a **system-level** comparison: the VITS2 package is removed as
a whole and the two models come from different codebases, so no single
component's effect can be attributed and a non-significant difference is not
evidence of equivalence (component table:
[piper-component-comparison.md](piper-component-comparison.md)).

## 2. Fixed settings

| Item | Setting | Source |
|---|---|---|
| Initialization | From scratch; no pretrained Piper/baseline weights | Protocol |
| Data | Project LJSpeech cache, `dataset.jsonl` SHA-256 `f4a72ae0…295b`; IDs from `banhmi_phonemize` (13,100/13,100 reproduced); no F0 | Shared with baseline |
| Split | `results/manifests/canonical_split.json` (SHA-256 `e678cf43…ae9d`): 12,500 / 100 / 500 | Same as baseline (author) |
| Budget | 1,500 epochs, 2 GPUs (DDP), batch 16 per GPU = 392 batches/epoch/rank, 588,000 generator updates | Same as the three internal models (all `max_epochs` 1500, 2 × 16) |
| Precision | bf16 | Author; Config A |
| Gradient clipping | Norm 1.0 per optimizer (EdgeTTS `grad_clip`) | Config A run hparams; baseline final phases also 1.0 |
| Seed | 1234 only | Author |
| Validation / selection | `val_loss_mel` on the 100 validation utterances each epoch; evaluate the checkpoint with the lowest `val_loss_mel`; keep `last.ckpt` | Author ("best + last"); same rule as the internal models |
| Sampler | Length-bucket, seed 1234 (baseline's); EdgeTTS's fixed-order loader available | Harness parity |
| Max phoneme IDs | 400 (Config A); removes nothing (longest row 399) | Config A |

## 3. Baseline side

The released `baseline-resblock2` (epoch 1489) was trained by earlier
BanhmiTTS code (no non-finite skip, no health gate, clip null in its first
phase, two resumes after SDP collapse). Comparing a new Config A run with that
checkpoint therefore also compares code versions and training histories.
Options, for the author:

1. Compare with the released baseline as an uncontrolled reference (no new
   baseline run).
2. Retrain `baseline-resblock2-vits2` with this harness under the same budget
   and selection rule (controlled pair across codebases).
3. For the cleanest VITS2 test, also train EdgeTTS Config C (`use_vits2` only)
   in the same harness and compare A versus C (one codebase, one switch).

## 4. Analysis rules

- Pre-register metrics (WER, UTMOSv2, RTF) and the paired tests before evaluation.
- Evaluate on test-500 / Harvard-720 with the existing scripts after selection;
  never use them for selection or tuning.
- Report effect sizes with confidence intervals; claim equivalence only with an
  equivalence test and a stated margin.
- Do not attribute differences to any single component.
- Smoke-test outputs and their WAVs are not results.
