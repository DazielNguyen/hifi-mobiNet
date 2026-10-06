# Protocol: hifi-mobiNet baseline without VITS2 (Piper_no_VITS2_cpn) versus with VITS2

Status: settings fixed by the author on 2026-10-06; written before any long run.
Nothing has been trained beyond functional smoke tests. Where the comparison
goes in the paper is not decided.

## 1. Question

Do the VITS2 components make the model better or worse? `Piper_no_VITS2_cpn`
is the hifi-mobiNet baseline **without** the VITS2 components (EdgeTTS Config A:
vanilla Piper). The released `baseline-resblock2` is Piper **with** the VITS2
components and is not retrained; only the model without VITS2 is trained. This is a **system-level** comparison: the VITS2 package is removed as
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
| Data location | Staged copy inside hifi-mobiNet (`data/ljspeech-medium`, Git-ignored), byte-verified against the source | Author: all training code and data in hifi-mobiNet |
| Budget | 1,500 epochs, 2 GPUs (DDP), batch 16 per GPU = 392 batches/epoch/rank, 588,000 generator updates | Same as the three internal models (all `max_epochs` 1500, 2 × 16) |
| Precision | bf16 | Author; Config A |
| Gradient clipping | None | SEQ/MRF runs (author: follow SEQ/MRF setup). The released baseline used 1.0 from its second phase; EdgeTTS's own Config A run used 1.0 |
| Seed | 1234 only | Author |
| Validation / selection | `val_loss_mel` on the 100 validation utterances each epoch; evaluate the checkpoint with the lowest `val_loss_mel`; keep `last.ckpt` | Author ("best + last"); same rule as the internal models |
| Sampler | Length-bucket, seed 1234, as in the BanhmiTTS models | Author |
| Data workers | 1 | SEQ/MRF runs |
| Max phoneme IDs | None (Config A's 400 would remove nothing; longest row 399) | SEQ/MRF runs |

## 3. Baseline side (decided)

The comparison uses the released `baseline-resblock2` (epoch 1489) as the model
with VITS2. It was trained by earlier BanhmiTTS code (no non-finite skip, no
health gate, clip null in its first phase and 1.0 afterwards, two resumes after
SDP collapse, `num_workers` 8). The two sides therefore also differ in codebase
and training history; results must be reported as a system-level comparison.
EdgeTTS Config C and a baseline retrain are not part of this plan.

## 4. Analysis rules

- Pre-register metrics (WER, UTMOSv2, RTF) and the paired tests before evaluation.
- Evaluate on test-500 / Harvard-720 with the existing scripts after selection;
  never use them for selection or tuning.
- Report effect sizes with confidence intervals; claim equivalence only with an
  equivalence test and a stated margin.
- Do not attribute differences to any single component.
- Smoke-test outputs and their WAVs are not results.
