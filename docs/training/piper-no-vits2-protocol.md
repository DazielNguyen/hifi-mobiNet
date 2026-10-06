# Protocol: Piper_no_VITS2_cpn versus the internal baseline

Status: **draft, written before any long run.** Items marked *Decision* need
the author's choice; nothing here has been trained beyond the functional smoke
tests.

## 1. Question

Does training the upstream Piper VITS graph (no VITS2 components) in the same
data/split/harness give different quality or cost than the internal ResBlock2
baseline with the VITS2 package? This compares **systems**. The two models also
differ in non-VITS2 details (component table, section 3), so no per-component
effect can be attributed, and a non-significant difference is not evidence of
equivalence.

## 2. What can be held equal

| Factor | Baseline recipe | Piper_no_VITS2_cpn (matched config) | Equal? |
|---|---|---|---|
| Dataset / preprocessing | Project LJSpeech cache, `dataset.jsonl` SHA-256 `f4a72ae0…295b` | Same files | Yes |
| Phonemizer / IDs / vocabulary | `banhmi_phonemize` en-us, 256 symbols; 13,100/13,100 IDs reproduced | Same IDs | Yes (not Piper's frontend) |
| Split | `results/manifests/canonical_split.json` SHA-256 `e678cf43…ae9d`: 12,500 / 100 / 500 | Same | Yes |
| Sample rate, STFT, mel, segment | 22,050 Hz; 1024/256/1024; 80 mel; segment 8,192 | Same | Yes |
| Decoder | ResBlock2 3/5/7, upsampling 8/8/4 | Same except final LeakyReLU slope 0.01 vs 0.1 | No (upstream detail) |
| Seed / initialization | torch seed 1234; module-default init | Same seed; upstream init | Seed only (graphs differ) |
| Batch / accumulation | 16 per GPU, no accumulation | Same | Yes |
| Optimizer / LR schedule | AdamW 2e-4 (0.8, 0.99, 1e-9), ExponentialLR 0.999875/epoch | Same values; non-fused AdamW kernel | Values yes |
| Precision | bf16 | fp32 (upstream fails under bf16) | **No** |
| Gradient clipping | 1.0 | 1.0 | Yes |
| Numerical guards in model | SDP clamps/eps, infer clamp | None (upstream) | No |
| Harness policies | Non-finite skip, health gate, RNG save | Same | Yes |
| Sampler | Length-bucket, seed 1234 | Same | Yes |
| Validation and selection | `val_loss_mel` (teacher-forced, 100 val utterances), top-3 + last | Same | Yes |
| Training budget | *Decision* (section 4) | Same number of optimizer updates | By construction |

Budget arithmetic (length-bucket sampler on the 12,500 training utterances):
batch 16 on 1 GPU = 784 batches/epoch; 2 GPUs = 392 batches/epoch/rank
(1 generator + 1 discriminator update per batch). The historical baseline used
2 GPUs × 16 for 1,500 epochs = 588,000 generator updates.

## 3. Baseline identity

The released `baseline-resblock2` (epoch 1489) was trained by **earlier**
BanhmiTTS code: no non-finite skip, no health gate, clip null for its first
phase, two resumes after SDP collapses (journal evidence package). The
`baseline-resblock2-vits2` recipe here runs the **current** code. Therefore:

- Comparing a new Piper_no_VITS2_cpn run against the historical baseline keeps
  confounders (code version, resume history, precision, split handling of
  that run). Report it only as an uncontrolled reference.
- A controlled comparison needs **both** models trained now with this
  harness, same split, same budget, same selection rule. *Decision:* train the
  new model only, or both (recommended for a journal claim).

## 4. Proposed settings (pending decisions)

| Item | Proposal | Status |
|---|---|---|
| Initialization | From scratch for both; no pretrained Piper or baseline weights | Fixed by protocol |
| Budget | Equal optimizer updates, e.g. 588,000 generator updates (= 1,500 epochs at 2 × 16) | *Decision* |
| GPUs | Same device count and per-device batch for both runs | *Decision* (1 or 2 GPUs) |
| Seeds | 1234 for both; additional seeds only if budget allows | *Decision* |
| Checkpoint selection | Lowest `val_loss_mel` among top-3, ties → later epoch; evaluate exactly one checkpoint per model | *Decision* |
| Precision | Baseline bf16, Piper fp32; alternatively run both in fp32 to remove this difference (baseline fp32 not yet smoke-tested) | *Decision* |
| Evaluation | Held-out test-500 / Harvard-720 with the existing scripts, after selection; test data never used for selection | Fixed |

## 5. Analysis rules

- Pre-register metrics (WER, UTMOSv2, RTF) and paired tests before evaluation.
- Report effect sizes with confidence intervals; non-significance is not
  equivalence. Use an equivalence test with a stated margin if equivalence is
  claimed.
- Do not attribute differences to any single component.
- Smoke-test outputs, the 50-update runs and their WAVs are not results.
