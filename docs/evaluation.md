# Evaluation and statistics

No synthesis dataset, ASR model or UTMOSv2 model is executed during repository
preparation. Stored observations are kept separate from new audits.

```sh
python scripts/evaluation/audit_results.py summarize tests/fixtures/metrics.json
python scripts/evaluation/audit_results.py paired results/historical/harvard/harvard_seq_results.json results/historical/harvard/harvard_mrf_results.json --metric wer
python scripts/evaluation/recompute_harvard.py --input-dir results/historical/harvard --output-dir outputs/harvard-statistics
```

The last command ports only the input/output paths of the archived
`recompute_B_stats.py`. Its 10,000 paired bootstrap resamples, seed 20261004,
SciPy two-sided Wilcoxon defaults, Mann–Whitney defaults and exploratory Holm
adjustment over 15 comparisons are unchanged. The original script is retained
under `scripts/evaluation/original/`. These are **prior audit settings**, not a
retrospectively pre-specified study protocol. The older run's `stats.py` used
5,000 resamples and seed 0 for mean CIs; these two analyses must not be conflated.

WER uses JiWER 4.0.0: lowercase, remove punctuation, collapse repeated whitespace,
strip, split into words. Numbers are not additionally verbalized. Reported WER
is the arithmetic mean of sentence WER fractions; corpus WER is separate and
exploratory. RTF means per-sentence ratios; ratio of total times to total durations
is a different aggregation. UTMOSv2 is a predicted score, not listening-test MOS.

The CLI rejects failed/non-finite observations rather than inventing scores or
silently changing denominators. Paired comparisons require identical IDs, text
and ordering. LJSpeech-500 MRF has a different test manifest; it must not be fed
into a paired comparison against the other models. Harvard-720 comparisons are
paired by sentence, but this does not establish independence from model selection.

## Harvard-720 evaluation of newly trained checkpoints

`scripts/evaluation/harvard/` runs the historical Harvard protocol on a checkpoint
trained by this repository's harness. It runs strictly one step at a time.

```sh
# training venv, from the checkout root (relative paths keep records host-independent)
python scripts/evaluation/harvard/synthesize.py --config <config.yaml> --checkpoint <ckpt> \
  --manifest results/manifests/harvard720.json --wav-dir <run>/eval/harvard720/wavs --output <run>/eval/harvard720/synth.json
# scorer environment (UTMOSv2, Whisper, JiWER), same directory
python scripts/evaluation/harvard/score.py --synth <run>/eval/harvard720/synth.json \
  --output results/<model>/harvard_<label>_results.json --record results/<model>/score_record.json
python scripts/evaluation/harvard/compare.py --candidate results/<model>/harvard_<label>_results.json \
  --label <label> --reference-dir results/historical/harvard --output results/<model>/comparison.json
```

- **`synthesize.py`** reproduces the historical synthesis settings:
  - PyTorch FP32 on CPU, scales 0.667/1.0/0.8;
  - `torch.manual_seed(1234)` before every utterance;
  - decoder weight norm removed; only the forward call is timed.
  - The checkpoint is loaded strictly (`weights_only=True`). Its stored hyper-parameters must equal the config.
- **`score.py`** is the historical `score.py` with arguments added:
  - UTMOSv2 `fusion_stage3` fold 0, one prediction per WAV, without explicit seeding;
  - Whisper `small` on the WAV path, `language="en"`, `fp16=False`;
  - the WER transform above.
  - Additions: Hugging Face is forced offline, WAV hashes are checked, and scorer weight hashes are recorded.
- **`compare.py`** pairs the new results with `results/historical/harvard/`, using `hifimobinet.evaluation`:
  - Wilcoxon and Mann–Whitney tests;
  - a paired bootstrap of the mean difference (10,000 resamples, a fresh `default_rng(20261004)` per comparison);
  - an exploratory Holm adjustment.
  - It is not an equivalence test.
- **`scripts/training/summarize_run.py`** reads the per-epoch `val_loss_mel` of a finished run, and optionally of a reference run, from TensorBoard scalars. It also lists checkpoint hashes.

UTMOSv2 crops randomly and the historical protocol does not seed it. The same WAV
can therefore score slightly differently in another scoring session.

Full acoustic rescoring requires the original local Whisper-small/UTMOSv2
artifacts, installation identities and waveform handling protocol. Historical
scorers were inspected and indexed; no automatic scorer-weight download is
included. Historical ONNX scores have additional protocol/provenance gaps and
cannot validate the Q05 graphs. See `results/README.md` and `release-gaps.md`.
