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

Full acoustic rescoring requires the original local Whisper-small/UTMOSv2
artifacts, installation identities and waveform handling protocol. Historical
scorers were inspected and indexed; no automatic scorer-weight download is
included. Historical ONNX scores have additional protocol/provenance gaps and
cannot validate the Q05 graphs. See `results/README.md` and `release-gaps.md`.
