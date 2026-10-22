# Frontend check: Piper external and the en-us phonemes of Harvard-720

*Tiếng Việt: [README.vi.md](README.vi.md)*

**Exploratory analysis of stored results** (2026-10-23). Nothing was synthesized or scored. The causal test is prepared in `docs/plan-frontend-causal-test.md` and has not been run.

## Why

On Harvard-720, Piper_no_VITS2_cpn has a macro WER of 6.12% against 17.47% for the external Piper checkpoint. Most of that gap is substitutions (529 of the 632 fewer word errors).

**The setup that may explain it:**
- Harvard-720 feeds **en-us** espeak phonemes to every system (`results/manifests/harvard720.json`).
- The released config of the Piper checkpoint names espeak voice **"en"** (`docs/evidence/claim-evidence-2026-10-07/source-records/public/public_piper_lj_med_config.json`). espeak-ng's "en" is British English.
- The model card describes the voice as US English, and every hifi-mobiNet model was trained on en-us (the training `config.json`).
- The phoneme-ID maps of Piper and the project agree on all 157 shared symbols. The input was therefore valid for Piper, but in another dialect's transcription than the one its config names.

## Method

`scripts/evaluation/harvard/dialect_check.py` produced `dialect_check.json`:
- **Re-phonemization:** every Harvard sentence is phonemized again with the frontend that built the manifest (`banhmi_phonemize`). The en-us output equals the manifest for 720/720 sentences; the en output is the comparison.
- **Sentence level:** the en/en-us phoneme edit distance, without stress marks and normalized by length, is compared with each system's WER.
- **Word level:**
  - Each reference word is marked as an error if the JiWER alignment (the scorer's transform) substitutes or deletes it.
  - A word "differs" if its en and en-us phonemes differ.
  - Only the 551 sentences whose phoneme words align one-to-one with the reference words are used (4,246 words).
- **Control:** Piper_no_VITS2_cpn is the control for word difficulty. The difference-in-differences has a bootstrap interval over sentences (5,000 resamples).

## Results

| Distance tercile (n) | Piper WER | Piper_no_VITS2_cpn WER | Gap |
|---|---:|---:|---:|
| lowest (238) | 10.42% | 4.72% | +5.70 |
| middle (220) | 17.24% | 6.74% | +10.51 |
| highest (262) | **24.06%** | 6.87% | **+17.19** |

Spearman correlation of the distance with WER: Piper +0.360 (p 2e-23); Piper_no_VITS2_cpn +0.074 (p 0.05); the Piper − Piper_no_VITS2_cpn gap +0.303.

| Word error rate | words whose en and en-us phonemes differ (1,458) | other words (2,788) | ratio |
|---|---:|---:|---:|
| Piper external | **34.2%** | 7.5% | 4.56 |
| Piper_no_VITS2_cpn | 5.8% | 5.5% | 1.05 |

- **Difference-in-differences:** **+26.4 points** [+23.9, +29.1].
- **Where Piper's extra errors fall:** 88% of its extra word errors, against Piper_no_VITS2_cpn in these sentences, are on words whose phonemes differ between the two voices.
- **On the other words:** the two systems are 2 points apart (7.5% against 5.5%).
- **Examples from Harvard list 1** (word: en-us → en; what Whisper heard from Piper):
  - background: `bˈækɡɹaʊnd` → `bˈakɡɹaʊnd`; "book ground";
  - hogs: `hˈɑːɡz` → `hˈɒɡz`; "herbs";
  - chopped: `tʃˈɑːpt` → `tʃˈɒpt`; "sharp";
  - four: `fˈoːɹ` → `fˈɔː`; "fair".

## Reading

- **What it supports:** the data are consistent with a frontend mismatch as the main source of Piper's extra ASR errors on Harvard-720. Piper errs on the dialect-specific words, and a model trained on en-us does not.
- **What it does not show:**
  - **Causation:** the analysis is correlational. The causal test synthesizes Piper with its own "en" phonemes, plus the reverse for the team model, with predictions declared in advance (`docs/plan-frontend-causal-test.md`).
  - **The training voice:** it is inferred from the released config, not from Piper's preprocessed training data, which is not available.
  - **Measurement limits:** Whisper-small transcripts, one synthesis draw per sentence, and the word level covers only the 551 alignable sentences.
- **Consequence until the causal test is done:** every Harvard-720 comparison with the external Piper checkpoint in this repository fed it mismatched phonemes. That covers the Gen 2 models and the historical baseline, MRF and SEQ. Read those comparisons as unfair to Piper.

## Reproduction

In an environment with `banhmi_phonemize`, JiWER and SciPy (the scorer environment):

```sh
python scripts/evaluation/harvard/dialect_check.py \
  --candidate piper=results/historical/harvard/harvard_piper_results.json \
  --control piper_no_vits2_cpn=results/piper-no-vits2-cpn/harvard_piper_no_vits2_cpn_results.json \
  --output <dir>/dialect_check.json
```

The en manifest for the causal test, `results/manifests/harvard720_espeak_en.json`, was built with `scripts/evaluation/harvard/build_manifest.py --voice en`. It differs in 691 of 720 sentences, and every symbol in it is in Piper's map.
