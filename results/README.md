# Historical research results

*Tiếng Việt: [README.vi.md](README.vi.md)*

All files under `historical/` are byte-identical to the indicated files in the
existing journal evidence package. Their hashes, source locators and limitations
are in `manifest.json` and `docs/source-map.json`. No new research measurements
were produced to populate these directories.

| Model | Harvard mean RTF | UTMOSv2 predicted | Macro WER |
|---|---:|---:|---:|
| HiFi-GAN ResBlock2 | 0.04177 | 3.0260 | 0.1476 |
| Parallel-IR | 0.03388 | 3.5680 | 0.1073 |
| Sequential-IR | 0.03081 | 3.5443 | 0.1222 |
| Piper external checkpoint | 0.03916 | 3.4507 | 0.1747 |

These are **historical PyTorch CPU observations** on 720 common Harvard sentences,
from the development workstation under WSL. They are not edge-device or ONNX
results. UTMOSv2 is not human MOS. There was one training run per model, one
synthesis draw per sentence, and no repeated timing protocol. Historical runtime
versions were bracketed by nearby records rather than hashed inside every run.

The later Q05 FP32 exports are identified and had engineering checks, but these
quality scores are not bound to those exports. The demo is a functional interface,
not evidence that the new code reproduces the table above.

`lj500/` retains both the selected MRF epoch-1386 evaluation
(`eval_mrf_final1500_500_results.json`) and an earlier MRF table. Filenames and values
remain unchanged. MRF and canonical LJSpeech test sets share only 21 IDs; do not
interpret all the 500-row files as a common paired test. `manifests/` includes
the canonical split observed later and Q04 frozen ordered IDs; this is not proof
that every manifest was recorded contemporaneously with training.

`onnx_historical/` is explicitly exploratory. The legacy MRF graph was not bound
to the selected checkpoint; historical calibration overlaps test data and PTQ
scopes vary. These tables are retained for transparency, not promoted as final
ONNX/PTQ claims. `prior_audit/` contains an unchanged later statistical audit.

The new portable statistical script reproduces that audit into `outputs/`, leaving
raw inputs untouched. Paired tests and bootstrap intervals concern sentence-level
observations, not variation across training runs or hardware. Non-significance
does not establish equivalence, and recipe/split differences prevent a claim of
decoder-only causation.

## New measurements

`piper-no-vits2-cpn/` holds the first new measurements in this repository (2026-10-10/11):
the Harvard-720 evaluation of `Piper_no_VITS2_cpn`, the baseline without the VITS2
components. It contains its own README, per-sentence results, scorer records, paired
comparisons with the historical files above, and same-session controls. These files
add to the historical tables above and do not replace them. The historical RTF values
are not comparable with RTF measured in that session (a timing drift of +8.3% was
measured on the same baseline model).

`piper-parir-no-vits2-cpn/` (2026-10-16) holds the Harvard-720 quality evaluation of
`Piper_ParIR_no_VITS2_cpn`: the same backbone with the released Parallel-IR decoder
bundle (inverted-residual decoder, SnakeBeta, MRD). It is compared by sentence with
`Piper_no_VITS2_cpn` and the four historical systems. It includes a same-session scorer
control that rescored the `Piper_no_VITS2_cpn` WAVs. Its RTF was measured later, in the
2026-10-21 session below.

`piper-seqir-no-vits2-cpn/` (2026-10-21) holds the evaluation of `Piper_SeqIR_no_VITS2_cpn`:
the same backbone with the released Sequential-IR decoder bundle (two IR blocks per stage,
SnakeBeta, MRD).
- It is compared by sentence with the two other Config A models and the four historical systems.
- It includes same-session rescoring of the `Piper_no_VITS2_cpn` and ParIR WAVs.
- Its README also reports the idle-machine timing session. The session timed the three
  Config A models back to back, with `Piper_no_VITS2_cpn` as the control. Its records are
  in `piper-ir-timing-session/`.

| New model | UTMOSv2 (predicted) | Macro WER | Mean RTF, 2026-10-21 session | Relative RTF |
|---|---:|---:|---:|---:|
| Piper_no_VITS2_cpn (ResBlock2, control) | 3.517 | 0.0612 | 0.0510 | — |
| Piper_ParIR_no_VITS2_cpn | 3.530 | 0.0726 | 0.0392 | −23.2% |
| Piper_SeqIR_no_VITS2_cpn | 3.493 | 0.0696 | 0.0340 | −33.3% |

The RTF column comes from one session and is comparable only within it. The control ran 1.21×
slower than in its own 2026-10-10 session (0.0421), so neither that value nor the historical
RTF table above can be set beside these values.

## Model size

`model_parameters.json` (2026-10-21) records parameter counts and sizes for all seven
Harvard-720 models. It was written by `scripts/evaluation/count_parameters.py`, which reads
the three new checkpoints with `weights_only=True` without building or running a model.
The historical rows are taken from the stored Q05 reports and `models/manifest.json`.

| Model | Parameters (Q05 definition) | Used at inference | Decoder | FP32 weights | ONNX FP32 file | Training checkpoint |
|---|---:|---:|---:|---:|---:|---:|
| Piper_no_VITS2_cpn (ResBlock2) | 16,425,776 | 15,663,771 | 1,664,768 | 65.7 MB | not exported | 845.9 MB |
| Piper_ParIR_no_VITS2_cpn | 15,918,032 | 15,156,027 | 1,157,024 | 63.7 MB | not exported | 843.4 MB |
| Piper_SeqIR_no_VITS2_cpn | 15,870,768 | 15,108,763 | 1,109,760 | 63.5 MB | not exported | 842.7 MB |
| HiFi-GAN ResBlock2 (baseline, with VITS2) | 17,620,784 | — | — | 70.5 MB | 68.5 MB | 867.0 MB |
| Parallel-IR (MRF) | 17,113,040 | — | — | 68.5 MB | 66.5 MB | 864.5 MB |
| Sequential-IR (SEQ) | 17,065,776 | — | — | 68.3 MB | 66.3 MB | 863.9 MB |
| Piper external checkpoint | 16,425,776 | — | — | 65.7 MB | 63.5 MB | 845.9 MB |

How to read the columns (MB = 10^6 bytes):
- **Parameters (Q05 definition):**
  - generator parameters without the posterior encoder `enc_q`, which is used only in training;
  - counted in weight-norm form, with buffers excluded;
  - the same definition as the Q05 export reports, so all seven rows are comparable.
  - It still includes 762,005 parameters of the stochastic duration predictor that inference does not use: the training-only posterior path `dp.post_*` (640,248) and the flow dropped in reverse mode (121,757).
- **Used at inference:** the Q05 count minus those 762,005. It was computed for the new models only.
- **FP32 weights:** 4 bytes × the Q05 count. This is computed, not a file size.
- **ONNX FP32 file:** the measured size of the Q05 exports. The ONNX graphs hold only the inference path, so they are smaller than the FP32-weights column. The new models have not been exported.
- **Training checkpoint:** the full Lightning state: generator, posterior encoder, discriminators and optimizer states. It is not a deployable model size.

What the counts show:
- **Shared part:** in the three new models, everything outside the decoder is identical, 21,999,024 parameters including `enc_q`.
- **Decoder:** ParIR and SeqIR have 30.5% and 33.3% fewer decoder parameters than ResBlock2. Overall that is 3.1% and 3.4% fewer, because the decoder is a small part of the generator.
- **Piper_no_VITS2_cpn and the external Piper checkpoint:** both have exactly 16,425,776 parameters. Both use the Piper (Config A) architecture.
- **Training only:** the MPD (46,747,132) and, for the IR models, the MRD (280,902) are discriminators. They are not part of the synthesis model.

Public redistribution of sentence collections, result records and generated
audio remains an author decision. Local inclusion does not assert those rights.
