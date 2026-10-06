# Historical research results

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

Public redistribution of sentence collections, result records and generated
audio remains an author decision. Local inclusion does not assert those rights.
