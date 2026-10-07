# Provenance and limitations

## Evidence classes

Each file in `source-manifest.json` and each evidence reference in `claim-evidence-index.json` carries one class. Classes are kept apart because they support different statements.

| Class | Meaning | Files here |
|---|---|---|
| Run-time record | Written by the run itself when it happened | Q05 `RUN_CONFIG.json` and `REPORT.*.json` (2026-10-04); historical ONNX result JSONs (repo `results/historical/onnx_historical/`); the 40 parity `.npz` files (transfer bundle); the Piper download manifest (2026-09-30) |
| Current check of artifacts/metadata | Produced on 2026-10-07 (or the 2026-10-04 audit) by inspecting files as they exist now | `outputs/J-C001_key_inventories.json`; `outputs/pc/*` except J-C011/J-C023; `source-records/onnx/onnx_inventory_extract.json`; graph sections of `outputs/verify_report.json` |
| Recomputed from stored outputs | A current script applied to outputs stored earlier | parity NRMSE (`J-C011`), RTF pairing (`J-C023`), text overlap (`J-C026`), calibration re-sampling (`J-C010`), Harvard statistics (repo `results/prior_audit/`) |
| Public source | Third-party public pages and metadata | `source-records/public/*` (model card, file listing with LFS oids, config, revision record) |
| Design record | Plans and candidates that were not executed | `calibration60.candidate.jsonl`, `PROTOCOL_Q04_v0.2.json` |
| Script record | Scripts that produced records, copied as records | `scripts/pc/`, `scripts/q05/`, `scripts/historical/` |
| AI interpretation | Synthesis and suggested wording | `source-records/pc-package/README.md`, `proposed_narrowed_claim` fields, the PC labels in `pc_reported_assessment` |
| AI search log | What was searched and not found | `source-records/search-records.md` |
| Not found | Claims with no record in the scanned scope | J-C012, J-C013, J-C015 |

## What specific sources can and cannot show

- **Checkpoint name inventories (J-C001).**
  - They can show structural compatibility of stored state.
  - They cannot show shapes, values, forward equivalence, the executed code or its origin.
  - The code side is the current checkout, not a training-time snapshot.
- **ONNX metadata `checkpoint_sha256` (J-C011).**
  - It records what the export run declared.
  - The Q05 runner also hashed the checkpoint snapshot it loaded strictly in the same run (REPORT `checkpoint.sha256`).
  - The two agree, but the metadata alone does not prove which bytes were read.
- **Named-weight comparisons of historical graphs.**
  - "422/422 equal" links a graph to a checkpoint at the level of named initializers.
  - Folded or renamed weights are not covered.
  - "0/478 equal" shows the graph is not from that checkpoint, without identifying its source.
- **INT8 → FP32 decoder match (J-C023).**
  - The decoder stays FP32 under the historical PTQ, so byte-identical decoder initializers tie an INT8 graph to its FP32 parent.
  - Quantized parts are not compared.
- **Op counts (J-C025).**
  - They count exported operators. Head counts, layer design and module boundaries are not read from them.
- **Historical timings (J-C023).**
  - They are single measurements per utterance without warm-up, on different days and on the development workstation.
  - They are not a benchmark of the Q05 graphs.
- **Harvard statistics (J-C019).**
  - They are exploratory: one synthesis draw, one scorer pass and one training run per model.
  - A CI that excludes zero is not an equivalence test.
- **Text overlap (J-C026).**
  - The comparison is against the project's LJSpeech transcripts only.
  - Piper's own training transcripts are not available; the model card is the source for Piper.
- **Search logs (J-C012, J-C013, J-C015).**
  - Absence holds only within the scanned scope listed in the log.

## Not imported as verification

The PC session labelled several claims "supported" or "contradicted". Those labels are kept verbatim as `pc_reported_assessment` and nothing more. In particular:
- J-C019 is **not** labelled CONTRADICTED here. The data show a trade-off, and equivalence was never tested.
- J-C023 numbers compare SEQ with the **baseline**, not with MRF.

## Redactions and private originals

- 14 of the 29 copied files contained absolute paths of the author's machines.
- They were copied with these replacements:
  - `<WORKSPACE>/`: project workspace root;
  - `<WINDOWS_USER_HOME>/`;
  - `<WSL_PROJECT_HOME>/`;
  - `<WSL_USER_HOME>/`.
- No other content was changed.
- For each file, `source-manifest.json` records `source_sha256` (original) and `destination_sha256` (sanitized).
- The originals travel only in `private-originals/` of the transfer ZIP, together with an unchanged copy of the whole PC package. They are not for Git or public release.
- Two copies are therefore not runnable as-is:
  - `scripts/pc/collect_claim_evidence.sanitized.py`;
  - `scripts/historical/collect_C_data_split.py`.
- No credentials or tokens appear in any copied file. This was checked on 2026-10-07 by a pattern scan for Hugging Face and GitHub token formats and for `api_key` / `token` / `password` / `secret` assignments. The only keyword hits were LJSpeech text ("Secret Service") and the phrase "blank tokens".

## Known gaps

- **J-C023:** there is no identified MRF benchmark graph, and the Q05 graphs have never been timed.
- **J-C010 and J-C024:** there is no run-time calibration manifest (it is reconstructed), and no PTQ of the Q05 graphs.
- **J-C012, J-C013, J-C015:** there are no records.
- **J-C001:** there is no training-time code snapshot.
