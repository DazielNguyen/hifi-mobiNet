# Journal evidence package of 2026-10-04 (imported 2026-10-22)

*Tiếng Việt: [README.vi.md](README.vi.md)*

This folder holds the BanhmiTTS `journal-evidence-package` of 2026-10-04: the Gen 1 evidence
for baseline-resblock2, Parallel-IR (MRF), Sequential-IR (SEQ) and the external Piper
checkpoint. It contains:
- claim ledger J-E001–J-E033 (`claim-ledger.json`, `evidence/claim_table.md`);
- context document (`JOURNAL_EVIDENCE_CONTEXT.md`, in Vietnamese);
- Harvard, data-split, architecture and ONNX/PTQ records (`evidence/B_` … `F_`);
- author notes (`evidence/author_notes/`), including the earlier Q2 checklist and the MRF/SEQ note.

Overview and current assessment: [`../../q2-readiness-seq-par.md`](../../q2-readiness-seq-par.md).

**How it was imported** (`../historical-import-2026-10-22.json`, one record per file):
- **Already in the repository (32 files):** files whose bytes were already tracked are not duplicated. This covers the Harvard and LJ-500 result files, the ONNX historical results, the Harvard manifest and the canonical split. The record names the existing path, for example `results/historical/harvard/`, `results/manifests/` or `docs/evidence/claim-evidence-2026-10-07/`. Relative links inside the package that point to those files therefore do not resolve here.
- **Byte copies:** the files without host paths.
- **Redacted copies:** files that named host paths were copied with placeholders and no other change: `<WORKSPACE>/`, `<WINDOWS_USER_HOME>/`, `<WSL_PROJECT_HOME>/`, `<WSL_USER_HOME>/`, `<CLAUDE_PROJECT_SLUG>`, `<HOST>`, `<EMAIL>`.
  - The import record keeps the original and redacted SHA-256 of each file.
  - The originals are kept privately in WSL (`<WSL_PROJECT_HOME>/hifimobinet-private/`), outside OneDrive and Git, and are not for release.
  - Redacted scripts are records, not runnable as they are.
- **Logs (16 files):** not tracked, because the repository ignores `*.log` (as it has since its first commit). Each is listed by hash in the import record and stays at its source.
- **Checks:**
  - a credential-pattern scan of every copied file;
  - an independent host-path scan after the import, which found 0 matches;
  - every destination hash re-checked.

**Reading it:**
- **Not current verdicts.** The package's assessments (`SUPPORTED`, `CONTRADICTED`, …) are those of 2026-10-04.
- **Author notes are context, not measurements.** In particular, `evidence/author_notes/MRF_vs_SEQ_comparison_Q2.md` contains claims that the ledger marks as contradicted (J-E030, J-E031).
- **Newer evidence.** The Gen 2 runs of October 2026 supersede several open items, for example the Q05/Q06 ONNX and PTQ work for the Config A models. See the overview.
