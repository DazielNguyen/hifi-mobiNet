Q01 evidence archive
{
  "archive_root_wsl": "<WSL_PROJECT_HOME>/paper_evidence/q01_2026-10-03T220747Z",
  "finished_utc": "2026-10-03T22:08:33.114145+00:00",
  "status": "verified_copy_complete",
  "copied_files": 9360,
  "copied_bytes": 6846969300,
  "checkpoint_files": 4,
  "onnx_files": 35,
  "wav_files": 8931,
  "result_tables": 27,
  "source_destination_sha256_matches": 9360,
  "failures": [],
  "not_run": [
    "training",
    "synthesis/inference",
    "export",
    "quantization",
    "audio rescoring",
    "benchmark"
  ],
  "scope": "Selected final checkpoints, Harvard and main Q2 evaluation evidence; supplementary historical result files marked separately by catalog coverage."
}

raw/ mirrors original absolute source paths. Originals were not moved or edited.
FILE_MANIFEST.json maps source paths to archived files and verified SHA-256 values.
RESULT_CATALOG.json/RESULT_SUMMARY.csv map result rows to evidence and explicit provenance gaps.
ENVIRONMENT_AT_ARCHIVE.json describes the CURRENT environment, not the historical run.
CHECKPOINT_METADATA.json uses weights_only metadata inspection without model execution.
PROVENANCE_GAPS.json contains unresolved scientific provenance; Q03-Q09 stay open.
The scripts retain original absolute paths; to rerun later, configure paths explicitly. No rerun occurred here.
SHA256SUMS covers every archived raw and generated file except SHA256SUMS itself.
This archive is on the same WSL disk. It is persistent beyond /tmp cleanup, not an off-device backup.
