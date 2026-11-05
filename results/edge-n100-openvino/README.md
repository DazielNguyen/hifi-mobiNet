# Intel N100 / OpenVINO CPU results

| Session | Scope | Status |
|---|---|---|
| [2026-11-05 common97](2026-11-05-common97/README.md) · [Tiếng Việt](2026-11-05-common97/README.vi.md) | Three Piper models; FP32/common97 selective INT8; 4 threads; 3 × Harvard-720 | 12,960 timing records verified; RTF reduced 12.73–13.88%; preparation/IR hashes, 96 FQ edges/model and 54 WAVs checked; demo catalog matches recorded startup and measured INT8; independent quality pending |
| [2026-11-01](2026-11-01/README.md) · [Tiếng Việt](2026-11-01/README.vi.md) | Six internal models; FP32 and selective NNCF INT8; 4 threads; 3 × Harvard-720 | Timing complete; duration observations retained; INT8 quality pending |

Each session keeps its received records, logs, provenance, references and verification instructions, explicitly listing evidence not supplied. Compare measurements only within a compatible protocol; workstation PyTorch/ORT results remain separate. The November 1 INT8 graphs have the subsequently identified duration/alignment defect; their historical timings are not common97 results.
