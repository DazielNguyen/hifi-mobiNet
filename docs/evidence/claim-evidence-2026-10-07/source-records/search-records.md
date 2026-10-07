# Search records for J-C012, J-C013 and J-C015

Evidence type: **AI search log** (assistant-run, read-only searches). These records show what was searched and what was
found. A search that finds nothing is not evidence that something never existed outside the scanned scope.

- First search: 2026-10-07 (PC session that produced `source-records/pc-package/README.md`).
- Re-run for this package: 2026-10-07T04:30Z (commands below), same conclusions.
- Scanned scope:
  - the author's Windows project workspace (`<WORKSPACE>`, every sub-repository), excluding `.git`, virtual
    environments, `node_modules` and third-party literature caches;
  - the WSL project account home (`<WSL_PROJECT_HOME>`), excluding virtual environments and caches;
  - earlier assistant session logs for the VNNI test package (searched in the first pass only).
- Not scanned: the author's Mac, external drives, cloud storage other than the OneDrive-synced workspace, devices
  that were never connected to this workstation.

## J-C012: final PTQ uses a corrected evaluator sample rate and is reproducible

Searches:
```
find <WORKSPACE> -name "eval_wer_cheap*.py"
grep -n -i -E "sample_rate|sr=|16000|22050|resample" <each hit>
find <WSL_PROJECT_HOME> -name "*int8*.onnx" -newermt 2026-10-04
```
Found:
- 4 copies of `eval_wer_cheap.py`, all byte-identical (SHA-256 prefix `29e88c096cfc567a`):
  - `BanhmiTTS/journal-evidence-package/evidence/E_onnx_ptq/scripts_and_logs/`
  - `BanhmiTTS/docs/evidence/q04_2026-10-04/archived_source/...`
  - `paper_revisions/thesis_ptq/evidence_*_2026-10-01/raw/...` (two copies)
- The script passes the ONNX output array (22,050 Hz) straight to `whisper_model.transcribe(audio, ...)` (line 71),
  with no resampling and no sample-rate argument. Whisper interprets a raw array as 16 kHz, so the sample-rate
  problem is **present** in every copy found.
- No INT8 ONNX graph modified after 2026-10-04 was found (no PTQ re-run).
- `PROTOCOL_Q04_v0.2.json` is a candidate design ("not yet executed"); see `calibration/`.

Conclusion: **not found in the scanned scope.** No corrected evaluator, no re-run PTQ selection, no re-evaluation
result.

## J-C013: runtime and memory measured on Intel N150 and Raspberry Pi 5

Searches:
```
grep -rIl -i -E "\bN150\b|raspberry|\brpi ?5\b|\bpi ?5\b" --include=*.log --include=*.csv --include=*.json --include=*.txt <WORKSPACE>
(same pattern) <WSL_PROJECT_HOME>
```
Found: files that mention the devices, all of them plans or manuscript material, none a measurement log.
- Protocols: `PROTOCOL_Q04_v0.1/v0.2.json`.
- `FP32_HANDOFF.json`: pending-work list.
- Claim ledger.
- Thesis revision text and edit lists under `paper_revisions/thesis_ptq/`.
- Research spreadsheet cells.

There are no `.log` or `.csv` files with device timings or memory readings.

The only CPU benchmark package (`vnni_test_package`, 2026-08-23):
- ran on the development workstation (Intel i7-14700), not on an N150 or a Pi 5;
- used an EdgeTTS-era Piper voice, not a hifi-mobiNet model.

The author stated (2026-10-07) that the N150 and Pi 5 measurements will be made once the devices are available.

Conclusion: **not found in the scanned scope.** The planned device matrix is not a result.

## J-C015: a human listening study confirms the perceived quality of the deployed model

Searches:
```
grep -rIl -i -E "listening (test|study)|MUSHRA|participants?\b.*(rated|consent)|consent form|subjective (test|evaluation)|human MOS" \
     --include=*.md --include=*.json --include=*.csv --include=*.txt --include=*.xlsx <WORKSPACE>
grep -rIl -i -E "MUSHRA|listening (test|study)|consent form|human MOS" ... <WSL_PROJECT_HOME>
```
Found: only these kinds of file:
- agent/skill instructions;
- reporting guides;
- paper drafts and literature text;
- protocols and plans;
- the claim ledger;
- the public listening-demo pages and manifests.

The demo pages are listening samples, not a study.

There was no protocol with participants, no ratings, no consent records and no analysis.

Conclusion: **not found in the scanned scope.** UTMOSv2 scores do not substitute for a human study.
