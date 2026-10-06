# Local demo

```sh
python -m streamlit run demo/streamlit_app.py --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
```

Comparison does not need weights, PyTorch or a phonemizer. Each entry records
utterance ID/text, model, source, hash, frame count and WAV format. Historical bytes
are passed to the player unchanged; missing or altered files have explicit errors.

The catalog selects Harvard IDs 0000–0002 before listening or score selection.
Actual WAVs remain external local assets until access/redistribution is settled.
Use the installer in `setup.md` or set `HIFIMOBINET_AUDIO_DIR` to an operator-managed
directory containing `baseline/`, `mrf/`, `seq/` and `piper/`. Web users cannot set paths.

TTS requires a local functional pass, identified FP32 weights and the original
frontend. Limits are 300 characters and 800 IDs. Only one synthesis request can
run per process; overlapping requests fail clearly. Streamlit caches one model,
but eviction and concurrent sessions do not establish a hard process-memory limit.

New audio is labelled separately and uses the original peak-to-PCM16 conversion.
There is no timing/quality display or claim that demo output reproduces old samples.

Programmatic tests cover rendering, sentence selection, WAV widgets, hash rejection
and missing-weight states. Synthetic silent WAV fixtures are test-only, never
research samples. Human/browser listening quality is not assessed by those tests.
