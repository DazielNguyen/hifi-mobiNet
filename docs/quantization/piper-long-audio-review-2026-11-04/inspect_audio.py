"""Read-only input analysis. No TTS/ASR inference, model loading or quantization.

Writes derived JSON/CSV/PNG beside this script. The source WAV stays in Downloads.
Run from the repo with an existing Python containing numpy, scipy and Pillow:
  .venv/Scripts/python.exe -B docs/quantization/piper-long-audio-review-2026-11-04/inspect_audio.py --wav PTQ_PATH --reference FP32_PATH --text TEXT
"""
import argparse
import csv
import hashlib
import json
import platform
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy import signal
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "evidence"
EVIDENCE = ROOT / "results/edge-n100-openvino/2026-11-01/evidence"
RUNS = EVIDENCE / "results-n100-t4-20261101T080423Z"
MODELS = ["piper-parir-no-vits2-cpn", "piper-seqir-no-vits2-cpn"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path):
    try:
        name = path.relative_to(ROOT).as_posix()
    except ValueError:
        name = str(path)
    return {"path": name, "bytes": path.stat().st_size, "sha256": sha(path)}


def features(a, sr):
    freq, ps = signal.welch(a, fs=sr, nperseg=min(512, len(a)))
    band = ps[freq >= 80]
    return {
        "rms_dbfs": float(20 * np.log10(max(np.sqrt(np.mean(a * a)), 1e-15))),
        "peak_abs_pcm_normalized": float(np.abs(a).max()),
        "zero_crossing_fraction": float(np.mean(a[:-1] * a[1:] < 0)),
        "spectral_flatness_above_80hz": float(np.exp(np.log(np.maximum(band, 1e-30)).mean()) / max(band.mean(), 1e-30)),
        "power_fraction_above_4000hz": float(ps[freq >= 4000].sum() / max(ps.sum(), 1e-30)),
        "power_fraction_below_1000hz": float(ps[freq < 1000].sum() / max(ps.sum(), 1e-30)),
        "power_fraction_1000_4000hz": float(ps[(freq >= 1000) & (freq < 4000)].sum() / max(ps.sum(), 1e-30)),
        "near_full_scale_fraction": float(np.mean(np.abs(a) >= 32760 / 32768)),
    }


def render_signal(x, sr, path, label):
    """Static diagnostic raster from SciPy PSD, not a speech-quality score."""
    duration = len(x) / sr
    width, height = 1300, 820
    left, right, plotwidth = 95, 1255, 1160
    im = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(im)
    d.text((left, 18), label + " - waveform and spectrogram (no denoising)", fill="black")
    d.text((left, 39), f"Mono PCM16 | {sr} Hz | {len(x)} samples | {duration:.6f} s", fill="black")
    top, bottom = 95, 290
    middle = (top + bottom) / 2
    d.line((left, middle, right, middle), fill="#aaaaaa")
    edges = np.linspace(0, len(x), plotwidth + 1).astype(int)
    for i in range(plotwidth):
        a = x[edges[i]:edges[i + 1]]
        d.line((left + i, middle - float(a.max()) * 92, left + i, middle - float(a.min()) * 92), fill="#246bb2")
    d.text((left, 73), "Waveform: min/max per display column; normalized PCM [-1, 1]", fill="black")
    d.text((35, top), "+1", fill="black")
    d.text((35, middle), "0", fill="black")
    d.text((35, bottom - 10), "-1", fill="black")

    freq, times, ps = signal.spectrogram(x, fs=sr, window="hann", nperseg=1024,
                                        noverlap=768, detrend=False, scaling="density", mode="psd")
    db = 10 * np.log10(np.maximum(ps, 1e-30))
    values = np.clip((db - db.max() + 80) / 80, 0, 1)
    # Display-only color mapping, fixed -80..0 dB relative to this file's maximum PSD.
    colors = np.array([[10, 10, 35], [38, 48, 120], [50, 130, 150], [170, 185, 95], [255, 245, 170]], dtype=float)
    grid = np.linspace(0, 1, len(colors))
    rgb = np.stack([np.interp(values, grid, colors[:, c]) for c in range(3)], axis=-1).astype(np.uint8)
    y0, y1 = 385, 705
    d.text((left, 350), "Spectrogram: Hann 1024, hop 256; colors -80..0 dB relative to peak PSD", fill="black")
    # Locate columns at their actual window-center times (do not stretch to full WAV length).
    xa = left + round(times[0] / duration * plotwidth)
    xb = left + round(times[-1] / duration * plotwidth)
    im.paste(Image.fromarray(rgb[::-1]).resize((xb - xa, y1 - y0)), (xa, y0))
    d = ImageDraw.Draw(im)
    for hz in [0, 2000, 4000, 6000, 8000, 10000]:
        y = y1 - hz / (sr / 2) * (y1 - y0)
        d.text((25, y - 5), str(hz), fill="black")
    for t in range(9):
        xp = left + t / duration * plotwidth
        for a, b in [(top, bottom), (y0, y1)]:
            d.line((xp, a, xp, b), fill="#d04444" if t == 8 else "#dddddd", width=2 if t == 8 else 1)
        d.text((xp - 3, bottom + 8), str(t), fill="black")
        d.text((xp - 3, y1 + 8), str(t), fill="black")
    d.text((left, 747), f"Time (seconds). Red line = 8 s; {max(0, duration-8):.6f} s remains after this line.", fill="black")
    d.text((left, 769), "Spectral energy is not a measure of intelligibility. Each file uses its own peak PSD scale.", fill="black")
    im.save(path)


def analyze_wav(path, prefix):
    path = path.resolve()
    original_hash = sha(path)
    with wave.open(str(path), "rb") as wav:
        sr, n, channels, width = wav.getframerate(), wav.getnframes(), wav.getnchannels(), wav.getsampwidth()
        assert (channels, width, wav.getcomptype()) == (1, 2, "NONE"), "Expected mono PCM16"
        raw = wav.readframes(n)
        assert len(raw) == n * channels * width, "Truncated WAV data"
    pcm = np.frombuffer(raw, dtype="<i2")
    x = pcm.astype(np.float64) / 32768
    windows = []
    for start_s in np.arange(0, n / sr, 0.5):
        start, end = int(start_s * sr), min(int((start_s + 0.5) * sr), n)
        windows.append({"start_s": start / sr, "end_s": end / sr, "samples": end - start,
                        **features(x[start:end], sr)})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{prefix}-windows.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(windows[0]))
        writer.writeheader()
        writer.writerows(windows)
    render_signal(x, sr, OUT / f"{prefix}-signal.png", path.name)
    audio = {
        "source": source(path), "sample_rate": sr, "samples": n, "channels": channels,
        "pcm_bits": width * 8, "seconds": n / sr, "seconds_after_8": max(0, n / sr - 8),
        "sample_min": int(pcm.min()), "sample_max": int(pcm.max()),
        "near_full_scale_samples": int(np.count_nonzero(np.abs(pcm.astype(np.int32)) >= 32760)),
        "near_full_scale_is_not_proof_of_model_clipping": True,
        "whole_signal": features(x, sr),
        "segments": [{"start_s": a, "end_s": min(b, n / sr), **features(x[int(a*sr):min(int(b*sr), n)], sr)}
                     for a, b in [(0, 4), (4, 8), (0, 6.4), (6.4, 8), (7.5, 8), (8, n / sr)] if a < n/sr],
        "segment_selection_note": "6.4 s was selected by visual inspection of the supplied PTQ spectrogram; not a validated change-point detector",
        "precision_verified_from_session_json": False,
        "listening_assessment": "not performed",
        "conclusion": "One WAV cannot establish a systematic duration-dependent failure threshold.",
    }
    assert sha(path) == original_hash, "Source WAV changed during analysis"
    return audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    audio = analyze_wav(args.wav, "demo-ptq")
    reference = analyze_wav(args.reference, "demo-fp32-seq")
    stats, inputs = [], []
    for model in MODELS:
        for precision in ["fp32", "int8"]:
            rows, files = [], []
            for repeat in range(3):
                f = RUNS / f"r{repeat}_{model}_{precision}_t4/utterances.jsonl"
                records = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
                assert len(records) == 720, (f, len(records))
                assert len({r['idx'] for r in records}) == 720
                rows.extend(records)
                files.append(source(f))
            seconds = np.array([r["audio_seconds"] for r in rows])
            assert np.allclose(seconds, np.array([r["samples"] for r in rows]) / 22050)
            stats.append({"model": model, "precision": precision, "measurement_count": len(rows),
                          "unique_utterance_ids": len({r['idx'] for r in rows}),
                          "min_s": float(seconds.min()), "median_s": float(np.median(seconds)),
                          "p95_s": float(np.percentile(seconds, 95)), "max_s": float(seconds.max()),
                          "count_above_8_s": int((seconds > 8).sum()), "count_above_10_s": int((seconds > 10).sum()),
                          "quality_assessment": "Timing records only, not speech quality certification", "inputs": files})
            inputs.extend(files)
    distributions = {}
    for name, filename in [("calibration", "calibration60.candidate.jsonl"), ("harvard", "harvard720.json")]:
        f = EVIDENCE / "metadata" / filename
        txt = f.read_text(encoding="utf-8")
        rows = [json.loads(line) for line in txt.splitlines() if line.strip()] if f.suffix == ".jsonl" else json.loads(txt)
        lengths = [len(r["phoneme_ids"]) for r in rows]
        distributions[name] = {"rows": len(rows), "min_ids": min(lengths), "median_ids": float(np.median(lengths)),
                               "max_ids": max(lengths), "generated_audio_durations": "not inferred from token lengths", "source": source(f)}
        inputs.append(source(f))
    code = ["src/hifimobinet/n100_demo.py", "src/hifimobinet/n100_server.py", "src/hifimobinet/frontend.py",
            "demo/n100_web/app.js", "demo/n100_models.json", "scripts/evaluation/onnx/common.py",
            "vendor/edgetts/vits/models.py", "results/edge-n100-openvino/2026-11-01/evidence/openvino_suite.py"]
    inputs += [source(ROOT / s) for s in code]
    inputs += [source(EVIDENCE / "work-n100-clean-20261101T075458Z" / m / "preparation.json") for m in MODELS]
    for entry in inputs:
        assert sha(ROOT / entry["path"]) == entry["sha256"], entry["path"]
    for wav_record in [audio, reference]:
        assert sha(Path(wav_record["source"]["path"])) == wav_record["source"]["sha256"]
    result = {"analysis_utc": datetime.now(timezone.utc).isoformat(),
              "method": "PCM/SciPy signal analysis and fresh parsing of existing records; no model execution",
              "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
              "user_association": {"model": "SeqIR", "text": args.text,
                                   "ptq_file": args.wav.name, "fp32_file": args.reference.name,
                                   "association_source": "user reply; session JSON not yet supplied"},
              "user_reported_affected_models": ["SeqIR", "ParIR"],
              "parir_paired_wavs_supplied": False,
              "audio": audio, "fp32_reference": reference,
              "paired_comparison": {"same_sample_rate": audio["sample_rate"] == reference["sample_rate"],
                                    "same_sample_count": audio["samples"] == reference["samples"],
                                    "duration_difference_s": audio["seconds"] - reference["seconds"],
                                    "samplewise_error_not_computed": "Native graph RNG and phoneme timing need not be matched"},
              "historical_durations": stats, "phoneme_id_lengths": distributions,
              "input_sources": inputs, "source_hashes_unchanged_after_analysis": True}
    (OUT / "inspection.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audio": audio, "fp32_reference": reference,
                      "historical_durations": [{k: v for k, v in s.items() if k != "inputs"} for s in stats],
                      "phoneme_id_lengths": distributions}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
