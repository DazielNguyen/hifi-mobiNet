"""Score synthesized Harvard-720 WAVs with UTMOSv2 and Whisper-small WER.

Scoring code is the historical BanhmiTTS scripts/harvard/score.py (archived in the
journal evidence package) with paths turned into arguments:

- UTMOSv2 ``create_model(pretrained=True, device="cpu")``, one ``predict`` per WAV
  on the native SoundFile float32 waveform (no extra seeding, as before);
  the sample rate must match the synthesis record. UTMOS handles any resampling;
- Whisper ``small`` on CPU, ``transcribe(<wav path>, language="en", fp16=False)``,
  other decoding arguments at their defaults, as before;
- WER per sentence with JiWER: lowercase, remove punctuation, collapse spaces,
  strip, split into words.

Additions that do not change any score: the WAV SHA-256 recorded at synthesis is
checked before scoring; Hugging Face access is forced offline so cached scorer
weights are used and nothing is downloaded; scorer weight hashes and package
versions are recorded.

    python scripts/evaluation/harvard/score.py --synth <synth.json> --output <results.json> --record <score_record.json>

``--output`` keeps the historical per-sentence schema (a JSON array; plus ``wav_sha256``),
so ``scripts/evaluation/audit_results.py`` and ``compare.py`` read it directly;
``--record`` holds the scorer identities and environment.

Run from the directory the synthesis was run from (WAV paths are relative to it).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

def read_wave(path: Path, expected_sr: int):
    """Reject mislabeled, empty, stereo or non-finite audio before a scorer sees it."""
    import numpy as np
    import soundfile as sf

    wave, sr = sf.read(path, dtype="float32")
    if sr != expected_sr:
        raise ValueError(f"{path}: sample rate {sr} differs from synthesis {expected_sr}")
    if wave.ndim != 1 or not wave.size or not np.isfinite(wave).all():
        raise ValueError(f"{path}: expected nonempty finite mono audio")
    return wave, sr


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synth", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--record", required=True, type=Path)
    p.add_argument("--cer", action="store_true", help="also record character error rate with the same text normalization")
    p.add_argument("--progress", type=Path, help="new JSONL file flushed after each sentence; never used as final results")
    a = p.parse_args(argv)
    if a.output.exists() or a.record.exists():
        p.error("output or record exists; choose new paths")
    if a.progress and a.progress.exists():
        p.error("progress file exists; choose a new path")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import importlib.metadata as md

    import jiwer
    import numpy as np
    import soundfile as sf
    import torch
    import utmosv2
    import whisper
    from utmosv2.utils._constants import _UTMOSV2_CHACHE

    synth = json.loads(a.synth.read_text(encoding="utf-8"))
    rows_in = synth["rows"]
    sample_rate = int(synth["protocol"]["sample_rate"])
    wav_dir = Path(synth["wav_dir"])  # relative paths resolve against the current directory, as at synthesis
    bad = [r["idx"] for r in rows_in if sha256(wav_dir / r["wav"]) != r["wav_sha256"]]
    if bad:
        raise SystemExit(f"WAV hashes differ from the synthesis record: {bad[:10]}")

    whisper_weights = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "whisper" / "small.pt"
    utmos_weights = _UTMOSV2_CHACHE / "models" / "fusion_stage3" / "fold0_s42_best_model.pth"
    for weights in (whisper_weights, utmos_weights):
        if not weights.is_file():
            raise FileNotFoundError(f"offline scorer weights missing: {weights}")
    utmos_model = utmosv2.create_model(pretrained=True, device="cpu")
    asr = whisper.load_model("small", device="cpu")
    transform = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(),
                               jiwer.Strip(), jiwer.ReduceToListOfListOfWords()])
    char_transform = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(),
                                    jiwer.Strip(), jiwer.ReduceToListOfListOfChars()]) if a.cer else None
    if a.progress:
        a.progress.parent.mkdir(parents=True, exist_ok=True)
        a.progress.touch(exist_ok=False)

    results = []
    t_all = time.perf_counter()
    for i, s in enumerate(rows_in):
        wav_path = str(wav_dir / s["wav"])
        wave, sr_read = read_wave(Path(wav_path), sample_rate)
        u2 = float(utmos_model.predict(data=wave, sr=sr_read, device="cpu"))
        hyp = asr.transcribe(wav_path, language="en", fp16=False)["text"]
        wer = jiwer.wer(s["text"], hyp, reference_transform=transform, hypothesis_transform=transform)
        results.append({**{k: s[k] for k in ("idx", "text", "synth_time_s", "audio_duration_s", "rtf")},
                        "asr_text": hyp, "utmosv2": u2, "wer": wer, "wav_sha256": s["wav_sha256"]})
        if a.cer:
            results[-1]["cer"] = jiwer.cer(s["text"], hyp, reference_transform=char_transform,
                                           hypothesis_transform=char_transform)
        if not all(np.isfinite(results[-1][k]) for k in ("utmosv2", "wer", "rtf")):
            raise ValueError(f"sentence {s['idx']}: non-finite score")
        if a.progress:
            with a.progress.open("a", encoding="utf-8") as progress:
                progress.write(json.dumps(results[-1], ensure_ascii=False, allow_nan=False) + "\n")
        if (i + 1) % 50 == 0 or i == len(rows_in) - 1:
            el = time.perf_counter() - t_all
            print(f"[score {i + 1}/{len(rows_in)}] {el / 60:.1f} min ETA {el / (i + 1) * (len(rows_in) - i - 1) / 60:.1f} min",
                  flush=True)

    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "synth_record": a.synth.name, "synth_record_sha256": sha256(a.synth),
        "model_id": synth["model_id"], "checkpoint": synth["checkpoint"], "checkpoint_sha256": synth["checkpoint_sha256"],
        "scorers": {
            "utmosv2": {"package": md.version("utmosv2"), "config": "fusion_stage3", "fold": 0, "weights": utmos_weights.name,
                        "weights_sha256": sha256(utmos_weights), "repetitions_per_wav": 1, "explicit_seed": None},
            "whisper": {"package": md.version("openai-whisper"), "model": "small", "weights_sha256": sha256(whisper_weights),
                        "language": "en", "fp16": False, "input": "WAV path (Whisper resamples to 16 kHz)"},
            "jiwer": md.version("jiwer"),
            "wer_transform": "ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip, ReduceToListOfListOfWords",
            "hf_offline": True,
            "source_sample_rate": sample_rate,
            "utmos_input_sample_rate": sample_rate,
            "utmos_internal_sample_rate": int(utmos_model._cfg.sr),
            "resampling": "native WAV unchanged; UTMOS internal torchaudio Resample if rates differ; Whisper WAV loader to 16000 Hz",
            "cer_transform": "ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip, ReduceToListOfListOfChars" if a.cer else None,
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
                        "numpy": np.__version__, "soundfile": sf.__version__, "torch_threads": torch.get_num_threads()},
        "script_sha256": sha256(Path(__file__)),
        "wall_time_s": time.perf_counter() - t_all,
        "results_file": a.output.name,
        "summary": {
            "n": len(results), "mean_utmosv2": float(np.mean([r["utmosv2"] for r in results])),
            "macro_wer": float(np.mean([r["wer"] for r in results])),
            "corpus_wer": jiwer.wer([r["text"] for r in results], [r["asr_text"] for r in results],
                                    reference_transform=transform, hypothesis_transform=transform),
            "sentences_wer_zero": sum(r["wer"] == 0 for r in results),
        },
    }
    if a.cer:
        record["summary"].update(
            macro_cer=float(np.mean([r["cer"] for r in results])),
            corpus_cer=jiwer.cer([r["text"] for r in results], [r["asr_text"] for r in results],
                                 reference_transform=char_transform, hypothesis_transform=char_transform))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(results, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    record["results_sha256"] = sha256(a.output)
    a.record.parent.mkdir(parents=True, exist_ok=True)
    a.record.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    r = {k: np.array([x[k] for x in results]) for k in ("rtf", "utmosv2", "wer")}
    print(f"=== n={len(results)}: RTF={r['rtf'].mean():.4f} UTMOSv2={r['utmosv2'].mean():.3f} WER={r['wer'].mean():.4f} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
