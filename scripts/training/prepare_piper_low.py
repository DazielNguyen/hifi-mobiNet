"""Derive a separate 16 kHz cache from the verified Medium cache, on CPU.

No model is instantiated and no training, phonemization or inference is run.
Preserve dataset.jsonl bytes/keys and the canonical split; --data-root selects
the new tensors. Existing output directories are refused, never overwritten.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import sys

SOURCE_DATASET_SHA = "f4a72ae0ed238dfdb3ca995b6678dfe33fe3b59297331d6551a4d6960102295b"
SOURCE_CONFIG_SHA = "4b751b911b4d42f5dc1a8c604e4c7cd5b19d298b9768ba07564e96696d665b6e"
SPLIT_SHA = "e678cf436075a10bb68c9a1b435f33c4a40ff1f095b7b78ac12f546c5f12ae9d"
DATASET_REL = Path("training_output/ljspeech/medium")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def contained(root, relative):
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"unsafe tensor path: {relative}")
    result = (root / p).resolve()
    result.relative_to(root.resolve())
    return result


def validate_membership(rows, split):
    keys = [r["audio_norm_path"] for r in rows]
    specs = [r["audio_spec_path"] for r in rows]
    partitions = [k for name in ("train", "val", "test") for k in split[name]]
    if len(set(keys + specs)) != len(keys) + len(specs):
        raise ValueError("duplicate or colliding tensor paths")
    if len(set(partitions)) != len(partitions) or set(keys) != set(partitions):
        raise ValueError("split must cover all rows exactly once, with no overlap")


def convert_audio(audio):
    import torch
    from scipy.signal import resample_poly
    if not isinstance(audio, torch.Tensor) or audio.ndim != 2 or audio.shape[0] != 1:
        raise ValueError("expected a mono [1, samples] tensor")
    if audio.dtype != torch.float32 or not torch.isfinite(audio).all() or audio.shape[1] <= 1024:
        raise ValueError("expected finite float32 audio longer than one FFT window")
    # Keep source trimming, loudness and phoneme IDs; only change sample rate.
    # No clipping/renormalization. Any FIR overshoot is measured in the record.
    result = resample_poly(audio.numpy(), 320, 441, axis=-1, window=("kaiser", 5.0), padtype="constant")
    out = torch.from_numpy(result.copy()).contiguous()
    if out.shape[-1] != math.ceil(audio.shape[-1] * 320 / 441) or not torch.isfinite(out).all():
        raise ValueError("resampling produced an invalid length or nonfinite values")
    return out


def verify(source, output, split_path):
    record = json.loads((output / "PREPARATION_RECORD.json").read_text())
    if record["status"] != "complete" or sha256(split_path) != record["split_sha256"]:
        raise ValueError("incomplete preparation or different split")
    if sha256(output / "TENSOR_MANIFEST.jsonl") != record["tensor_manifest_sha256"]:
        raise ValueError("tensor manifest hash mismatch")
    for name, expected in record["metadata_hashes"].items():
        if sha256(output / name) != expected:
            raise ValueError(f"metadata hash mismatch: {name}")
    for name, expected in record["source_metadata_hashes"].items():
        if sha256(source / DATASET_REL / name) != expected:
            raise ValueError(f"source metadata hash mismatch: {name}")
    count = 0
    for line in (output / "TENSOR_MANIFEST.jsonl").read_text().splitlines():
        entry = json.loads(line)
        if sha256(contained(source, entry["source_audio"])) != entry["source_sha256"]:
            raise ValueError(f"source audio hash mismatch: {entry['source_audio']}")
        for field in ("audio", "spectrogram"):
            item = entry[field]
            path = contained(output, item["path"])
            if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                raise ValueError(f"derived tensor mismatch: {path}")
        count += 1
    if count != record["rows"]:
        raise ValueError("manifest row count mismatch")
    return {"verified_rows": count, "verified_tensors": count * 2, "source_hashes": "pass", "derived_hashes": "pass"}


def prepare(source, output, split_path):
    import inspect
    import numpy
    import scipy
    import torch
    from hifimobinet.training.vendor import vendor_module

    ds = source / DATASET_REL
    for path, expected in ((ds / "dataset.jsonl", SOURCE_DATASET_SHA),
                           (ds / "config.json", SOURCE_CONFIG_SHA), (split_path, SPLIT_SHA)):
        if sha256(path) != expected:
            raise ValueError(f"unexpected source hash: {path}")
    config = json.loads((ds / "config.json").read_text())
    if config["audio"]["sample_rate"] != 22050:
        raise ValueError("source must be the 22050 Hz Medium cache")
    rows = [json.loads(line) for line in (ds / "dataset.jsonl").read_text().splitlines() if line.strip()]
    split = json.loads(split_path.read_text())
    validate_membership(rows, split)
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("source and destination must be separate trees")
    if output.exists():
        raise FileExistsError(f"refusing existing output (use --verify-only to check it): {output}")
    if shutil.disk_usage(output.parent).free < 40 * 2**30:
        raise ValueError("need at least 40 GiB free for a separate Low cache")
    # Resolve all source paths before creating output; fail on missing evidence.
    for row in rows:
        for field in ("audio_norm_path", "audio_spec_path"):
            if not contained(source, row[field]).is_file():
                raise FileNotFoundError(row[field])
    torch.set_num_threads(2)
    mel = vendor_module("banhmi", "mel_processing")
    sampler = vendor_module("banhmi", "vits.length_bucket_sampler")
    output.mkdir()
    write_json(output / "PREPARATION_PENDING.json", {"status": "incomplete", "source": str(source)})
    lengths, frames_by_audio = {}, {}
    source_samples = target_samples = overshoot_samples = 0
    peak = 0.0
    with (output / "TENSOR_MANIFEST.jsonl").open("w", encoding="utf-8") as manifest:
        for i, row in enumerate(rows, 1):
            raw = contained(source, row["audio_norm_path"]).read_bytes()
            src = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
            audio = convert_audio(src)
            spec = mel.spectrogram_torch(audio, n_fft=1024, hop_size=256, win_size=1024).squeeze(0)
            if spec.ndim != 2 or spec.shape[0] != 513 or not torch.isfinite(spec).all():
                raise ValueError(f"invalid spectrogram for {row['audio_norm_path']}")
            entry = {"source_audio": row["audio_norm_path"], "source_sha256": hashlib.sha256(raw).hexdigest(),
                     "source_samples": src.shape[-1], "target_samples": audio.shape[-1], "frames": spec.shape[-1]}
            for field, key, tensor in (("audio", "audio_norm_path", audio), ("spectrogram", "audio_spec_path", spec)):
                dest = contained(output, row[key])
                dest.parent.mkdir(parents=True, exist_ok=True)
                torch.save(tensor, dest)
                saved = torch.load(dest, map_location="cpu", weights_only=True)
                if not torch.equal(tensor, saved):
                    raise ValueError(f"tensor readback failed: {dest}")
                entry[field] = {"path": row[key], "bytes": dest.stat().st_size, "sha256": sha256(dest)}
            manifest.write(json.dumps(entry) + "\n")
            lengths[row["audio_spec_path"]] = frames_by_audio[row["audio_norm_path"]] = spec.shape[-1]
            source_samples += src.numel()
            target_samples += audio.numel()
            overshoot_samples += int((audio.abs() > 1).sum())
            peak = max(peak, float(audio.abs().max()))
            if i % 500 == 0:
                print(f"prepared and readback-verified {i}/{len(rows)} utterances", flush=True)
    target_ds = output / DATASET_REL
    target_ds.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ds / "dataset.jsonl", target_ds / "dataset.jsonl")
    config["audio"].update(sample_rate=16000, quality="low")
    write_json(target_ds / "config.json", config)
    write_json(target_ds / ".spectrogram_lengths_cache.json", lengths)
    batches = len(sampler.LengthBucketBatchSampler(
        [frames_by_audio[k] for k in split["train"]], batch_size=16,
        boundaries=(150, 250, 350, 450, 600, 800, 1000), seed=1234, num_replicas=2, rank=0))
    record = {
        "status": "complete", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_data_root": str(source), "output_data_root": str(output), "dataset_dir_relative": DATASET_REL.as_posix(),
        "source_metadata_hashes": {"dataset.jsonl": SOURCE_DATASET_SHA, "config.json": SOURCE_CONFIG_SHA},
        "metadata_hashes": {(DATASET_REL / name).as_posix(): sha256(target_ds / name)
                            for name in ("dataset.jsonl", "config.json", ".spectrogram_lengths_cache.json")},
        "split_sha256": SPLIT_SHA, "rows": len(rows), "split_counts": {k: len(split[k]) for k in ("train", "val", "test")},
        "source_rate": 22050, "target_rate": 16000, "source_samples": source_samples, "target_samples": target_samples,
        "resampler": "scipy.signal.resample_poly(up=320, down=441, window=('kaiser', 5.0), padtype='constant')",
        "normalization": "retain source normalization/trimming; no additional clipping or gain",
        "target_peak_abs": peak, "target_samples_above_unit_amplitude": overshoot_samples,
        "spectrogram": {"n_fft": 1024, "hop": 256, "win": 1024, "center": False, "reflect_pad_each_side": 384},
        "tensor_manifest_sha256": sha256(output / "TENSOR_MANIFEST.jsonl"),
        "batches_per_epoch_per_rank": batches, "optimizer_steps_per_batch": 2,
        "environment": {"python": sys.version.split()[0], "numpy": numpy.__version__, "scipy": scipy.__version__, "torch": torch.__version__},
        "code_sha256": {"prepare_piper_low.py": sha256(__file__), "mel_processing.py": sha256(inspect.getfile(mel)),
                        "length_bucket_sampler.py": sha256(inspect.getfile(sampler))},
        "not_copied": ["raw WAVs (audio_path retains source provenance)", "F0 (unused by Config A)"],
        "training_started": False,
    }
    write_json(output / "PREPARATION_RECORD.json", record)
    checked = verify(source, output, split_path)
    write_json(output / "VERIFICATION.json", {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **checked})
    (output / "PREPARATION_PENDING.json").unlink()
    print(json.dumps({**checked, "batches_per_epoch_per_rank": batches, "dataset_config_sha256": sha256(target_ds / "config.json")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-root", required=True, type=Path)
    parser.add_argument("--output-data-root", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    source, output, split = args.source_data_root.resolve(), args.output_data_root.resolve(), args.split.resolve()
    if args.verify_only:
        print(json.dumps(verify(source, output, split)))
    else:
        prepare(source, output, split)


if __name__ == "__main__":
    main()
