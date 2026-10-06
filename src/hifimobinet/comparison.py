"""Verified historical audio access, independent of model availability."""
from __future__ import annotations
import io
import json
import os
import wave
from pathlib import Path
from .registry import repository_root, sha256, within


def audio_catalog(root: Path | None = None) -> dict:
    root=root or repository_root()
    data=json.loads((root/'demo/audio_manifest.json').read_text(encoding='utf-8'))
    seen=set(); texts={}
    for sample in data['samples']:
        key=(sample['utterance_id'],sample['model_id'])
        if key in seen: raise ValueError('Duplicate utterance/model in audio manifest')
        seen.add(key)
        if sample['utterance_id'] in texts and texts[sample['utterance_id']]!=sample['text']:
            raise ValueError('Comparison entries do not share the same sentence')
        texts[sample['utterance_id']]=sample['text']
    return data


def audio_bytes(sample: dict, root: Path | None = None) -> bytes:
    root=root or repository_root()
    storage=Path(os.environ.get('HIFIMOBINET_AUDIO_DIR',root/'.local/audio')).resolve()
    path=within(storage,sample['relative_path'])
    if not path.is_file(): raise FileNotFoundError('Historical WAV unavailable in this installation')
    if path.stat().st_size!=sample['bytes'] or sha256(path)!=sample['sha256']:
        raise ValueError('Historical WAV checksum mismatch; playback refused')
    data=path.read_bytes()
    with wave.open(io.BytesIO(data),'rb') as wav:
        if (wav.getframerate(),wav.getnchannels(),wav.getnframes())!=(sample['sample_rate'],sample['channels'],sample['frames']):
            raise ValueError('WAV header differs from the recorded inventory')
    return data
