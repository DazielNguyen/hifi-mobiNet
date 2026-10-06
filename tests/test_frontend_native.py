"""Optional native frontend check against previously stored phoneme IDs."""
from pathlib import Path
import json
import pytest
pytest.importorskip('banhmi_phonemize')
from hifimobinet.frontend import encode


def test_first_three_harvard_encodings_match_recorded_ids():
    root=Path(__file__).resolve().parents[1]
    rows=json.loads((root/'results/manifests/harvard720.json').read_text(encoding='utf-8'))
    for row in rows[:3]:assert encode(row['text'])==row['phoneme_ids']
