"""Exercise Streamlit without opening a network-accessible public server."""
import hashlib
import io
import json
import wave
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from hifimobinet.comparison import audio_bytes

ROOT=Path(__file__).resolve().parents[1]


def test_missing_weights_and_audio_keep_comparison_ui_usable(monkeypatch,tmp_path):
    monkeypatch.setenv('HIFIMOBINET_MODEL_DIR',str(tmp_path/'missing-models'))
    monkeypatch.setenv('HIFIMOBINET_AUDIO_DIR',str(tmp_path/'missing-audio'))
    app=AppTest.from_file(str(ROOT/'demo/streamlit_app.py')).run(timeout=30)
    assert not app.exception
    assert len(app.selectbox[0].options)==3
    assert app.button[0].disabled
    assert any('unavailable' in i.value.lower() for i in app.info)
    app.selectbox[0].select('harvard-0001').run()
    assert not app.exception
    assert any('Glue the sheet' in x.value for x in app.markdown)


def test_audio_passthrough_and_corruption(monkeypatch,tmp_path):
    stream=io.BytesIO()
    with wave.open(stream,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(22050);w.writeframes(b'\x00\x00'*40)
    data=stream.getvalue(); p=tmp_path/'sample.wav';p.write_bytes(data)
    record={'relative_path':'sample.wav','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
            'sample_rate':22050,'channels':1,'frames':40}
    monkeypatch.setenv('HIFIMOBINET_AUDIO_DIR',str(tmp_path))
    assert audio_bytes(record,ROOT)==data
    p.write_bytes(data[:-1]+b'\x01')
    with pytest.raises(ValueError,match='checksum'): audio_bytes(record,ROOT)


def test_streamlit_audio_widget_without_model(monkeypatch,tmp_path):
    # This is a synthetic playback fixture, never a historical listening sample.
    import hifimobinet.comparison as comparison
    stream=io.BytesIO()
    with wave.open(stream,'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050); w.writeframes(b'\x00\x00'*40)
    monkeypatch.setattr(comparison,'audio_bytes',lambda *_args:stream.getvalue())
    monkeypatch.setenv('HIFIMOBINET_MODEL_DIR',str(tmp_path/'no-models'))
    app=AppTest.from_file(str(ROOT/'demo/streamlit_app.py')).run(timeout=30)
    assert not app.exception
    assert len(app.get('audio'))==4
    assert app.button[0].disabled
