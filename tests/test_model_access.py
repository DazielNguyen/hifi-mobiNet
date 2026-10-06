"""Manifest-only access and checksum failure behavior, without network requests."""
import hashlib
import io
from pathlib import Path
import pytest
from hifimobinet.registry import model_path,ModelUnavailable
import hifimobinet.download as access


def test_weights_reject_wrong_bytes(monkeypatch,tmp_path):
    payload=b'fixture, not an actual model'
    record={'artifact':{'filename':'fixture.onnx','format':'onnx','precision':'FP32','bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}}
    monkeypatch.setenv('HIFIMOBINET_MODEL_DIR',str(tmp_path))
    with pytest.raises(ModelUnavailable):model_path(record)
    p=tmp_path/'fixture.onnx';p.write_bytes(payload)
    assert model_path(record)==p
    p.write_bytes(payload[:-1]+b'!')
    with pytest.raises(ModelUnavailable,match='checksum'):model_path(record)


def test_download_checksum_and_no_overwrite(monkeypatch,tmp_path):
    payload=b'small test payload';url='https://example.invalid/test'
    item={'filename':'fixture.onnx','bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'url':url}
    monkeypatch.setattr(access,'model_record',lambda _id:{'artifact':item})
    class Response(io.BytesIO):
        def geturl(self):return url
    monkeypatch.setattr(access.urllib.request,'urlopen',lambda *_a,**_kw:Response(payload))
    target=access.download('fixture','artifact',tmp_path)
    assert target.read_bytes()==payload
    assert not list(tmp_path.glob('*.partial'))
    target.write_bytes(b'other')
    with pytest.raises(ValueError,match='exists'):access.download('fixture','artifact',tmp_path)


def test_pending_url_does_not_make_network_request(monkeypatch,tmp_path):
    monkeypatch.setattr(access,'model_record',lambda _id:{'artifact':{'url':None}})
    with pytest.raises(ValueError,match='pending'):access.download('fixture','artifact',tmp_path)
