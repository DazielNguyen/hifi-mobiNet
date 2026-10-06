"""Safety and behavior checks that do not synthesize a research dataset."""
import ast
import io
import json
import sys
import types
import wave
from pathlib import Path
import numpy as np
import pytest
from hifimobinet.frontend import encode
from hifimobinet.inference import pcm16_wav
from hifimobinet.registry import ModelUnavailable, manifest, model_record, within

ROOT = Path(__file__).resolve().parents[1]


def test_model_mapping():
    data = manifest(ROOT)
    expected = {"baseline-resblock2":"2", "parallel-ir":"mrf", "sequential-ir":"mb", "piper-original":"2"}
    assert {m['id'] for m in data['models']} == set(expected)
    for m in data['models']:
        c = json.loads((ROOT/m['config']).read_text())['parameters']
        assert c['resblock'] == expected[m['id']]
        assert c['upsample_rates'] == [8,8,4]
        assert c.get('use_f0',False) is False and c.get('use_vocos',False) is False
        assert len(m['artifact']['sha256']) == 64


def test_unknown_ids_and_paths_are_rejected(tmp_path):
    with pytest.raises(ModelUnavailable): model_record('../../untrusted.ckpt', ROOT)
    with pytest.raises(ValueError): within(tmp_path, '../outside.onnx')
    with pytest.raises(ValueError): within(tmp_path, str(tmp_path.parent/'outside'))


def test_pcm_matches_original_conversion():
    import importlib.util
    p=ROOT/'vendor/banhmi/audio_utils.py'
    spec=importlib.util.spec_from_file_location('original_audio_utils',p)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    for x in [np.zeros(20,np.float32),np.array([-2,-.3,0,.5,2],np.float32),np.array([.001,-.001],np.float32)]:
        with wave.open(io.BytesIO(pcm16_wav(x,22050))) as wav:
            assert wav.getframerate()==22050 and wav.getnchannels()==1
            np.testing.assert_array_equal(np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2'),mod.audio_float_to_int16(x))
    with pytest.raises(ValueError): pcm16_wav(np.array([np.nan]),22050)


def test_frontend_calls_original_with_flattened_sentences(monkeypatch):
    calls=[]
    def phonemize(text,voice):
        calls.append((text,voice)); return [['a',' '],['b','.']]
    def ids(symbols):
        assert symbols==['a',' ','b','.']; return [1,0,14,0,3,0,15,0,10,0,2]
    monkeypatch.setitem(sys.modules,'banhmi_phonemize',types.SimpleNamespace(phonemize_espeak=phonemize,phoneme_ids_espeak=ids))
    assert encode('A. B.')==[1,0,14,0,3,0,15,0,10,0,2]
    assert calls==[('A. B.','en-us')]
    with pytest.raises(ValueError): encode(' ')
    with pytest.raises(ValueError): encode('a'*301)


def test_model_forward_ast_unchanged():
    # Namespace relocation and optional imports must not silently change the model.
    def functions(path):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(tree)
                if isinstance(n,ast.FunctionDef) and n.name in ('forward','infer')}
    for source in (ROOT/'vendor/banhmi/vits').rglob('*.py'):
        target=ROOT/'src/hifimobinet/architecture/vits'/source.relative_to(ROOT/'vendor/banhmi/vits')
        if target.exists(): assert functions(source)==functions(target),source
