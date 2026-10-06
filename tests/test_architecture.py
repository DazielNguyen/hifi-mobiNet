"""Optional CPU functional checks of relocated model components, not benchmarks."""
import importlib.util
import sys
from pathlib import Path
import pytest

torch=pytest.importorskip('torch')
from hifimobinet.architecture.factory import build_synthesizer
ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('model_id,mode,branches',[
    ('baseline-resblock2','2',9),('parallel-ir','mrf',3),('sequential-ir','mb',2)])
def test_architecture_and_original_generator_parity(model_id,mode,branches):
    torch.set_num_threads(1)
    model=build_synthesizer(model_id,ROOT).eval()
    assert model.enc_p.emb.weight.shape==(256,192)
    assert model.enc_q.pre.weight.shape[1]==513
    assert model.dec.resblock_mode==mode
    assert not model.use_f0
    assert all(hasattr(model,k) for k in ['enc_p','enc_q','dp','flow','dec'])
    if mode=='2': assert len(model.dec.resblocks)==branches
    else: assert all(len(s)==branches for s in model.dec.mb_blocks)
    spec=importlib.util.spec_from_file_location('original_banhmi',ROOT/'vendor/banhmi/__init__.py',submodule_search_locations=[str(ROOT/'vendor/banhmi')])
    module=importlib.util.module_from_spec(spec);sys.modules['original_banhmi']=module;spec.loader.exec_module(module)
    from original_banhmi.vits.modules.generator import Generator
    original=Generator(192,mode,(3,5,7),((1,2),(2,6),(3,12)),(8,8,4),256,(16,16,8),mb_expansion=(1,1,1)).eval()
    original.load_state_dict(model.dec.state_dict(),strict=True)
    torch.manual_seed(1234)
    x=torch.randn(1,192,2)
    with torch.inference_mode():
        a=original(x);b=model.dec(x)
    assert a.shape==b.shape==(1,1,512)
    assert torch.isfinite(b).all()
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_no_checkpoint_or_external_factory_fallback():
    with pytest.raises(ValueError):build_synthesizer('piper-original',ROOT)
