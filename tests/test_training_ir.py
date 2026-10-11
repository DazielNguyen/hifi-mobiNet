"""Piper-ParIR / Piper-SeqIR: Config A backbone with the released IR decoder bundle."""
import importlib.util
import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/training"

torch = pytest.importorskip("torch")
pytest.importorskip("yaml")

from hifimobinet.training.config import ConfigError, load_config  # noqa: E402
from hifimobinet.training.registry import get_training_model  # noqa: E402

needs_lightning = pytest.mark.skipif(importlib.util.find_spec("pytorch_lightning") is None,
                                     reason="pytorch_lightning not installed")
needs_edgetts_mas = pytest.mark.skipif(
    not any((ROOT / "vendor/edgetts/vits/monotonic_align/monotonic_align").glob("core*")),
    reason="EdgeTTS MAS extension not built")

IR = {"Piper_ParIR_no_VITS2_cpn": ("mrf", "parallel-ir", [[(3, 1), (5, 1), (7, 1)]] * 3,
                                   {"mb_expansion": [1, 1, 1], "mb_mrf_kernel_sizes": [3, 5, 7]}),
      "Piper_SeqIR_no_VITS2_cpn": ("mb", "sequential-ir", [[(3, 1), (3, 1)]] * 3,
                                   {"mb_expansion": [1, 1, 1], "mb_kernel_size": 3, "mb_blocks_per_stage": 2})}


def ir_spec(model_id):
    """The IR config as specified: the Config A file with only the decoder block and use_mrd changed."""
    import yaml
    data = yaml.safe_load((CONFIGS / "piper-no-vits2-cpn.yaml").read_text(encoding="utf-8"))
    resblock, _, _, extra = IR[model_id]
    data["model_id"] = model_id
    data["model"].update(resblock=resblock, use_mrd=True, **extra)
    return data


def _write(tmp_path, data, mutate=lambda d: None):
    import yaml
    mutate(data)
    path = tmp_path / f"{data['model_id']}.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------- configs
@pytest.mark.parametrize("model_id", sorted(IR))
def test_ir_spec_validates(tmp_path, model_id):
    cfg = load_config(_write(tmp_path, ir_spec(model_id)))
    assert cfg.model_id == model_id and cfg.sections["model"]["resblock"] == IR[model_id][0]


@pytest.mark.parametrize("model_id", sorted(IR))
@pytest.mark.parametrize("mutate,match", [
    (lambda d: d["model"].update(use_vits2=True), "Config A"),
    (lambda d: d["model"].update(use_bigvgan=True), "Config A"),
    (lambda d: d["model"].update(use_mrd=False), "use_mrd"),
    (lambda d: d["model"].update(resblock="2"), "needs resblock"),
    (lambda d: d["model"].update(mb_expansion=[1, 1]), "mb_expansion"),
    (lambda d: d["trainer"].update(gradient_clip_val=1.0), "grad_clip"),
])
def test_ir_config_rejections(tmp_path, model_id, mutate, match):
    with pytest.raises(ConfigError, match=match):
        load_config(_write(tmp_path, ir_spec(model_id), mutate))


def test_ir_configs_reject_the_other_decoders_keys(tmp_path):
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(_write(tmp_path, ir_spec("Piper_ParIR_no_VITS2_cpn"), lambda d: d["model"].update(mb_kernel_size=3)))
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(_write(tmp_path, ir_spec("Piper_SeqIR_no_VITS2_cpn"), lambda d: d["model"].update(mb_mrf_kernel_sizes=[3])))


# ---------------------------------------------------------------- modules
def _module(config_path, seed=None):
    cfg = load_config(config_path)
    if seed is not None:
        torch.manual_seed(seed)
    return get_training_model(cfg.model_id).builder()(num_symbols=256, **cfg.flat_hparams())


def _ir_module(tmp_path, model_id, seed=None):
    return _module(_write(tmp_path, ir_spec(model_id)), seed)


@needs_lightning
@needs_edgetts_mas
@pytest.mark.parametrize("model_id", sorted(IR))
def test_ir_module_structure(tmp_path, model_id):
    resblock, _, expected, _ = IR[model_id]
    module = _ir_module(tmp_path, model_id)
    g, dec = module.model_g, module.model_g.dec
    assert [n for n, _ in module.named_children()] == ["model_g", "model_d", "model_d_mrd"]
    # VITS2 / F0 parts absent; MAS noise switched off at the source branch.
    classes = {type(m).__name__ for m in module.modules()}
    assert not any(k in c for c in classes for k in ("DurationDiscriminator", "TransformerCoupling", "F0Predictor"))
    assert not any("Attention" in type(m).__name__ for m in g.flow.modules())
    assert module.model_d_dur is None and g.use_noised_mas is False and g.use_f0 is False
    # Decoder: released IR blocks, pointwise expand kept at expansion 1, SnakeBeta only inside the decoder.
    assert type(dec).__name__ == "IRDecoder" and dec.resblock_mode == resblock
    assert [[(b.dw.kernel_size[0], b.pw_expand.out_channels // b.pw_expand.in_channels) for b in stage]
            for stage in dec.mb_blocks] == expected
    assert {type(m).__name__ for m in dec.modules() if "Snake" in type(m).__name__ or "LeakyReLU" in type(m).__name__} == {"SnakeBeta"}
    assert not [n for n, m in module.named_modules() if "Snake" in type(m).__name__ and not n.startswith("model_g.dec.")]
    # MRD: three resolutions, in the discriminator optimizer, not the generator's.
    assert type(module.model_d_mrd).__name__ == "MultiResolutionDiscriminator"
    assert [d.resolution for d in module.model_d_mrd.discriminators] == [(512, 50, 240), (1024, 120, 600), (2048, 240, 1200)]
    (opt_g, opt_d), _ = module.configure_optimizers()
    ids = lambda opt: {id(p) for grp in opt.param_groups for p in grp["params"]}
    mrd = {id(p) for p in module.model_d_mrd.parameters()}
    assert mrd <= ids(opt_d) and not mrd & ids(opt_g) and ids(opt_g) == {id(p) for p in g.parameters()}
    for key in ("resblock", "mb_expansion", "use_mrd"):
        assert key in module.hparams


@needs_lightning
@needs_edgetts_mas
@pytest.mark.parametrize("model_id", sorted(IR))
def test_ir_decoder_matches_released_ir_topology(tmp_path, model_id):
    from hifimobinet.architecture.factory import build_synthesizer
    released = build_synthesizer(IR[model_id][1]).dec.state_dict()
    ours = _ir_module(tmp_path, model_id).model_g.dec.state_dict()
    assert {k: tuple(v.shape) for k, v in ours.items()} == {k: tuple(v.shape) for k, v in released.items()}


@needs_lightning
@needs_edgetts_mas
@pytest.mark.parametrize("model_id", sorted(IR))
def test_ir_backbone_initialization_equals_config_a(tmp_path, model_id):
    a = _module(CONFIGS / "piper-no-vits2-cpn.yaml", seed=1234)
    ir = _ir_module(tmp_path, model_id, seed=1234)
    a_state, ir_state = a.state_dict(), ir.state_dict()
    shared = [k for k in a_state if not k.startswith("model_g.dec.")]
    assert all(k in ir_state and torch.equal(a_state[k], ir_state[k]) for k in shared)
    assert not any(k.startswith("model_d_mrd.") for k in a_state)


@needs_lightning
@needs_edgetts_mas
@pytest.mark.parametrize("model_id", sorted(IR))
def test_ir_losses_include_mrd_and_exclude_vits2(tmp_path, model_id):
    from hifimobinet.training.data import EdgeTTSBatchCollate
    from hifimobinet.training.vendor import vendor_module
    module = _ir_module(tmp_path, model_id, seed=7)
    logged = {}
    module.log = lambda name, value, **k: logged.__setitem__(name, float(value.detach() if torch.is_tensor(value) else value))
    tensors = vendor_module("banhmi", "vits.dataset").UtteranceTensors
    torch.manual_seed(7)
    utts = [tensors(phoneme_ids=torch.randint(1, 100, (n,)), spectrogram=torch.rand(513, f),
                    audio_norm=torch.rand(1, f * 256) * 2 - 1) for n, f in ((21, 48), (17, 40))]
    batch = EdgeTTSBatchCollate(8192)(utts)
    loss_g, loss_mel = module.training_step_g(batch)
    loss_d = module.training_step_d(batch)
    assert torch.isfinite(loss_g).all() and torch.isfinite(loss_mel).all() and torch.isfinite(loss_d).all()
    assert logged["loss_gen_mrd"] > 0 and logged["loss_disc_mrd"] > 0
    assert logged["loss_dur_gen"] == 0 and logged["loss_disc_dur"] == 0 and logged["loss_f0"] == 0
    loss_g.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.model_g.dec.parameters() if p.requires_grad)


@needs_lightning
@needs_edgetts_mas
def test_ir_decoder_output_length_and_f0_guard(tmp_path):
    dec = _ir_module(tmp_path, "Piper_SeqIR_no_VITS2_cpn").model_g.dec
    out = dec(torch.randn(2, 192, 10))
    assert tuple(out.shape) == (2, 1, 10 * 256)
    with pytest.raises(ValueError, match="F0"):
        dec(torch.randn(1, 192, 4), f0=torch.ones(1, 1, 4))


@needs_lightning
@needs_edgetts_mas
def test_mas_noise_is_gated_by_use_noised_mas_in_source():
    from hifimobinet.training.vendor import vendor_module
    src = inspect.getsource(vendor_module("edgetts_vits", "models").SynthesizerTrn.forward)
    gate = src.index("if self.use_noised_mas:")
    noise = src.index("torch.randn_like(neg_cent)")
    assert gate < noise and src.count("torch.randn_like(neg_cent)") == 1


@needs_lightning
@needs_edgetts_mas
def test_config_a_module_is_unchanged_by_the_ir_path():
    module = _module(CONFIGS / "piper-no-vits2-cpn.yaml")
    assert [n for n, _ in module.named_children()] == ["model_g", "model_d"]
    assert type(module.model_g.dec).__module__.endswith("edgetts_vits.models") and module.model_d_mrd is None
