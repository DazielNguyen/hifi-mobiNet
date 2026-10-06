"""Training-harness checks (CPU). Functional tests, not research experiments.

Optional dependencies are skipped explicitly: PyTorch Lightning for module
tests and the compiled Piper MAS extension for tests that import upstream
Piper's models.py (it imports MAS at module import time).
"""
import importlib.util
import json
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "training"

torch = pytest.importorskip("torch")
pytest.importorskip("yaml")

from hifimobinet.training.config import ConfigError, load_config  # noqa: E402
from hifimobinet.training.registry import TRAINING_MODELS, get_training_model  # noqa: E402


def _edgetts_mas_built() -> bool:
    return any((ROOT / "vendor/edgetts/vits/monotonic_align/monotonic_align").glob("core*"))


def _banhmi_mas_built() -> bool:
    return any((ROOT / "src/hifimobinet/architecture/vits/utils/monotonic_align").glob("core*.so")) or any(
        (ROOT / "src/hifimobinet/architecture/vits/utils/monotonic_align").glob("core*.pyd"))


needs_lightning = pytest.mark.skipif(importlib.util.find_spec("pytorch_lightning") is None,
                                     reason="pytorch_lightning not installed")
needs_edgetts_mas = pytest.mark.skipif(not _edgetts_mas_built(), reason="EdgeTTS MAS extension not built")
needs_banhmi_mas = pytest.mark.skipif(not _banhmi_mas_built(), reason="internal MAS extension not built")


# ---------------------------------------------------------------- configs
@pytest.mark.parametrize("name", sorted(p.name for p in CONFIGS.glob("*.yaml")))
def test_shipped_configs_validate(name):
    config = load_config(CONFIGS / name)
    assert config.model_id in TRAINING_MODELS


def _write(tmp_path, mutate):
    import yaml
    data = yaml.safe_load((CONFIGS / "piper-no-vits2-cpn.yaml").read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_config_rejects_unknown_key(tmp_path):
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(_write(tmp_path, lambda d: d["optim"].update(learning_rat=1e-3)))


def test_config_rejects_missing_key(tmp_path):
    with pytest.raises(ConfigError, match="missing keys"):
        load_config(_write(tmp_path, lambda d: d["trainer"].pop("gradient_clip_val")))


def test_config_rejects_vits2_key_for_piper(tmp_path):
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(_write(tmp_path, lambda d: d["model"].update(mas_noise_scale_initial=0.01)))


@pytest.mark.parametrize("flag", ["use_bigvgan", "use_vits2", "use_f0"])
def test_config_a_rejects_component_flags(tmp_path, flag):
    with pytest.raises(ConfigError, match="Config A"):
        load_config(_write(tmp_path, lambda d: d["model"].update({flag: True})))


def test_config_a_rejects_trainer_clipping(tmp_path):
    with pytest.raises(ConfigError, match="grad_clip"):
        load_config(_write(tmp_path, lambda d: d["trainer"].update(gradient_clip_val=1.0)))


def test_config_rejects_unknown_model_id(tmp_path):
    with pytest.raises(ConfigError, match="model_id"):
        load_config(_write(tmp_path, lambda d: d.update(model_id="piper-original")))


def test_registry_has_no_fallback():
    with pytest.raises(KeyError):
        get_training_model("baseline-resblock2")  # release checkpoint ID, not a training recipe


# ---------------------------------------------------------------- data
def _tiny_dataset(tmp_path, n=6):
    rows, keys = [], []
    for i in range(n):
        frames = 40 + 3 * i
        audio = torch.rand(1, frames * 256) * 2 - 1
        spec = torch.rand(513, frames)
        a, s = f"tensors/u{i}.pt", f"tensors/u{i}.spec.pt"
        (tmp_path / "tensors").mkdir(exist_ok=True)
        torch.save(audio, tmp_path / a)
        torch.save(spec, tmp_path / s)
        rows.append({"text": f"utt {i}", "phoneme_ids": [1, 0] + [5 + i % 7, 0] * (4 + i) + [2],
                     "audio_norm_path": a, "audio_spec_path": s})
        keys.append(a)
    (tmp_path / "dataset.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (tmp_path / "split.json").write_text(json.dumps({"train": keys[:4], "val": keys[4:5], "test": keys[5:]}),
                                         encoding="utf-8")
    return keys


def test_split_overlap_rejected(tmp_path):
    from hifimobinet.training.data import load_split
    (tmp_path / "s.json").write_text(json.dumps({"train": ["a", "b"], "val": ["c"], "test": ["a"]}))
    with pytest.raises(ValueError, match="both"):
        load_split(tmp_path / "s.json")


@pytest.mark.parametrize("family", ["banhmi", "edgetts"])
def test_training_data_excludes_test_and_resolves_paths(tmp_path, family):
    from hifimobinet.training.data import load_split, load_training_data
    keys = _tiny_dataset(tmp_path)
    data = load_training_data(family, tmp_path / "dataset.jsonl", tmp_path, load_split(tmp_path / "split.json"))
    used = {data.keys[i] for i in list(data.train.indices) + list(data.val.indices)}
    assert keys[5] not in used and len(data.train) == 4 and len(data.val) == 1
    assert all(Path(u.audio_norm_path).is_absolute() for u in data.full.utterances)


def test_edgetts_batches_match_edgetts_collate_without_f0(tmp_path):
    """The F0-free batches equal EdgeTTS's own collate on every non-F0 field."""
    from hifimobinet.training.data import load_split, load_training_data
    from hifimobinet.training.vendor import vendor_module
    _tiny_dataset(tmp_path)
    data = load_training_data("edgetts", tmp_path / "dataset.jsonl", tmp_path, load_split(tmp_path / "split.json"))
    items = [data.train[i] for i in range(4)]
    ours = data.collate(8192)(items)
    ds = vendor_module("edgetts_vits", "dataset")
    theirs = ds.UtteranceCollate(False, 8192)([
        ds.UtteranceTensors(phoneme_ids=u.phoneme_ids, spectrogram=u.spectrogram, audio_norm=u.audio_norm,
                            f0=torch.rand(u.spectrogram.shape[1])) for u in items])
    assert isinstance(ours, ds.Batch) and ours.f0s is None and ours.speaker_ids is None
    for field in ("phoneme_ids", "phoneme_lengths", "spectrograms", "spectrogram_lengths", "audios", "audio_lengths"):
        assert torch.equal(getattr(ours, field), getattr(theirs, field)), field


# ---------------------------------------------------------------- harness
def test_rng_state_is_weights_only_loadable(tmp_path):
    import numpy as np
    from hifimobinet.training.harness import capture_rng_state, restore_rng_state
    state = capture_rng_state()
    expected = (random.random(), float(np.random.rand()), torch.rand(3))
    torch.save({"rng": state}, tmp_path / "s.pt")
    loaded = torch.load(tmp_path / "s.pt", weights_only=True)["rng"]
    assert restore_rng_state(loaded)
    again = (random.random(), float(np.random.rand()), torch.rand(3))
    assert expected[0] == again[0] and expected[1] == again[1] and torch.equal(expected[2], again[2])


def _module(name, config):
    cfg = load_config(CONFIGS / config)
    return get_training_model(cfg.model_id).builder()(num_symbols=256, **cfg.flat_hparams())


@needs_lightning
@needs_edgetts_mas
def test_piper_model_is_config_a_without_vits2():
    module = _module("Piper_no_VITS2_cpn", "piper-no-vits2-cpn.yaml")
    classes = {type(m).__name__ for m in module.modules()}
    assert not any(m in c for c in classes for m in ("DurationDiscriminator", "Transformer", "MultiResolution",
                                                       "Snake", "F0Predictor"))
    assert not any("Attention" in type(m).__name__ for m in module.model_g.flow.modules())
    assert any("Attention" in type(m).__name__ for m in module.model_g.enc_p.modules())  # Piper text encoder keeps it
    assert module.model_d_dur is None and module.model_d_mrd is None
    assert not module.model_g.use_noised_mas and not module.model_g.use_f0 and not module.model_g.dec.use_snake
    assert [n for n, _ in module.named_children()] == ["model_g", "model_d"]


@needs_lightning
@needs_edgetts_mas
def test_piper_model_inherits_edgetts_training_unchanged():
    from hifimobinet.training.piper_module import EdgeTTSVitsModel, PiperNoVits2Module
    for name in ("training_step", "training_step_g", "training_step_d", "configure_optimizers",
                 "on_train_epoch_end", "forward"):
        assert getattr(PiperNoVits2Module, name) is getattr(EdgeTTSVitsModel, name), name
    assert _module("Piper_no_VITS2_cpn", "piper-no-vits2-cpn.yaml").automatic_optimization is False


@needs_lightning
@needs_banhmi_mas
def test_baseline_model_keeps_vits2_components():
    module = _module("baseline-resblock2-vits2", "baseline-resblock2-vits2.yaml")
    classes = {type(m).__name__ for m in module.modules()}
    assert "DurationDiscriminator" in classes and "TransformerCouplingLayer" in classes
    assert module.model_d_mrd is None


@needs_lightning
@needs_edgetts_mas
def test_piper_generator_loss_runs_without_f0():
    from hifimobinet.training.data import EdgeTTSBatchCollate
    from hifimobinet.training.vendor import vendor_module
    module = _module("Piper_no_VITS2_cpn", "piper-no-vits2-cpn.yaml")
    module.log = lambda *a, **k: None
    tensors = vendor_module("banhmi", "vits.dataset").UtteranceTensors
    torch.manual_seed(7)
    utts = [tensors(phoneme_ids=torch.randint(1, 100, (n,)), spectrogram=torch.rand(513, f),
                    audio_norm=torch.rand(1, f * 256) * 2 - 1) for n, f in ((21, 48), (17, 40))]
    loss_g, loss_mel = module.training_step_g(EdgeTTSBatchCollate(8192)(utts))
    assert torch.isfinite(loss_g).all() and torch.isfinite(loss_mel).all()


@needs_lightning
def test_lr_scheduler_hook_matches_lightning_default():
    from hifimobinet.training.harness import HarnessMixin
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=1.0)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.5)
    HarnessMixin.lr_scheduler_step(None, sched, 0, None)
    assert opt.param_groups[0]["lr"] == 0.5


@needs_lightning
def test_train_non_empty_run_dir_guard_only_in_launcher(tmp_path, monkeypatch):
    from hifimobinet.training.train import main
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_record.json").write_text("{}")
    argv = ["--model-id", "Piper_no_VITS2_cpn", "--config", str(CONFIGS / "piper-no-vits2-cpn.yaml"),
            "--dataset-dir", str(tmp_path / "missing"), "--data-root", str(tmp_path),
            "--split", str(tmp_path / "missing.json"), "--run-dir", str(run_dir)]
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    with pytest.raises(SystemExit, match="not empty"):
        main(argv)
    # A DDP child (rank >= 1) finds run_dir populated by rank 0 and must continue.
    monkeypatch.setenv("LOCAL_RANK", "1")
    with pytest.raises(FileNotFoundError):
        main(argv)


@needs_lightning
def test_run_dir_must_not_be_tracked_content():
    import shutil
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("needs a Git checkout")
    from hifimobinet.training.train import _outside_git_content
    assert _outside_git_content(ROOT / "training_output" / "some-run")  # ignored
    assert _outside_git_content(ROOT / "data" / "ljspeech-medium")  # ignored
    assert not _outside_git_content(ROOT / "docs" / "some-run")  # tracked area
    assert _outside_git_content(ROOT.parent / "elsewhere")


def test_train_cli_rejects_config_only_flags():
    pytest.importorskip("pytorch_lightning")
    from hifimobinet.training.train import main
    with pytest.raises(SystemExit, match="config file only"):
        main(["--precision", "16"])
