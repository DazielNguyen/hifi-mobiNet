"""Low/High recipe invariants and approval gate. Never instantiate/train a model."""
import importlib.util
import json
import math
from pathlib import Path

import pytest
import yaml

from hifimobinet.training.config import ConfigError, load_config
from hifimobinet.training.registry import get_training_model

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/training"


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/training" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("tier", ["low", "high"])
def test_size_recipe_only_changes_intended_model_settings(tier):
    medium = load_config(CONFIGS / "piper-no-vits2-cpn.yaml")
    variant = load_config(CONFIGS / f"piper-{tier}-no-vits2-cpn.yaml")
    assert variant.model_id == f"Piper_{tier.title()}_no_VITS2_cpn"
    spec = get_training_model(variant.model_id)
    assert spec.family == "edgetts"
    assert spec.builder is get_training_model(medium.model_id).builder
    for section in ("optim", "trainer", "data", "harness"):
        assert variant.sections[section] == medium.sections[section]
    expected = medium.section("model")
    if tier == "low":
        expected.update(sample_rate=16000)
    else:
        expected.update(resblock="1", resblock_kernel_sizes=[3, 7, 11],
                        resblock_dilation_sizes=[[1, 3, 5]] * 3, upsample_rates=[8, 8, 2, 2],
                        upsample_initial_channel=512, upsample_kernel_sizes=[16, 16, 4, 4])
    assert variant.section("model") == expected
    assert math.prod(expected["upsample_rates"]) == expected["hop_length"]


@pytest.mark.parametrize("tier", ["low", "high"])
@pytest.mark.parametrize("flag", ["use_vits2", "use_bigvgan", "use_f0"])
def test_new_recipes_keep_config_a_guards(tmp_path, tier, flag):
    data = yaml.safe_load((CONFIGS / f"piper-{tier}-no-vits2-cpn.yaml").read_text())
    data["model"][flag] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="Config A"):
        load_config(path)


def test_controller_plans_use_separate_data_and_measured_budgets():
    plans = {tier: json.loads((CONFIGS / f"sequences/piper-{tier}-2026-10-30.json").read_text())
             for tier in ("low", "high")}
    low_record = json.loads((ROOT / "docs/training/piper-sizes-2026-10-30/low-preparation_record.json").read_text())
    assert plans["low"]["budget"]["batches_per_epoch_per_rank"] == low_record["batches_per_epoch_per_rank"]
    assert plans["high"]["budget"]["batches_per_epoch_per_rank"] == 392
    assert plans["low"]["expected"]["dataset_config_sha256"] == low_record["metadata_hashes"]["training_output/ljspeech/medium/config.json"]
    for tier, plan in plans.items():
        args = dict(zip(plan["train_args"][::2], plan["train_args"][1::2]))
        root = "data/ljspeech-medium" if tier == "high" else "data/ljspeech-low-16000-2026-10-30"
        assert args["--data-root"] == root
        assert args["--dataset-dir"].startswith(root + "/")
        assert args["--length-cache"].startswith(root + "/")
        assert args["--split"] == "results/manifests/canonical_split.json"
        assert args["--max_epochs"] == "1500" and args["--devices"] == "2"
        assert plan["expected"]["split_sha256"] == low_record["split_sha256"]
        run = plan["runs"][0]
        assert load_config(ROOT / run["config"]).model_id == run["model_id"]
        assert "--resume" not in plan["train_args"]
    assert plans["low"]["state_dir"] != plans["high"]["state_dir"]
    assert plans["low"]["runs"][0]["run_dir"] != plans["high"]["runs"][0]["run_dir"]


def test_low_preparation_preserves_identity_and_detects_corruption(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("scipy")
    low = script("prepare_piper_low")
    source, output = tmp_path / "medium", tmp_path / "low"
    ds = source / low.DATASET_REL
    ds.mkdir(parents=True)
    rows = []
    for i in range(3):
        # Include an out-of-band component to exercise anti-alias filtering.
        t = torch.arange(22050) / 22050
        audio = (0.4 * torch.sin(2 * math.pi * 440 * t) + 0.2 * torch.sin(2 * math.pi * 10000 * t))[None]
        norm, spec = f"a{i}.pt", f"s{i}.pt"
        torch.save(audio, source / norm)
        torch.save(torch.zeros(513, 86), source / spec)
        rows.append({"audio_norm_path": norm, "audio_spec_path": spec, "text": f"text {i}", "phoneme_ids": [1, 2, i]})
    (ds / "dataset.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    low.write_json(ds / "config.json", {"audio": {"sample_rate": 22050, "quality": "medium"}, "num_symbols": 256})
    split = tmp_path / "split.json"
    low.write_json(split, {"train": ["a0.pt"], "val": ["a1.pt"], "test": ["a2.pt"]})
    for attr, path in (("SOURCE_DATASET_SHA", ds / "dataset.jsonl"), ("SOURCE_CONFIG_SHA", ds / "config.json"), ("SPLIT_SHA", split)):
        monkeypatch.setattr(low, attr, low.sha256(path))
    low.prepare(source, output, split)
    assert (output / low.DATASET_REL / "dataset.jsonl").read_bytes() == (ds / "dataset.jsonl").read_bytes()
    assert json.loads((output / low.DATASET_REL / "config.json").read_text())["audio"]["sample_rate"] == 16000
    wave = torch.load(output / "a0.pt", weights_only=True)
    assert wave.shape == (1, 16000)
    # 10 kHz would alias to 6 kHz without the low-pass filter.
    spectrum = torch.fft.rfft(wave).abs()[0]
    assert spectrum.argmax().item() == 440
    assert spectrum[6000] < spectrum[440] * 0.01
    assert low.verify(source, output, split)["verified_tensors"] == 6
    with pytest.raises(FileExistsError, match="refusing existing"):
        low.prepare(source, output, split)
    with (output / "s1.pt").open("ab") as f:
        f.write(b"corrupted")
    with pytest.raises(ValueError, match="derived tensor mismatch"):
        low.verify(source, output, split)


def test_low_preparation_rejects_split_leakage_and_unsafe_paths(tmp_path):
    low = script("prepare_piper_low")
    with pytest.raises(ValueError, match="no overlap"):
        low.validate_membership([{"audio_norm_path": "a", "audio_spec_path": "s"}],
                                {"train": ["a"], "val": ["a"], "test": []})
    with pytest.raises(ValueError, match="unsafe"):
        low.contained(tmp_path, "../source.pt")


def test_launcher_never_starts_without_confirmation(monkeypatch):
    launcher = script("run_piper_sizes")
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: pytest.fail("unexpected process launch"))
    assert launcher.main([]) == 0
    with pytest.raises(SystemExit):
        launcher.main(["--start"])


@pytest.mark.parametrize("exit_code", [0, 2])
def test_launcher_never_advances_to_high(monkeypatch, exit_code):
    launcher = script("run_piper_sizes")
    monkeypatch.setattr(launcher, "os", type("Platform", (), {"name": "posix"})())
    launched = []

    def fake_run(command, **kwargs):
        launched.append(command)
        return type("Exit", (), {"returncode": exit_code})()

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    assert launcher.main(["--start", "--confirm-training"]) == exit_code
    assert len(launched) == 1 and "piper-low-2026-10-30.json" in launched[0][-1]


def test_launcher_rejects_automatic_two_model_sequence(monkeypatch):
    launcher = script("run_piper_sizes")
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: pytest.fail("unexpected process launch"))
    with pytest.raises(SystemExit):
        launcher.main(["--run", "both", "--start", "--confirm-training"])
