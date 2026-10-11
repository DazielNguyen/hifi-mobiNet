"""The shipped Piper-ParIR / Piper-SeqIR configs equal the Config A file except the decoder block."""
from pathlib import Path

import pytest

pytest.importorskip("yaml")

from hifimobinet.training.config import load_config  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs/training"
SHIPPED = {"Piper_ParIR_no_VITS2_cpn": ("piper-parir-no-vits2-cpn.yaml", "mrf",
                                        {"mb_expansion": [1, 1, 1], "mb_mrf_kernel_sizes": [3, 5, 7]}),
           "Piper_SeqIR_no_VITS2_cpn": ("piper-seqir-no-vits2-cpn.yaml", "mb",
                                        {"mb_expansion": [1, 1, 1], "mb_kernel_size": 3, "mb_blocks_per_stage": 2})}


@pytest.mark.parametrize("model_id", sorted(SHIPPED))
def test_shipped_ir_config_is_config_a_plus_decoder_bundle(model_id):
    name, resblock, extra = SHIPPED[model_id]
    a = load_config(CONFIGS / "piper-no-vits2-cpn.yaml")
    ir = load_config(CONFIGS / name)
    assert ir.model_id == model_id
    for section in ("optim", "trainer", "data", "harness"):
        assert ir.sections[section] == a.sections[section], section
    expected_model = dict(a.sections["model"], resblock=resblock, use_mrd=True, **extra)
    assert dict(ir.sections["model"]) == expected_model
