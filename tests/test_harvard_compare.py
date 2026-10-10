import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("harvard_compare", ROOT / "scripts/evaluation/harvard/compare.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def rows(rtf, utmos, wer):
    return [{"idx": i, "text": f"sentence {i}", "asr_text": f"sentence {i}", "synth_time_s": r, "audio_duration_s": 1.0,
             "rtf": r, "utmosv2": u, "wer": w} for i, (r, u, w) in enumerate(zip(rtf, utmos, wer))]


def write_refs(tmp_path, ref):
    d = tmp_path / "refs"
    d.mkdir()
    for m in compare.REFERENCES:
        (d / f"harvard_{m}_results.json").write_text(json.dumps(ref))
    return d


def test_direction_better_counts_and_reproducible_interval(tmp_path):
    ref = rows([0.04] * 6, [3.0] * 6, [0.2] * 6)
    cand = rows([0.03] * 6, [3.5] * 6, [0.0, 0.1, 0.2, 0.2, 0.3, 0.2])
    (tmp_path / "cand.json").write_text(json.dumps(cand))
    refs = write_refs(tmp_path, ref)
    for name in ("a.json", "b.json"):
        compare.main(["--candidate", str(tmp_path / "cand.json"), "--label", "cand", "--reference-dir", str(refs),
                      "--output", str(tmp_path / name)])
    a = json.loads((tmp_path / "a.json").read_text())
    b = json.loads((tmp_path / "b.json").read_text())
    t = a["paired_tests"]
    assert t["cand_vs_baseline_rtf"]["mean_diff_x_minus_y"] == pytest.approx(-0.01)
    assert t["cand_vs_baseline_rtf"]["candidate_better_count"] == 6
    assert t["cand_vs_baseline_utmosv2"]["candidate_better_count"] == 6
    assert (t["cand_vs_baseline_wer"]["candidate_better_count"], t["cand_vs_baseline_wer"]["candidate_worse_count"]) == (2, 1)
    assert t["cand_vs_baseline_wer"]["paired_bootstrap95_mean_diff"] == b["paired_tests"]["cand_vs_baseline_wer"]["paired_bootstrap95_mean_diff"]
    assert all(v["exploratory_holm_adjusted_wilcoxon_p"] >= v["wilcoxon_p"] for v in t.values())


def test_refuses_unpaired_order(tmp_path):
    ref = rows([0.04] * 4, [3.0] * 4, [0.2] * 4)
    cand = list(reversed(rows([0.03] * 4, [3.5] * 4, [0.1] * 4)))
    (tmp_path / "cand.json").write_text(json.dumps(cand))
    refs = write_refs(tmp_path, ref)
    with pytest.raises(ValueError):
        compare.main(["--candidate", str(tmp_path / "cand.json"), "--label", "cand", "--reference-dir", str(refs),
                      "--output", str(tmp_path / "out.json")])
