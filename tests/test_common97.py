"""Regression checks for the duration FQ leak and common97 activation gates."""
from types import SimpleNamespace as NS
import warnings

import numpy as np
import pytest

from hifimobinet import common97 as c
from hifimobinet.ptq_numeric_guard import Recorder, NumericDiagnosticStop, numpy_methods

def node(name, typ):
    return NS(get_friendly_name=lambda: name, get_type_name=lambda: typ)


def test_scope_exact_and_excludes_unweighted_duration():
    for mid in c.MODEL_IDS:
        names = c.policy()["models"][mid]["allowed_node_names"]
        assert len(names) == len(set(names)) == 97
        ops = [node(n, "Convolution") for n in names] + [node(n, t) for n, t in (
            ("/CumSum", "CumSum"), ("/Exp", "Exp"), ("/Sub", "Subtract"),
            ("/dp/depthwise", "GroupConvolution"), ("/dec/conv", "Convolution"),
            ("/alignment", "MatMul"), ("scales", "Parameter"), ("mask", "Convert"))]
        result = c.scope(NS(get_ops=lambda: ops), mid)
        assert result["breakdown"] == c.EXPECTED
        assert set(result["ignored_nodes"]) == {"/CumSum", "/Exp", "/Sub", "/dp/depthwise", "/dec/conv", "/alignment", "mask"}
        with pytest.raises(ValueError, match="previously quantized"):
            c.scope(NS(get_ops=lambda: ops + [node("renamed", "FakeQuantize")]), mid)
        with pytest.raises(ValueError, match="scope drift"):
            c.scope(NS(get_ops=lambda: ops[1:]), mid)


def fq_model(target_name, target_type="Convolution", port=0):
    target = node(target_name, target_type)
    fq = node("harmless-renamed-fq", "FakeQuantize")
    fq.output = lambda _: NS(get_target_inputs=lambda: [NS(get_node=lambda: target, get_index=lambda: port)])
    def value(i):
        src = node("/CumSum" if i == 0 else "range", "CumSum" if i == 0 else "Constant")
        src.get_data = lambda: np.array(0 if i in (1, 3) else 1, dtype=np.float32)
        return NS(get_node=lambda: src)
    fq.input_value = value
    fq.get_levels = lambda: 256
    return NS(get_ops=lambda: [fq])


def test_duration_quantizer_cannot_hide_behind_97_kernel_count_or_new_name():
    allowed = {"/enc_p/conv"}
    for target, typ, port in [("/Reshape", "Reshape", 0), ("/dec/conv", "Convolution", 0),
                               ("/enc_p/conv", "Convolution", 1)]:
        with pytest.raises(ValueError, match="outside selected"):
            c.audit_fq_edges(fq_model(target, typ, port), allowed, expected_count=1)
    assert c.audit_fq_edges(fq_model("/enc_p/conv"), allowed, expected_count=1)["passed"]


def test_runtime_exact_nodes_not_only_count():
    mid = c.MODEL_IDS[0]
    names = c.policy()["models"][mid]["allowed_node_names"]
    kernels = [{"name": n, "layerType": "Convolution", "runtimePrecision": "i8", "originalLayersNames": n} for n in names]
    assert c.audit_runtime(kernels, mid)["int8_weighted_kernel_count"] == 97
    with pytest.raises(ValueError, match="drift"):
        c.audit_runtime(kernels[:-1], mid)
    with pytest.raises(ValueError, match="outside common97"):
        c.audit_runtime(kernels[:-1] + [{**kernels[0], "name": "/dec/conv", "originalLayersNames": "/dec/conv"}], mid)


def test_source_hash_and_unvalidated_model_rejected():
    with pytest.raises(ValueError, match="verified Piper"):
        c.check_source("parallel-ir", "0" * 64)
    with pytest.raises(ValueError, match="verified Piper"):
        c.check_source(c.MODEL_IDS[0], "0" * 64)


def test_numeric_guard_preserves_hooks_even_when_library_swallows_error(tmp_path):
    methods = numpy_methods()
    original = methods._mean
    warning_hook = warnings.showwarning
    err = np.geterr().copy()
    recorder = Recorder(tmp_path, np)
    def swallowed():
        with warnings.catch_warnings(), np.errstate(all="warn"):
            warnings.simplefilter("always")
            try:
                return np.mean(np.full(4, np.finfo(np.float32).max, dtype=np.float32))
            except RuntimeError:
                return 0
    with pytest.raises(NumericDiagnosticStop):
        recorder.guarded_call(swallowed)
    assert (tmp_path / "numeric-event.json").is_file()
    assert methods._mean is original and warnings.showwarning is warning_hook and np.geterr() == err
