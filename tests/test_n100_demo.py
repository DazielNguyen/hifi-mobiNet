"""Offline contract tests: fake OpenVINO only; no model inference or PTQ."""
import hashlib
import importlib.util
import io
import json
import sys
import types
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest

from hifimobinet import frontend
from hifimobinet import n100_demo as demo

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/n100_demo/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_catalog_matches_final_evidence():
    actual = demo.read(demo.CATALOG)
    assert actual == script("make_catalog").build_catalog()
    assert len(actual["models"]) == 6
    assert sum(len(files) for m in actual["models"] for files in m["artifacts"].values()) == 24


def test_hash_guard_rejects_corruption_and_escape(tmp_path):
    model = tmp_path / "model.xml"
    model.write_bytes(b"new corrected model")
    entry = {"file": model.name, "sha256": demo.sha(model)}
    assert demo.verify_file(tmp_path, entry) == model
    model.write_bytes(b"old numeric-defect model")
    with pytest.raises(ValueError, match="changed"):
        demo.verify_file(tmp_path, entry)
    with pytest.raises(ValueError, match="escapes"):
        demo.verify_file(tmp_path, {"file": "../outside.bin", "sha256": "0" * 64})


def test_pc_cannot_be_inference_server(monkeypatch):
    monkeypatch.setattr(demo.platform, "system", lambda: "Windows")
    with pytest.raises(RuntimeError, match="must run on the Linux N100"):
        demo.host_info()


def test_changed_environment_is_rejected_before_model_loading(monkeypatch, tmp_path):
    monkeypatch.setattr(demo, "host_info", lambda: {"packages": {"openvino": "unexpected"}})
    with pytest.raises(ValueError, match="versions differ"):
        demo.preflight(tmp_path)


def test_frontend_mismatch_is_not_a_pass(monkeypatch):
    rows = [{"idx": i, "text": f"sentence {i}", "phoneme_ids": [1, 2]} for i in range(720)]
    monkeypatch.setattr(frontend, "encode", lambda text: [1, 2] if text != "sentence 719" else [1, 3])
    with pytest.raises(ValueError, match="719"):
        demo.check_frontend(rows)


def test_wav_and_duration():
    samples = np.sin(np.arange(2205) * .1).astype(np.float32)[None, None, :]
    wav, info = demo.to_wav(samples, 22050)
    with wave.open(io.BytesIO(wav)) as handle:
        assert (handle.getnchannels(), handle.getframerate(), handle.getnframes(), handle.getsampwidth()) == (1, 22050, 2205, 2)
    assert info["audio_seconds"] == .1


@pytest.mark.parametrize("output", [np.zeros((1, 1, 20)), np.full((1, 1, 20), np.nan),
                                     np.ones(20), np.ones((1, 1, 22050 * 60 + 1))])
def test_invalid_audio_not_published(output):
    with pytest.raises(ValueError):
        demo.to_wav(output, 22050)


def runtime_node(precision, name="/enc_p/conv/Conv"):
    return types.SimpleNamespace(get_friendly_name=lambda: name, get_rt_info=lambda: {
        "layerType": "Convolution", "runtimePrecision": precision, "originalLayersNames": name})


@pytest.mark.parametrize("precision,node_precision,name", [
    ("fp32", "bf16", "/enc_p/conv/Conv"), ("int8", "f32", "/enc_p/conv/Conv"),
    ("int8", "i8", "/dec/conv_post/Conv")])
def test_wrong_executed_precision_rejected(precision, node_precision, name):
    compiled = types.SimpleNamespace(get_runtime_model=lambda: types.SimpleNamespace(
        get_ops=lambda: [runtime_node(node_precision, name)]))
    with pytest.raises(ValueError):
        demo.precision_evidence(compiled, precision)


@pytest.fixture
def fake_engine(monkeypatch, tmp_path):
    catalog = demo.read(demo.CATALOG)
    for model in catalog["models"]:
        for files in model["artifacts"].values():
            for entry in files:
                path = tmp_path / entry["file"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fake IR: never executed")
                entry["sha256"] = demo.sha(path)
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    monkeypatch.setattr(demo, "CATALOG", catalog_path)
    monkeypatch.setattr(demo, "preflight", lambda *_: {"host": {
        "hostname": "fake-edge-n100", "pid": 123, "packages": {"openvino": "fake"}}})
    monkeypatch.setattr(frontend, "encode", lambda _: [1, 0, 5, 0, 2])

    class Request:
        def __init__(self):
            self.calls = 0
            self.inputs = {}
            self.fail = False

        def set_tensor(self, name, value):
            self.inputs[name] = value

        def infer(self, **kwargs):
            self.calls += 1
            if self.fail:
                raise RuntimeError("synthetic inference failure")

        def get_output_tensor(self, _):
            return types.SimpleNamespace(data=np.sin(np.arange(2205) * .1).astype(np.float32)[None, None, :])

    class Core:
        def __init__(self):
            self.compiles = 0

        def get_property(self, *_):
            return "Intel(R) N100 (fake backend)"

        def read_model(self, xml, weights):
            return xml

        def compile_model(self, model, device, config):
            assert device == "CPU" and config["INFERENCE_NUM_THREADS"] == 4
            self.compiles += 1
            request = Request()
            return types.SimpleNamespace(
                get_property=lambda _: ["CPU"],
                inputs=[types.SimpleNamespace(get_any_name=lambda n=name: n)
                        for name in ("input", "input_lengths", "scales")], outputs=[None],
                create_infer_request=lambda: request,
                get_runtime_model=lambda: types.SimpleNamespace(get_ops=lambda: [
                    runtime_node("i8" if "int8.xml" in model else "f32")]))

    monkeypatch.setitem(sys.modules, "openvino", types.SimpleNamespace(Core=Core, Tensor=lambda value: value))
    return demo.Engine(tmp_path, tmp_path / "runs")


def test_fresh_inference_reuse_switch_and_saved_record(fake_engine):
    engine = fake_engine
    model = "piper-seqir-no-vits2-cpn"
    wav, first = engine.synthesize("New text.", model, "fp32")
    _, second = engine.synthesize("Another sentence.", model, "fp32")
    assert engine.core.compiles == 1 and engine.request.calls == 2
    assert first["first_call_after_load"] and not second["first_call_after_load"]
    assert second["load_compile_s"] == 0
    assert first["request_id"] != second["request_id"]
    assert first["rtf"] == first["inference_s"] / first["audio_seconds"]
    assert first["wav_sha256"] == hashlib.sha256(wav).hexdigest()
    assert (engine.session / first["wav_file"]).read_bytes() == wav
    assert demo.read(engine.session / (first["request_id"] + ".json"))["status"] == "generated"
    _, third = engine.synthesize("New text.", model, "int8")
    assert engine.core.compiles == 2 and engine.request.calls == 1
    assert third["execution"]["precision"]["int8_weighted_count"] == 1


def test_failure_and_concurrent_request_do_not_return_old_audio(fake_engine):
    engine = fake_engine
    model = "piper-seqir-no-vits2-cpn"
    engine.synthesize("First.", model, "fp32")
    engine.lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="another request"):
            engine.synthesize("Busy.", model, "fp32")
    finally:
        engine.lock.release()
    engine.request.fail = True
    with pytest.raises(RuntimeError, match="synthetic"):
        engine.synthesize("Fail.", model, "fp32")
    assert not engine.lock.locked() and engine.active is None
    failures = list(engine.session.glob("*-failed.json"))
    assert len(failures) == 1
    assert demo.read(failures[0])["status"] == "failed"
    assert len(list(engine.session.glob("*.wav"))) == 1


def test_bundle_roundtrip_is_code_only(tmp_path):
    builder = script("build_bundle")
    report = builder.build(tmp_path)
    assert all(report["checks"].values())
    assert not report["model_weights_included"] and not report["n100_live_inference_tested"]
    archive = tmp_path / report["zip"]
    assert archive.with_suffix(".zip.sha256").read_text().strip() == f"{demo.sha(archive)}  {archive.name}"
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        assert all(not name.endswith((".ckpt", ".onnx", ".bin", ".wav", ".pyc")) for name in names)
        assert any(name.endswith("licenses/espeak-ng.GPL-3.0.txt") for name in names)
        for name in names:
            if name.endswith(".sh"):
                assert b"\r" not in handle.read(name), "shell scripts must use LF"


@pytest.fixture
def http_server(fake_engine):
    import threading
    from http.server import ThreadingHTTPServer
    from hifimobinet.n100_server import handler_for
    with ThreadingHTTPServer(("127.0.0.1", 0), handler_for(fake_engine)) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        yield "http://127.0.0.1:" + str(server.server_port)
        server.shutdown()
        worker.join(timeout=5)


def test_http_text_to_wav_and_provenance(http_server):
    import base64
    from urllib.request import Request, urlopen
    status = json.loads(urlopen(http_server + "/api/status").read())
    assert status["server"]["host"]["hostname"] == "fake-edge-n100"
    assert len(status["models"]) == 6
    text = "A new sentence from the audience."
    request = Request(http_server + "/api/synthesize",
                      data=json.dumps({"text": text, "model_id": "piper-seqir-no-vits2-cpn", "precision": "fp32"}).encode(),
                      headers={"Origin": http_server, "Content-Type": "application/json"})
    with urlopen(request) as response:
        assert response.headers["Cache-Control"] == "no-store"
        result = json.loads(response.read())
    assert result["record"]["text"] == text
    assert result["record"]["execution"]["devices"] == ["CPU"]
    wav = base64.b64decode(result["wav_base64"])
    assert hashlib.sha256(wav).hexdigest() == result["record"]["wav_sha256"]
    with wave.open(io.BytesIO(wav)) as audio:
        assert audio.getframerate() == 22050
    page = urlopen(http_server).read().decode()
    assert '<audio id="audio" controls' in page
    assert urlopen(http_server + "/app.js").status == 200
    assert urlopen(http_server + "/style.css").status == 200


@pytest.mark.parametrize("origin,payload,expected", [
    ("http://other-host", {"text": "hello", "model_id": "piper-seqir-no-vits2-cpn", "precision": "fp32"}, 403),
    (None, {"text": "hello", "model_id": "piper-seqir-no-vits2-cpn", "precision": "fp32"}, 403),
    ("same", {"text": [], "model_id": "piper-seqir-no-vits2-cpn", "precision": "fp32"}, 400),
    ("same", {"text": "hello", "model_id": "unknown", "precision": "fp32"}, 400),
])
def test_http_rejects_invalid_requests(http_server, origin, payload, expected):
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    headers = {"Content-Type": "application/json"}
    if origin:
        headers["Origin"] = http_server if origin == "same" else origin
    request = Request(http_server + "/api/synthesize", data=json.dumps(payload).encode(), headers=headers)
    with pytest.raises(HTTPError) as error:
        urlopen(request)
    assert error.value.code == expected
    assert "wav_base64" not in json.loads(error.value.read())


def test_http_no_file_access_or_stale_audio_on_failure(http_server, fake_engine, monkeypatch):
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    for path in ("/../models.json", "/api/../server.json", "/runs/server.json"):
        with pytest.raises(HTTPError) as error:
            urlopen(http_server + path)
        assert error.value.code == 404
    def fail(**_):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(fake_engine, "synthesize", fail)
    request = Request(http_server + "/api/synthesize",
                      data=json.dumps({"text": "hello", "model_id": "piper-seqir-no-vits2-cpn", "precision": "fp32"}).encode(),
                      headers={"Origin": http_server, "Content-Type": "application/json"})
    with pytest.raises(HTTPError) as error:
        urlopen(request)
    assert error.value.code == 503
    assert json.loads(error.value.read()) == {"error": "synthetic failure"}
