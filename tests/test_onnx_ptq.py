"""Tests of scripts/evaluation/onnx/common.py (protocol docs/evaluation-onnx-ptq.md).

Written with unittest so that they run under pytest (training venv: torch) and with
``python -m unittest`` in the scorer environment (onnx, onnxruntime, no pytest). Tests whose
packages are missing are skipped.
"""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/evaluation/onnx"))
import common as C  # noqa: E402

HAS = {m: importlib.util.find_spec(m) is not None for m in ("numpy", "torch", "onnx", "onnxruntime")}


def _edgetts_models_importable() -> bool:
    """EdgeTTS models import the compiled MAS extension. A file may exist yet not load (a Linux
    build seen from Windows through the shared folder), so try the import itself."""
    if not HAS["torch"]:
        return False
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from hifimobinet.training.vendor import vendor_module

        vendor_module("edgetts_vits", "models")
        return True
    except ImportError:
        return False


EDGETTS_MAS_BUILT = _edgetts_models_importable()


class PureHelpers(unittest.TestCase):
    def test_module_of(self):
        self.assertEqual(C.module_of("/flow/flows.0/enc/in_layers.0/Conv"), "flow")
        self.assertEqual(C.module_of("/dp/RandomNormalLike"), "dp")
        self.assertEqual(C.module_of("/MatMul"), "")
        self.assertEqual(C.module_of("Slice_12"), "")

    def test_ptq_scope_is_conv_in_enc_p_dp_flow_only(self):
        nodes = [("/enc_p/encoder/Conv", "Conv"), ("/dp/convs/Conv", "Conv"), ("/flow/flows.0/pre/Conv", "Conv"),
                 ("/dec/conv_pre/Conv", "Conv"), ("/dec/ups.0/ConvTranspose", "ConvTranspose"),
                 ("/MatMul", "MatMul"), ("/enc_p/encoder/attn/MatMul", "MatMul"), ("/Conv_9", "Conv")]
        quantize, exclude = C.ptq_nodes(nodes)
        self.assertEqual(quantize, ["/dp/convs/Conv", "/enc_p/encoder/Conv", "/flow/flows.0/pre/Conv"])
        self.assertEqual(exclude, ["/Conv_9", "/dec/conv_pre/Conv"])

    def test_calibration_check(self):
        split = {"train": ["a", "b"], "val": ["c"], "test": ["d"]}
        dataset = [{"audio_norm_path": k, "phoneme_ids": [1, 2]} for k in "abcd"]
        good = [{"id": "A", "audio_norm_path": "a", "phoneme_ids": [1, 2]},
                {"id": "B", "audio_norm_path": "b", "phoneme_ids": [1, 2]}]
        self.assertTrue(C.check_calibration(good, split, dataset)["passed"])
        bad = good + [{"id": "D", "audio_norm_path": "d", "phoneme_ids": [1, 2]},
                      {"id": "B2", "audio_norm_path": "b", "phoneme_ids": [9]}]
        r = C.check_calibration(bad, split, dataset)
        self.assertFalse(r["passed"])
        self.assertEqual((r["train"], r["test"]), (3, 1))
        self.assertTrue(any("split membership ['test']" in x for x in r["problems"]))
        self.assertTrue(any("phoneme IDs differ" in x for x in r["problems"]))

    def test_the_designed_calibration_file_has_60_unique_rows(self):
        rows = [json.loads(x) for x in C.CALIBRATION.read_text(encoding="utf-8").splitlines() if x.strip()]
        self.assertEqual(len(rows), 60)
        self.assertEqual(len({r["id"] for r in rows}), 60)


@unittest.skipUnless(HAS["numpy"], "numpy missing")
class NumpyHelpers(unittest.TestCase):
    def test_acoustic_buffer_keeps_the_pytorch_noise_as_prefix(self):
        import numpy as np

        noise = np.random.default_rng(0).standard_normal((1, 4, 10)).astype(np.float32)
        buf = C.acoustic_buffer(noise, 7)
        self.assertEqual(buf.shape, (1, 4, 10 + 10 + 200))  # the PyTorch frames, then as many again plus 200
        np.testing.assert_array_equal(buf[:, :, :10], noise)
        np.testing.assert_array_equal(buf, C.acoustic_buffer(noise, 7))
        self.assertFalse(np.array_equal(buf, C.acoustic_buffer(noise, 8)))

    def test_parity_thresholds_and_shape(self):
        import numpy as np

        a = np.ones((1, 1, 100), np.float32)
        self.assertTrue(C.parity(a, a + 1e-4)["passed"])
        self.assertFalse(C.parity(a, a + 1e-2)["passed"])
        self.assertFalse(C.parity(a, a[:, :, :99])["passed"])
        self.assertFalse(C.parity(a, a * np.nan)["passed"])


@unittest.skipUnless(HAS["torch"], "torch missing")
class TorchHelpers(unittest.TestCase):
    def test_nan_to_num_matches_torch(self):
        import torch

        x = torch.tensor([float("nan"), float("inf"), -float("inf"), 1.5, -2.0])
        for args in ((0.0, 6.0, -6.0), (1.0, 403.0, 0.0)):
            self.assertTrue(torch.equal(C.nan_to_num(x, *args),
                                        torch.nan_to_num(x, nan=args[0], posinf=args[1], neginf=args[2])))

    @unittest.skipUnless(EDGETTS_MAS_BUILT, "EdgeTTS MAS extension not built")
    def test_export_wrapper_equals_edgetts_infer(self):
        import torch

        sys.path.insert(0, str(ROOT / "src"))
        from hifimobinet.training.vendor import vendor_module

        models = vendor_module("edgetts_vits", "models")
        torch.manual_seed(0)
        g = models.SynthesizerTrn(
            n_vocab=12, spec_channels=33, segment_size=8, inter_channels=8, hidden_channels=8, filter_channels=16,
            n_heads=2, n_layers=1, kernel_size=3, p_dropout=0.0, resblock="2", resblock_kernel_sizes=(3,),
            resblock_dilation_sizes=((1, 2),), upsample_rates=(4, 4), upsample_initial_channel=16,
            upsample_kernel_sizes=(8, 8), n_speakers=1, gin_channels=0, use_sdp=True, use_snake=False, use_f0=False,
            use_transformer_flows=False).eval()
        x = torch.LongTensor([[1, 3, 5, 7, 2, 4, 6]])
        xl = torch.LongTensor([7])
        refs = []
        for seed in (1, 2):
            torch.manual_seed(seed)
            with torch.no_grad():
                refs.append(g.infer(x, xl, noise_scale=0.667, length_scale=1.0, noise_scale_w=0.8)[0])
        wrapper, cast = C.export_wrapper(g)
        self.assertGreater(cast, 0)
        for seed, ref in zip((1, 2), refs):
            torch.manual_seed(seed)
            with torch.no_grad():
                out = wrapper(x, xl, torch.tensor([0.667, 1.0, 0.8]))
            self.assertTrue(torch.equal(ref, out))
        names = {n for n, _ in wrapper.named_modules() if n.count(".") == 0 and n}
        self.assertEqual(names, {"enc_p", "dp", "flow", "dec"})


@unittest.skipUnless(HAS["onnx"] and HAS["numpy"], "onnx missing")
class GraphSurgery(unittest.TestCase):
    def tiny(self):
        from onnx import TensorProto, helper

        nodes = [helper.make_node("RandomNormalLike", ["X"], ["dur_noise"], name="/dp/RandomNormalLike", dtype=1),
                 helper.make_node("Add", ["X", "dur_noise"], ["A"], name="/dp/Add"),
                 helper.make_node("RandomNormalLike", ["Y"], ["ac_noise"], name="/RandomNormalLike", dtype=1),
                 helper.make_node("Add", ["Y", "ac_noise"], ["B"], name="/Add")]
        graph = helper.make_graph(nodes, "tiny",
                                  [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, "n"]),
                                   helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, "f"])],
                                  [helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 2, "n"]),
                                   helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 3, "f"])])
        # IR version 10, which onnxruntime 1.23 accepts (onnx 1.22 would default to 13)
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)

    def test_replacement_strips_back_and_checks(self):
        import onnx

        model = self.tiny()
        matched, info = C.matched_noise_graph(model, 3)
        onnx.checker.check_model(matched, full_check=True)
        self.assertEqual(info["duration_input"], "dur_noise")
        self.assertFalse(any(n.op_type.startswith("Random") for n in matched.graph.node))
        self.assertEqual(C.strip_matched_noise(matched).SerializeToString(),
                         C.without_random_nodes(model).SerializeToString())

    def test_random_nodes_requires_two(self):
        from onnx import helper

        model = self.tiny()
        del model.graph.node[0]
        with self.assertRaises(SystemExit):
            C.random_nodes(model.graph)
        model = self.tiny()
        model.graph.node[0].name = "/RandomNormalLike_dp"
        with self.assertRaises(SystemExit):
            C.random_nodes(model.graph)
        self.assertIsNotNone(helper)

    @unittest.skipUnless(HAS["onnxruntime"], "onnxruntime missing")
    def test_acoustic_noise_is_sliced_to_the_like_tensor(self):
        import numpy as np
        import onnxruntime as ort

        matched, _ = C.matched_noise_graph(self.tiny(), 3)
        sess = ort.InferenceSession(matched.SerializeToString(), providers=["CPUExecutionProvider"])
        x = np.zeros((1, 2, 4), np.float32)
        y = np.zeros((1, 3, 5), np.float32)
        dur = np.arange(8, dtype=np.float32).reshape(1, 2, 4)
        buf = np.arange(3 * 12, dtype=np.float32).reshape(1, 3, 12)
        a, b = sess.run(None, {"X": x, "Y": y, "dur_noise": dur, C.NOISE_INPUT: buf})
        np.testing.assert_array_equal(a, dur)
        np.testing.assert_array_equal(b, buf[:, :, :5])


if __name__ == "__main__":
    unittest.main()
