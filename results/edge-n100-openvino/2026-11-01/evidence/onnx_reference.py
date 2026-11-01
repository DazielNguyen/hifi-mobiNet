"""Load the unchanged project parity helpers shipped under reference/."""
import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parent / "reference/scripts/evaluation/onnx/common.py"
_spec = importlib.util.spec_from_file_location("edge_project_onnx_common", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
matched_noise_graph = _module.matched_noise_graph
strip_matched_noise = _module.strip_matched_noise
without_random_nodes = _module.without_random_nodes
ort_session = _module.ort_session
parity = _module.parity
check_calibration = _module.check_calibration
