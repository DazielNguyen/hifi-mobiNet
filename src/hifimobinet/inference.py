"""Identified FP32 ONNX inference, deliberately without pickle loading."""
from __future__ import annotations
import io
import threading
import wave
from pathlib import Path
import numpy as np

from .frontend import encode
from .registry import ModelUnavailable, model_path, model_record

_SYNTHESIS_SLOT = threading.BoundedSemaphore(1)


def pcm16_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Match Banhmi audio_float_to_int16; applies documented peak scaling."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("Expected a non-empty, finite mono waveform")
    # This scaling applies ONLY to newly synthesized audio, never historical WAVs.
    pcm = np.clip(audio * (32767.0 / max(0.01, np.max(np.abs(audio)))), -32767, 32767).astype("<i2")
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())
    return stream.getvalue()


class Voice:
    """One cached session per allowed model; serial calls for bounded demo usage."""
    def __init__(self, model_id: str, root: Path | None = None):
        import onnxruntime as ort
        self.record = model_record(model_id, root)
        path = model_path(self.record, root, verify=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        observed = {x.name: x.type for x in self.session.get_inputs()}
        expected = {"input": "tensor(int64)", "input_lengths": "tensor(int64)", "scales": "tensor(float)"}
        if observed != expected:
            raise ModelUnavailable("ONNX input contract differs from the verified Q05 contract")
        self.lock = threading.Lock()

    def synthesize(self, text: str) -> bytes:
        ids = encode(text)
        x = np.asarray([ids], dtype=np.int64)
        feeds = {"input": x, "input_lengths": np.asarray([len(ids)], dtype=np.int64),
                 "scales": np.asarray([0.667, 1.0, 0.8], dtype=np.float32)}
        if not _SYNTHESIS_SLOT.acquire(blocking=False):
            raise ModelUnavailable("A synthesis request is already running; retry when it finishes")
        try:
            with self.lock:
                output = self.session.run(None, feeds)[0]
        finally:
            _SYNTHESIS_SLOT.release()
        if output.ndim != 3 or output.shape[:2] != (1, 1):
            raise ModelUnavailable("Unexpected ONNX output shape")
        rate = self.record["sample_rate"]
        if output.shape[-1] > rate * 60:
            raise ModelUnavailable("Output exceeds the demo's 60-second limit")
        return pcm16_wav(output[0, 0], rate)
