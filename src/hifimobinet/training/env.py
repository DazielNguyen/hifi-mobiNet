"""Print the actual training environment as JSON (versions observed at runtime).

    python -m hifimobinet.training.env
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional

import torch

from .vendor import repository_root


def _run(*cmd: str) -> Optional[str]:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_record(include_host: bool = False) -> Dict[str, Any]:
    import numpy
    import pytorch_lightning as pl
    root = str(repository_root())
    record: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "pytorch_lightning": pl.__version__,
        "numpy": numpy.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "torch_num_threads": torch.get_num_threads(),
        "repository_commit": _run("git", "-C", root, "rev-parse", "HEAD"),
        "repository_dirty": bool(_run("git", "-C", root, "status", "--porcelain")),
    }
    for name in ("librosa", "torchmetrics", "Cython", "scipy", "soundfile"):
        try:
            record[name.lower()] = __import__(name).__version__
        except ImportError:
            record[name.lower()] = None
    if include_host:
        record["nvidia_smi"] = _run("nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
                                    "--format=csv,noheader")
        record["cpu_count"] = os.cpu_count()
        try:
            with open("/proc/meminfo", encoding="utf-8") as handle:
                record["mem_total_kb"] = int(handle.readline().split()[1])
        except OSError:
            record["mem_total_kb"] = None
        usage = shutil.disk_usage(os.environ.get("HIFI_TRAIN_ROOT", root))
        record["disk_free_gb"] = round(usage.free / 1e9, 1)
    return record


def main() -> None:
    print(json.dumps(environment_record(include_host=True), indent=2))


if __name__ == "__main__":
    main()
