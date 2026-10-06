"""Python wrapper around the compiled `core` Cython extension.

Note: upstream Piper/VITS's equivalent file does `from .monotonic_align.core
import maximum_path_c` -- a relative import that, taken literally, expects
a *nested* `monotonic_align` package inside this one, which does not exist
in either their tree or this one (verified by listing the directory). This
is a latent bug carried over from the original VITS repo; the correct
import (used here) is `from .core import maximum_path_c`.
"""
import numpy as np
import torch

from .core import maximum_path_c


def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """neg_cent: [b, t_t, t_s] negative cross-entropy, mask: [b, t_t, t_s]."""
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = neg_cent.data.cpu().numpy().astype(np.float32)
    path = np.zeros(neg_cent.shape, dtype=np.int32)

    t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
    t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)
    maximum_path_c(path, neg_cent, t_t_max, t_s_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)
