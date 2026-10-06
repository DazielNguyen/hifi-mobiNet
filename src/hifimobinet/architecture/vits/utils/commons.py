"""Small tensor utilities shared across the VITS components.

A few functions present in upstream VITS/Piper (`intersperse`, the
standalone `kl_divergence`, `clip_grad_value_`, `rand_gumbel*`, and the
sinusoidal `*_timing_signal_1d` helpers) are dropped here: none of them are
called anywhere in the training pipeline (grep-verified against the whole
piper_train tree) -- `intersperse`'s job is already done in banhmi_phonemize
(BOS/PAD/EOS interleaving), the real KL term lives in losses.kl_loss, and
gradient clipping goes through PyTorch Lightning's `gradient_clip_val`
instead of a hand-rolled clipper.
"""
from typing import Optional

import torch
from torch.nn import functional as F


def init_weights(module: torch.nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if module.__class__.__name__.find("Conv") != -1:
        module.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def subsequent_mask(length: int) -> torch.Tensor:
    return torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor, input_b: torch.Tensor, n_channels: torch.Tensor
) -> torch.Tensor:
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    return t_act * s_act


def slice_segments(x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4) -> torch.Tensor:
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = max(0, int(ids_str[i]))
        ret[i] = x[i, :, idx_str : idx_str + segment_size]
    return ret


def rand_slice_segments(x: torch.Tensor, x_lengths=None, segment_size: int = 4):
    b, _d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b], device=x.device) * ids_str_max).to(dtype=torch.long)
    return slice_segments(x, ids_str, segment_size), ids_str


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """duration: [b, 1, t_x], mask: [b, 1, t_y, t_x] -> hard alignment path."""
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).type_as(mask)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, (0, 0, 1, 0, 0, 0))[:, :-1]
    return path.unsqueeze(1).transpose(2, 3) * mask
