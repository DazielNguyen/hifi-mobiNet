"""Piecewise rational-quadratic spline flow (Durkan et al. 2019, "Neural
Spline Flows"), as used by VITS's ConvFlow for the normalizing-flow prior.

Ported with the same numerics as upstream (this is dense, easy-to-get-subtly
-wrong math), reorganized for clarity: dead commented-out alternate lines
removed, variable names kept close to the paper (widths/heights/derivatives
of the spline bins) since that's what makes the formulas checkable against
the paper's equations.
"""
import numpy as np
import torch
from torch.nn import functional as F

_MIN_BIN_WIDTH = 1e-3
_MIN_BIN_HEIGHT = 1e-3
_MIN_DERIVATIVE = 1e-3
# Shared epsilon for every division/log guard in _rational_quadratic_spline.
# widths/heights/derivatives are all kept strictly positive by construction
# (softmax+MIN_BIN_*, MIN_DERIVATIVE+softplus), so these quantities are only
# ever supposed to graze zero through bf16/fp32 rounding right at the edge of
# a spline bin -- not a sign that the math itself is wrong. Ported from
# upstream Piper/VITS (same gap exists there too, see banhmi/EdgeTTS
# root-cause comparison); fixing it here since it directly caused
# StochasticDurationPredictor's reverse-sampling to collapse to NaN during
# real training.
_EPS = 1e-6


def piecewise_rational_quadratic_transform(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails=None,
    tail_bound=1.0,
):
    if tails is None:
        return _rational_quadratic_spline(
            inputs,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=inverse,
        )

    return _unconstrained_rational_quadratic_spline(
        inputs,
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
        inverse=inverse,
        tail_bound=tail_bound,
    )


def _searchsorted(bin_locations: torch.Tensor, inputs: torch.Tensor, eps: float = 1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def _unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tail_bound=1.0,
):
    """Outside [-tail_bound, tail_bound], the transform is the identity
    (linear tails); the spline only warps the inside interval."""
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    constant = np.log(np.exp(1 - _MIN_DERIVATIVE) - 1)
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0

    # Community-known VITS failure mode (see e.g. coqui-ai/TTS#1959): if
    # every element of this call happens to land outside [-tail_bound,
    # tail_bound] -- plausible early in training when dp's flow output
    # isn't regularized yet -- inside_interval_mask is all-False, and
    # indexing/calling _rational_quadratic_spline on the resulting
    # zero-sized selection is undefined behavior upstream (reported as a
    # hard crash there; empty-tensor ops here can instead surface as NaN
    # under bf16). outputs/logabsdet are already correctly populated for
    # every element via the outside-mask branch above, so skipping the
    # inside-mask call entirely when there's nothing inside is a safe,
    # exact no-op -- not a fallback that changes any in-bounds result.
    if inside_interval_mask.any():
        outputs[inside_interval_mask], logabsdet[inside_interval_mask] = _rational_quadratic_spline(
            inputs[inside_interval_mask],
            unnormalized_widths[inside_interval_mask, :],
            unnormalized_heights[inside_interval_mask, :],
            unnormalized_derivatives[inside_interval_mask, :],
            inverse=inverse,
            left=-tail_bound,
            right=tail_bound,
            bottom=-tail_bound,
            top=tail_bound,
        )

    return outputs, logabsdet


def _rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
):
    num_bins = unnormalized_widths.shape[-1]

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = _MIN_BIN_WIDTH + (1 - _MIN_BIN_WIDTH * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = _MIN_DERIVATIVE + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = _MIN_BIN_HEIGHT + (1 - _MIN_BIN_HEIGHT * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    bin_idx = _searchsorted(cumheights if inverse else cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    # widths carries the same bf16 cancellation risk as input_bin_widths
    # above (both come from the same cumsum-then-difference) -- guard here
    # too, before gather, since a NaN/inf entry survives indexing.
    delta = heights / widths.clamp_min(_EPS)
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        # clamp before sqrt: bfloat16 rounding can produce small negative
        # values here even though the true discriminant is non-negative.
        discriminant = discriminant.clamp(min=0)

        root_denominator = -b - torch.sqrt(discriminant)
        # This is analytically bounded away from 0, but bf16/fp32 rounding
        # can still land it exactly on (or past) 0, turning the division
        # below into inf/NaN. Push away from 0 in whichever direction it's
        # already leaning (or negative by convention exactly at 0, matching
        # this formula's expectation that the denominator stays negative)
        # while guaranteeing a nonzero magnitude. torch.where+comparison
        # instead of torch.copysign: the latter has no ONNX opset-15 op.
        sign = torch.where(root_denominator >= 0, 1.0, -1.0)
        root_denominator = torch.where(
            root_denominator.abs() < _EPS,
            sign * _EPS,
            root_denominator,
        )
        root = (2 * c) / root_denominator
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator.clamp_min(_EPS)) - 2 * torch.log(
            denominator.clamp_min(_EPS)
        )
        return outputs, -logabsdet

    # input_bin_widths is analytically >= _MIN_BIN_WIDTH (softmax + floor,
    # see the construction above), but it's the result of a cumsum-then-
    # difference (line ~111) -- under bf16's ~7-bit mantissa, that
    # subtraction can catastrophically cancel to exactly 0.0 for a rare
    # bin/step even though the true value is a small positive number,
    # turning this division into inf/NaN. Same failure family as every
    # other division/log in this function (see _EPS comment above); this
    # one was the gap that let StochasticDurationPredictor's forward (NLL)
    # pass collapse under bf16 training, poisoning dp's Adam state for the
    # rest of the run (dec/enc_p unaffected since they don't depend on dp's
    # output).
    theta = (inputs - input_cumwidths) / input_bin_widths.clamp_min(_EPS)
    theta_one_minus_theta = theta * (1 - theta)

    numerator = input_heights * (
        input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
    )
    denominator = input_delta + (
        (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
        * theta_one_minus_theta
    )
    outputs = input_cumheights + numerator / denominator.clamp_min(_EPS)

    derivative_numerator = input_delta.pow(2) * (
        input_derivatives_plus_one * theta.pow(2)
        + 2 * input_delta * theta_one_minus_theta
        + input_derivatives * (1 - theta).pow(2)
    )
    logabsdet = torch.log(derivative_numerator.clamp_min(_EPS)) - 2 * torch.log(
        denominator.clamp_min(_EPS)
    )

    return outputs, logabsdet
