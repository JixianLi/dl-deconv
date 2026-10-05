"""Each loss estimates the statistic core/losses.py claims for it, checked by fitting one
constant to a synthetic sample of pixel fluxes; plus finite gradients and exact
to_normalized_ideal round trips. Run as `uv run python -m test.test_losses`.

The sample is zero-inflated like the ideal image: 70% exact zeros, 30% Pareto(1.5) fluxes
starting at 0.4 median sigma. Its mean is far above its median (zero), so the losses
separate:
  flux_mse, flux_relative_mse, tweedie  -> the sample mean
  l1                                    -> the median (0)
  mse                                   -> below the mean (Jensen: asinh is concave)
beta_nll's flux mean is exact only when asinh(flux) is Gaussian, so it is checked on a
Gaussian sample in stretched space, and only reported on the zero-inflated one.
"""

import numpy as np
import torch

from core import losses
from core.normalize import AsinhNorm

SEED = 0
SAMPLE_SIZE = 20_000
ZERO_FRACTION = 0.7
PARETO_SHAPE = 1.5
MEDIAN_FLUX_SIGMA = 0.013
IDEAL_NORM = AsinhNorm(median=0.0, beta=0.005, lo_s=0.0, hi_s=13.2)  # ~ M31 H100 norm.json
GAUSSIAN_STRETCHED_MEAN, GAUSSIAN_STRETCHED_STD = 2.0, 0.5
FIT_STEPS = 4000
FIT_LEARNING_RATE = 0.02
MEAN_TOLERANCE = 0.01  # relative
NAMES = ("l1", "mse", "huber", "flux_mse", "flux_relative_mse", "blurred_mse",
         "blurred_mse_sparse", "beta_nll", "tweedie")
MEAN_ESTIMATORS = ("flux_mse", "flux_relative_mse", "tweedie")


def zero_inflated_flux(rng):
    flux = MEDIAN_FLUX_SIGMA * 0.4 * (1.0 + rng.pareto(PARETO_SHAPE, SAMPLE_SIZE))
    flux[rng.random(SAMPLE_SIZE) < ZERO_FRACTION] = 0.0
    return flux


def gaussian_stretched_flux(rng):
    stretched = rng.normal(GAUSSIAN_STRETCHED_MEAN, GAUSSIAN_STRETCHED_STD, SAMPLE_SIZE)
    return IDEAL_NORM.median + IDEAL_NORM.beta * np.sinh(stretched)


def initial_raw_output(name):
    """A start well away from every loss's answer, so convergence is actually exercised."""
    if name == "tweedie":
        return [0.0]  # flux 1, ~100x the sample mean
    if name == "beta_nll":
        return [0.3, 0.0]
    return [0.3]


def fit_constant(name, flux):
    """Flux of the constant output that gradient descent on the loss converges to."""
    loss = build(name)
    target = torch.from_numpy(IDEAL_NORM.forward(flux)).reshape(1, 1, -1, 1)
    parameter = torch.tensor(initial_raw_output(name), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([parameter], lr=FIT_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, FIT_STEPS)
    for _ in range(FIT_STEPS):
        optimizer.zero_grad()
        loss(parameter.reshape(1, -1, 1, 1).expand(1, -1, target.shape[2], 1), target).backward()
        optimizer.step()
        scheduler.step()
    with torch.no_grad():
        normalized = loss.to_normalized_ideal(parameter.reshape(1, -1, 1, 1))
    return float(IDEAL_NORM.inverse(normalized.item()))


def build(name):
    return losses.build_loss(name, IDEAL_NORM, MEDIAN_FLUX_SIGMA)


def check_statistics(flux):
    mean, median = flux.mean(), np.median(flux)
    print(f"zero-inflated sample: mean flux {mean:.4g}, median {median:.4g}")
    for name in NAMES:
        estimate = fit_constant(name, flux)
        print(f"  {name:<19} estimate / mean = {estimate / mean:.4f}")
        if name in MEAN_ESTIMATORS:
            assert abs(estimate / mean - 1) < MEAN_TOLERANCE, f"{name}: does not estimate the mean"
        if name == "l1":
            assert abs(estimate - median) < 0.01 * MEDIAN_FLUX_SIGMA, f"l1: does not estimate the median"
        if name == "mse":
            assert estimate < mean, "mse: expected the Jensen shortfall below the mean"


def check_beta_nll_on_gaussian(flux):
    estimate = fit_constant("beta_nll", flux)
    print(f"Gaussian-in-asinh sample: beta_nll estimate / mean = {estimate / flux.mean():.4f}")
    assert abs(estimate / flux.mean() - 1) < MEAN_TOLERANCE, "beta_nll: flux-mean correction is off"


def check_gradients(rng):
    target = torch.from_numpy(IDEAL_NORM.forward(zero_inflated_flux(rng)[:4096])).reshape(1, 1, 64, 64)
    for name in NAMES:
        loss = build(name)
        for scale in (0.1, 5.0):  # typical outputs, and far outside the target range
            raw = (scale * torch.randn(1, loss.num_output_channels, 64, 64, dtype=torch.float64,
                                       generator=torch.Generator().manual_seed(SEED))).requires_grad_()
            loss(raw, target).backward()
            assert torch.isfinite(raw.grad).all(), f"{name}: non-finite gradient at scale {scale}"
    print("all gradients finite")


def raw_output_for(name, normalized):
    """The raw output whose to_normalized_ideal is `normalized`."""
    if name == "tweedie":
        return torch.log(losses.flux_from_normalized(IDEAL_NORM, normalized))
    if name == "beta_nll":
        return torch.cat([normalized, torch.full_like(normalized, -50.0)], dim=1)  # σ at its floor
    return normalized


def check_round_trips():
    normalized = torch.linspace(0.01, 1.0, 200, dtype=torch.float64).reshape(1, 1, -1, 1)
    for name in NAMES:
        loss = build(name)
        recovered = loss.to_normalized_ideal(raw_output_for(name, normalized))
        assert recovered.shape == normalized.shape, f"{name}: wrong output shape {recovered.shape}"
        error = float((recovered - normalized).abs().max())
        assert error < 1e-4, f"{name}: to_normalized_ideal round trip off by {error:.3g}"
    print("all to_normalized_ideal round trips exact")


def main():
    rng = np.random.default_rng(SEED)
    check_gradients(rng)
    check_round_trips()
    check_statistics(zero_inflated_flux(rng))
    check_beta_nll_on_gaussian(gaussian_stretched_flux(rng))
    print("all loss checks pass")


if __name__ == "__main__":
    main()
