"""Training losses, chosen by name (train.loss), each with its map back to the ideal image.

A loss owns the model's output head: `num_output_channels` is what the model must emit,
`loss(raw_output, target)` scores a batch against the asinh-normalized ideal target
(B, 1, H, W), and `to_normalized_ideal(raw_output)` turns the raw output into a 1-channel
asinh-normalized ideal, so eval, export and photometry stay loss-agnostic.

What each loss estimates per pixel, and its fixed hyperparameter (none are swept; see
docs/loss_survey.md). F = flux, y = asinh-normalized flux, σ = median Cramér–Rao flux
sigma of the dataset (noise.json), ŷ = model output.

  l1                 |ŷ − y|                          median
  mse                (ŷ − y)²                          mean of y (biased low after the inverse)
  huber              Huber(ŷ − y; δ)                   mean for |error| < δ, median above.
                     δ = y(σ) − y(0): an error of one σ in flux at zero flux, so
                     sources below S/N ~1 are fit in the L2 (mean) regime. In asinh
                     space bright-core errors are small too, so the L1 regime is
                     mostly missed or invented sources above S/N ~1.
  flux_mse           ((F̂ − F) / σ)²                    mean flux. χ² with a uniform weight: the
                     per-pixel σ map depends on the target and is ~uniform anyway.
  flux_relative_mse  (F̂ − F)² / (stopgrad(F̂)₊ + σ)²     mean flux (Noise2Noise, Lehtinen 2018):
                     the stop-gradient denominator only reweights pixels, so the
                     fixed point is still the mean, with bright pixels down-weighted.
  blurred_mse        (g∗ŷ − g∗y)², g Gaussian σ_g = 1 px (Deep-STORM, Nehme 2018)
                     mean of the blurred map: a 1 px shift costs little.
  blurred_mse_sparse blurred_mse + λ|ŷ|               as above, spurious flux suppressed.
  beta_nll           stopgrad(s^2β)·(½ log s² + (ŷ − y)²/(2s²)), β = 0.5 (Seitzer 2022)
                     2 channels: mean μ and log variance log s² of y. With S the
                     stretched value (y·(hi_s − lo_s) + lo_s) ~ N(m, v), the flux mean is
                     E[median + b·sinh S] = median + b·sinh(m)·exp(v/2). This is exact only
                     when S is Gaussian; on a zero-inflated pixel it is not.
  tweedie            −F·μ^(1−p)/(1−p) + μ^(2−p)/(2−p), p = 1.5, μ = exp(ŷ) in flux units
                     mean flux: d/dμ = μ^(−p)(μ − F) vanishes in expectation at μ = E[F].
                     Compound Poisson–gamma: a point mass at zero plus a skewed tail.

The flux-space losses invert the asinh normalization inside the loss, on a torch port of
AsinhNorm (core/normalize.py), so the model still outputs on the normalized scale.
"""

import json
import math
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

from core.normalize import normalization_from_dict

# Flux-space losses clamp the normalized output to 25% beyond the fitted [0, 1] range of
# the ideal image, so sinh stays finite in float32 if an early step overshoots.
NORMALIZED_OUTPUT_RANGE = (-0.25, 1.25)
BLUR_SIGMA_PX = 1.0
BLUR_RADIUS_PX = 3
# λ of blurred_mse_sparse: the sparsity term is 10% of the blurred term on the storage/runs/mse
# checkpoint (val patches: blurred MSE 5.16e-5, mean |ŷ| 2.99e-2; train patches agree to 1%).
SPARSITY_WEIGHT = 1.7e-4
BETA_NLL_BETA = 0.5
# Bounds on the normalized σ of beta_nll. 1e-3 is ~1% of huber's δ (a median-σ flux at zero);
# 0.5 is half the normalized range, and keeps exp(v/2) in the flux mean finite in float32.
BETA_NLL_SIGMA_RANGE = (1.0e-3, 0.5)
TWEEDIE_POWER = 1.5
# exp(-20) ~ 2e-9 flux is far below any source; exp(12) ~ 1.6e5 is >100x the brightest pixel.
TWEEDIE_LOG_FLUX_RANGE = (-20.0, 12.0)


def load_ideal_norm(data_dir):
    return normalization_from_dict(json.loads((Path(data_dir) / "norm.json").read_text())["ideal"])


def load_median_flux_sigma(data_dir):
    noise = json.loads((Path(data_dir) / "noise.json").read_text())
    return float(noise["crlb_flux_sigma_percentiles"]["50"])


def check_asinh(ideal_norm):
    if ideal_norm.method != "asinh":
        raise ValueError(f"flux-space losses and metrics need an asinh ideal normalization, "
                         f"got {ideal_norm.method}")


def flux_from_normalized(ideal_norm, normalized):
    check_asinh(ideal_norm)
    stretched = normalized * (ideal_norm.hi_s - ideal_norm.lo_s) + ideal_norm.lo_s
    return ideal_norm.median + ideal_norm.beta * torch.sinh(stretched)


def normalized_from_flux(ideal_norm, flux):
    check_asinh(ideal_norm)
    stretched = torch.asinh((flux - ideal_norm.median) / ideal_norm.beta)
    return (stretched - ideal_norm.lo_s) / (ideal_norm.hi_s - ideal_norm.lo_s)


class Loss:
    num_output_channels = 1

    def __call__(self, raw_output, target):
        raise NotImplementedError

    def to_normalized_ideal(self, raw_output):
        return raw_output


class PixelLoss(Loss):
    def __init__(self, pixel_loss):
        self.pixel_loss = pixel_loss

    def __call__(self, raw_output, target):
        return self.pixel_loss(raw_output, target)


class FluxMSELoss(Loss):
    def __init__(self, ideal_norm, median_flux_sigma, relative):
        check_asinh(ideal_norm)
        self.ideal_norm = ideal_norm
        self.median_flux_sigma = median_flux_sigma
        self.relative = relative

    def __call__(self, raw_output, target):
        predicted = flux_from_normalized(self.ideal_norm, raw_output.clamp(*NORMALIZED_OUTPUT_RANGE))
        true = flux_from_normalized(self.ideal_norm, target)
        if self.relative:
            scale = predicted.detach().clamp(min=0.0) + self.median_flux_sigma
        else:
            scale = self.median_flux_sigma
        return torch.mean(((predicted - true) / scale) ** 2)


class BlurredMSELoss(Loss):
    def __init__(self, sparsity_weight):
        offsets = torch.arange(-BLUR_RADIUS_PX, BLUR_RADIUS_PX + 1, dtype=torch.float32)
        profile = torch.exp(-0.5 * (offsets / BLUR_SIGMA_PX) ** 2)
        kernel = torch.outer(profile, profile)
        self.kernel = (kernel / kernel.sum())[None, None]
        self.sparsity_weight = sparsity_weight

    def blur(self, image):
        kernel = self.kernel.to(device=image.device, dtype=image.dtype)
        return F.conv2d(image, kernel, padding=BLUR_RADIUS_PX)

    def __call__(self, raw_output, target):
        loss = F.mse_loss(self.blur(raw_output), self.blur(target))
        if self.sparsity_weight:
            loss = loss + self.sparsity_weight * raw_output.abs().mean()
        return loss


class BetaNLLLoss(Loss):
    num_output_channels = 2

    def __init__(self, ideal_norm):
        check_asinh(ideal_norm)
        self.ideal_norm = ideal_norm

    def mean_and_log_variance(self, raw_output):
        """Mean and log variance of y. The log variance is bounded smoothly (sigmoid), not
        clamped: a raw output of 0 at init lies above the upper bound, and a clamp there
        would start the variance channel with zero gradient."""
        low, high = (2.0 * math.log(sigma) for sigma in BETA_NLL_SIGMA_RANGE)
        return raw_output[:, :1], low + (high - low) * torch.sigmoid(raw_output[:, 1:2])

    def __call__(self, raw_output, target):
        mean, log_variance = self.mean_and_log_variance(raw_output)
        variance = torch.exp(log_variance)
        nll = 0.5 * (log_variance + (target - mean) ** 2 / variance)
        return torch.mean(variance.detach() ** BETA_NLL_BETA * nll)

    def to_normalized_ideal(self, raw_output):
        mean, log_variance = self.mean_and_log_variance(raw_output)
        stretch = self.ideal_norm.hi_s - self.ideal_norm.lo_s
        stretched_mean = mean * stretch + self.ideal_norm.lo_s
        stretched_variance = torch.exp(log_variance) * stretch ** 2
        flux_mean = (self.ideal_norm.median
                     + self.ideal_norm.beta * torch.sinh(stretched_mean) * torch.exp(stretched_variance / 2))
        return normalized_from_flux(self.ideal_norm, flux_mean)


class TweedieLoss(Loss):
    def __init__(self, ideal_norm):
        check_asinh(ideal_norm)
        self.ideal_norm = ideal_norm

    def __call__(self, raw_output, target):
        log_flux = raw_output.clamp(*TWEEDIE_LOG_FLUX_RANGE)
        true = flux_from_normalized(self.ideal_norm, target).clamp(min=0.0)
        power = TWEEDIE_POWER
        deviance = (-true * torch.exp((1 - power) * log_flux) / (1 - power)
                    + torch.exp((2 - power) * log_flux) / (2 - power))
        return torch.mean(deviance)

    def to_normalized_ideal(self, raw_output):
        flux = torch.exp(raw_output.clamp(*TWEEDIE_LOG_FLUX_RANGE))
        return normalized_from_flux(self.ideal_norm, flux)


def huber_delta(ideal_norm, median_flux_sigma):
    flux = torch.tensor([0.0, median_flux_sigma], dtype=torch.float64)
    zero, one_sigma = normalized_from_flux(ideal_norm, flux).tolist()
    return one_sigma - zero


def build_loss(name, ideal_norm, median_flux_sigma):
    builders = {
        "l1": lambda: PixelLoss(F.l1_loss),
        "mse": lambda: PixelLoss(F.mse_loss),
        "huber": lambda: PixelLoss(partial(F.huber_loss,
                                           delta=huber_delta(ideal_norm, median_flux_sigma))),
        "flux_mse": lambda: FluxMSELoss(ideal_norm, median_flux_sigma, relative=False),
        "flux_relative_mse": lambda: FluxMSELoss(ideal_norm, median_flux_sigma, relative=True),
        "blurred_mse": lambda: BlurredMSELoss(sparsity_weight=0.0),
        "blurred_mse_sparse": lambda: BlurredMSELoss(sparsity_weight=SPARSITY_WEIGHT),
        "beta_nll": lambda: BetaNLLLoss(ideal_norm),
        "tweedie": lambda: TweedieLoss(ideal_norm),
    }
    if name not in builders:
        raise ValueError(f"unknown loss: {name} (known: {sorted(builders)})")
    return builders[name]()


def load_loss(name, data_dir):
    return build_loss(name, load_ideal_norm(data_dir), load_median_flux_sigma(data_dir))
