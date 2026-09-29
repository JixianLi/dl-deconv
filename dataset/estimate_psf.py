"""Estimate the PSF and noise that map the ideal image onto the observed image.

  uv run python -m dataset.estimate_psf config/baseline.yaml

Forward model, per pixel q:

  O[q] = b + sum_k K[k] I[q - k] + n[q]

O is the observed (trimmed) image, I the ideal image, b a constant background, K the
kernel on a (2R+1)^2 support, and n the noise. K is the PSF times the gain, in observed
counts per ideal flux unit, so sum(K) is the gain. Subtracting the image means removes b;
least squares then gives the normal equations

  sum_k K[k] R_II[k - j] = R_IO[j]   for every lag j in the support,

with R_II[d] = sum_p I[p] I[p + d] and R_IO[d] = sum_p I[p] O[p + d] (mean-subtracted
images), both computed by FFT on zero-padded images so nothing wraps around. The fit
uses the whole image: K and the noise model serve evaluation only, never training.

Noise model. With S = (I conv K)[q] the fitted source signal in counts above background,
the per-pixel noise variance is modelled as background plus photon noise:

  var(S) = a + c * max(S, 0).

a and c are fit by relative least squares to the robust (1.4826 * MAD) residual variance
in bins of S, excluding a border of R px where the convolution sees zero padding. On
M31 H100 this matches within ~15% up to S ~ 500 counts; on bright-star cores the residual
variance is several times larger, from PSF mismatch (stars off pixel centres), which is a
limit of our forward model rather than noise, so it is deliberately left out.

Per-source bound. For a source of flux F at pixel p, isolated, position known, with the
rest of the image known, the Fisher information about F is sum_k K[k]^2 / var(p + k), so
the Cramer-Rao bound on its flux error is

  sigma_F(p) = 1 / sqrt( sum_k K[k]^2 / var(p + k) ),

a cross-correlation of 1/var with K^2 (computed by FFT). The image outside the frame
contributes no information.

Writes into data.out_dir: psf.npy (K, float64, shape (2R+1, 2R+1)), noise.json, and
crlb_flux_sigma.npy (sigma_F per pixel, ideal flux units, float32, shape (H, W)).
"""

import json
import sys
from pathlib import Path

import numpy as np
from numpy.fft import irfft2, rfft2

from core.config import load_config
from dataset.gen_data import load_fits

KERNEL_RADIUS = 20  # the fitted PSF holds 99.5% of its flux within 20 px on M31 H100
ROBUST_SCATTER_FACTOR = 1.4826  # MAD -> Gaussian sigma
NOISE_BIN_PERCENTILES = np.concatenate([np.linspace(0, 95, 20),
                                        [97, 98, 99, 99.5, 99.8, 99.9, 99.95, 99.99, 100]])


def place_kernel(kernel, padded_shape):
    """Kernel of odd size with its centre at lag 0, on a periodic grid of padded_shape."""
    radius = kernel.shape[0] // 2
    lag_y, lag_x = (grid.ravel() for grid in np.mgrid[-radius:radius + 1, -radius:radius + 1])
    placed = np.zeros(padded_shape)
    placed[lag_y % padded_shape[0], lag_x % padded_shape[1]] = kernel.ravel()
    return placed


def fit_kernel(observed, ideal, radius):
    """Least-squares K on a (2*radius+1)^2 support; returns (K, fitted observed image)."""
    height, width = ideal.shape
    padded_shape = (height + 4 * radius, width + 4 * radius)
    ideal_centered = ideal - ideal.mean()
    observed_centered = observed - observed.mean()
    ideal_spectrum = rfft2(ideal_centered, padded_shape)
    observed_spectrum = rfft2(observed_centered, padded_shape)
    autocorrelation = irfft2(np.abs(ideal_spectrum) ** 2, padded_shape)
    cross_correlation = irfft2(np.conj(ideal_spectrum) * observed_spectrum, padded_shape)

    lag_y, lag_x = (grid.ravel() for grid in np.mgrid[-radius:radius + 1, -radius:radius + 1])
    normal_matrix = autocorrelation[(lag_y[None, :] - lag_y[:, None]) % padded_shape[0],
                                    (lag_x[None, :] - lag_x[:, None]) % padded_shape[1]]
    right_hand_side = cross_correlation[lag_y % padded_shape[0], lag_x % padded_shape[1]]
    size = 2 * radius + 1
    kernel = np.linalg.solve(normal_matrix, right_hand_side).reshape(size, size)

    fitted = irfft2(ideal_spectrum * rfft2(place_kernel(kernel, padded_shape)),
                    padded_shape)[:height, :width]
    return kernel, fitted + observed.mean()


def robust_sigma(values):
    return ROBUST_SCATTER_FACTOR * float(np.median(np.abs(values - np.median(values))))


def fit_variance_model(signal, residual):
    """(a, c) of var = a + c * max(signal, 0), by relative least squares over signal bins."""
    edges = np.unique(np.percentile(signal, NOISE_BIN_PERCENTILES))
    bin_signals, bin_variances = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (signal >= low) & (signal < high)
        bin_signals.append(max(float(np.median(signal[in_bin])), 0.0))
        bin_variances.append(robust_sigma(residual[in_bin]) ** 2)
    bin_signals, bin_variances = np.array(bin_signals), np.array(bin_variances)
    design = np.stack([np.ones_like(bin_signals), bin_signals], axis=1) / bin_variances[:, None]
    (floor, per_count), *_ = np.linalg.lstsq(design, np.ones_like(bin_signals), rcond=None)
    return float(floor), float(per_count)


def crlb_flux_sigma(kernel, variance):
    """sigma_F(p) = 1 / sqrt(sum_k K[k]^2 / variance[p + k]), zero information off the frame."""
    radius = kernel.shape[0] // 2
    height, width = variance.shape
    padded_shape = (height + 2 * radius, width + 2 * radius)
    weight_spectrum = rfft2(1.0 / variance, padded_shape)
    kernel_squared_spectrum = rfft2(place_kernel(kernel ** 2, padded_shape))
    information = irfft2(weight_spectrum * np.conj(kernel_squared_spectrum),
                         padded_shape)[:height, :width]
    return 1.0 / np.sqrt(information)


def main(config_path):
    data = load_config(config_path).data
    out_dir = Path(data.out_dir)
    observed = load_fits(data.observed_fits).astype(np.float64)
    ideal = load_fits(data.ideal_fits).astype(np.float64)

    radius = KERNEL_RADIUS
    kernel, fitted = fit_kernel(observed, ideal, radius)
    interior = (slice(radius, -radius), slice(radius, -radius))
    residual = (observed - fitted)[interior]
    distance = np.hypot(*np.mgrid[-radius:radius + 1, -radius:radius + 1])
    gain = float(kernel.sum())
    background = float(observed.mean() - gain * ideal.mean())
    signal = fitted - background
    variance_floor, variance_per_count = fit_variance_model(signal[interior], residual)
    variance = variance_floor + variance_per_count * np.clip(signal, 0.0, None)
    crlb = crlb_flux_sigma(kernel, variance)

    noise = {
        "kernel_radius": radius,
        "gain": gain,
        "background": background,
        "variance_floor": variance_floor,
        "variance_per_count": variance_per_count,
        "residual_robust_sigma": robust_sigma(residual),
        "residual_std": float(residual.std()),
        "variance_explained": float(1.0 - residual.var() / observed[interior].var()),
        "matched_filter_norm": float(np.sqrt((kernel ** 2).sum())),
        "enclosed_flux": {str(r): float(kernel[distance <= r].sum() / gain) for r in (1, 2, 3, 5, 8, 12, 20)},
        "crlb_flux_sigma_percentiles": {str(q): float(np.percentile(crlb, q)) for q in (1, 50, 99, 100)},
    }
    np.save(out_dir / "psf.npy", kernel)
    np.save(out_dir / "crlb_flux_sigma.npy", crlb.astype(np.float32))
    (out_dir / "noise.json").write_text(json.dumps(noise, indent=2))
    print(json.dumps(noise, indent=2))
    print(f"wrote psf.npy, crlb_flux_sigma.npy and noise.json to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
