"""Forced photometry of a trained run on the val region, in ideal flux units.

  uv run python -m script.photometry storage/runs/baseline

Every nonzero pixel of the ideal image inside the val blocks is a source at a known
position. The prediction is the exact full-image pass, inverse-normalized to ideal flux
units, and truth is the raw ideal FITS. Each source is scored at two apertures, the pixel
itself (1x1) and a 3x3 box summed identically on prediction and truth. In this crowded
field a 3x3 box usually holds several sources, so the 3x3 score describes the flux of the
region around a source, binned by the brightness of its centre pixel; it does not isolate
flux the model moved onto neighbours, because missed neighbouring sources also enter it.

Per-source error is dm = -2.5 log10(F_pred / F_true) (positive = predicted too faint),
binned by true magnitude m = -2.5 log10(F_true). The zero point is arbitrary; only flux
ratios enter dm. Sources with F_pred <= 0 have no dm and count as not recovered.

The detection level is the 1st percentile of train-region source flux. A source counts as
detected when its predicted aperture flux reaches it, and a true-zero pixel counts as
spurious when its prediction exceeds it, so completeness and false positives share one
threshold. (F_pred > 0 alone is useless: the model's output floor is slightly positive.)

Sources are binned twice: by true magnitude, and by per-source S/N = F_true / sigma_F(p),
where sigma_F(p) is the Cramer-Rao bound on the flux error of a source at pixel p
(crlb_flux_sigma.npy from dataset.estimate_psf: isolated source, known position,
background + photon noise). It is the smallest error any unbiased estimator can reach.
S/N is binned twice too: 0.25 dex wide ("snr") and into the regimes 0.3-1, 1-3, 3-10, >10
("snr_coarse"), whose per-source medians compare_runs tabulates.
In the 1x1 aperture, per bin,

  plain_efficiency  = mean(sigma_F^2) / mean(e^2),                          e = F_pred - F_true,
  robust_efficiency = mean(sigma_F^2) / (median(e)^2 + robust_scatter(e)^2),

so 1 means the model's flux errors are as small as the bound allows. The robust form
ignores the error tails, so with heavy-tailed errors it can overstate skill several-fold;
the plain form counts every error and a few large misses (bright-star cores) can drive it
to ~0. Report both. Caveats on either: the bound is for an isolated source, so in this
crowded field the real bound is larger and the efficiency is conservative; the bound
leaves out PSF-model mismatch, which dominates the residual on bright-star cores; and a
biased estimator (one using priors, or predicting ~0 for sources below the noise) can
exceed 1, which happens below S/N ~1 and does not mean it measures those sources (the
zero predictor does it too). The 3x3 aperture holds several
sources, so it gets neither efficiency.

Per-source scores cannot show flux that is restored on average but spread over the wrong
pixels, so faint flux is also scored in aggregate. The faint region is the val pixels
outside a 5x5 box around every true source (train or val) with S/N >= 3; there the
truth is faint sources only. faint_region_flux_ratio is predicted / true flux summed over
it, and its spread over space comes from the same ratio in 32x32 blocks (median and
16th/84th percentiles, over blocks at least a quarter covered by the faint region).

Two reference predictors (all zeros, and the train-region mean flux) are scored the same
way so every number has a floor to beat.

Needs noise.json and crlb_flux_sigma.npy in the dataset dir (run dataset.estimate_psf first).
Writes into <run_dir>/eval/<dataset>/: photometry.csv (one row per predictor x aperture x
binning x bin), photometry_summary.json (whole-region flux totals, spurious flux, noise),
and four figures.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from core.config import load_config
from core.losses import load_loss
from core.normalize import normalization_from_dict
from core.runtime import resolve_device
from dataset.gen_data import load_fits
from model import build_model
from script.eval import predict_full_image

APERTURE_SIZES = (1, 3)
MAGNITUDE_BIN_WIDTH = 0.5
SNR_BIN_WIDTH_DEX = 0.25
SNR_REFERENCE_LEVELS = (3, 5)
SNR_COARSE_EDGES = (0.3, 1, 3, 10, np.inf)  # the regimes compare_runs tabulates
RECOVERED_WITHIN_MAG = 0.5
ROBUST_SCATTER_FACTOR = 1.4826  # MAD -> Gaussian sigma
DETECTION_LEVEL_PERCENTILE = 1.0  # of train-region source fluxes
DM_PLOT_RANGE = (-4.0, 9.0)
FAINT_REGION_EXCLUSION_BOX = 5  # px, centred on each S/N >= FAINT_REGION_SNR source
FAINT_REGION_SNR = 3
FAINT_REGION_BLOCK_SIZE = 32
FAINT_REGION_MIN_BLOCK_COVERAGE = 0.25

PREDICTOR_COLORS = {"model": "#2a78d6", "zero": "#eb6834", "train_mean": "#1baf7a"}
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "sequential_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
REFERENCE_LINE_COLOR = "#8a8984"
TEXT_SECONDARY = "#52514e"


def box_sum(image, size):
    """Sum over the size x size box centred on each pixel, zero outside the image."""
    radius = size // 2
    padded = np.pad(image.astype(np.float64), radius)
    summed = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1))
    summed[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)
    return (summed[size:, size:] - summed[:-size, size:]
            - summed[size:, :-size] + summed[:-size, :-size])


def magnitude(flux):
    return -2.5 * np.log10(flux)


def delta_magnitude(predicted_flux, true_flux):
    """dm per source; NaN where the prediction is not positive."""
    ratio = np.divide(predicted_flux, true_flux,
                      out=np.full_like(true_flux, np.nan), where=predicted_flux > 0)
    return -2.5 * np.log10(ratio)


def robust_scatter(values, center):
    return ROBUST_SCATTER_FACTOR * float(np.median(np.abs(values - center)))


def binned_rows(predictor, aperture, binning, bin_values, bin_edges, dm, detected, flux_error,
                crlb_sigma):
    """One row per non-empty bin; crlb_sigma (per source) is None where no efficiency applies."""
    rows = []
    bin_index = np.digitize(bin_values, bin_edges) - 1
    for index_bin in range(len(bin_edges) - 1):
        in_bin = bin_index == index_bin
        count = int(in_bin.sum())
        if count == 0:
            continue
        dm_bin = dm[in_bin]
        finite = dm_bin[np.isfinite(dm_bin)]
        median_dm = float(np.median(finite)) if finite.size else float("nan")
        scatter = robust_scatter(finite, median_dm) if finite.size else float("nan")
        absolute = np.abs(np.nan_to_num(dm_bin, nan=np.inf))
        error_bin = flux_error[in_bin]
        median_error = float(np.median(error_bin))
        scatter_error = robust_scatter(error_bin, median_error)
        if crlb_sigma is None:
            plain_efficiency = robust_efficiency = float("nan")
        else:
            bound_variance = float(np.mean(crlb_sigma[in_bin] ** 2))
            plain_efficiency = bound_variance / float(np.mean(error_bin ** 2))
            robust_efficiency = bound_variance / (median_error ** 2 + scatter_error ** 2)
        rows.append({
            "predictor": predictor,
            "aperture": f"{aperture}x{aperture}",
            "binning": binning,
            "bin_low": float(bin_edges[index_bin]),
            "bin_high": float(bin_edges[index_bin + 1]),
            "num_sources": count,
            "median_dm": median_dm,
            "robust_scatter_dm": scatter,
            "frac_detected": float(detected[in_bin].mean()),
            "frac_within_0.1": float((absolute < 0.1).mean()),
            "frac_within_0.2": float((absolute < 0.2).mean()),
            f"frac_within_{RECOVERED_WITHIN_MAG}": float((absolute < RECOVERED_WITHIN_MAG).mean()),
            "median_flux_error": median_error,
            "robust_scatter_flux_error": scatter_error,
            "plain_efficiency": plain_efficiency,
            "robust_efficiency": robust_efficiency,
        })
    return rows


def flux_summary(predicted, truth, val_mask, sources, bright, detection_level):
    true_total = float(truth[val_mask].sum())
    zero_pixels = val_mask & (truth == 0)
    predicted_on_zero = np.clip(predicted[zero_pixels], 0.0, None)
    return {
        "total_flux_ratio": float(predicted[val_mask].sum()) / true_total,
        "bright_source_flux_ratio": float(predicted[sources & bright].sum() / truth[sources & bright].sum()),
        "faint_source_flux_ratio": float(predicted[sources & ~bright].sum() / truth[sources & ~bright].sum()),
        "spurious_flux_fraction": float(predicted_on_zero.sum()) / true_total,
        "num_zero_pixels": int(zero_pixels.sum()),
        "num_spurious_pixels": int((predicted_on_zero > detection_level).sum()),
    }


def faint_region_mask(truth, crlb_map, val_mask):
    bright_sources = truth >= FAINT_REGION_SNR * crlb_map
    return val_mask & (box_sum(bright_sources, FAINT_REGION_EXCLUSION_BOX) == 0)


def block_sums(image, block_size):
    """Sum over non-overlapping block_size blocks; the partial blocks at the far edges are dropped."""
    num_rows, num_columns = image.shape[0] // block_size, image.shape[1] // block_size
    trimmed = image[:num_rows * block_size, :num_columns * block_size]
    return trimmed.reshape(num_rows, block_size, num_columns, block_size).sum(axis=(1, 3))


def faint_region_summary(predicted, truth, faint_region):
    predicted_in_region = np.where(faint_region, predicted, 0.0)
    truth_in_region = np.where(faint_region, truth, 0.0)
    coverage = block_sums(faint_region, FAINT_REGION_BLOCK_SIZE) / FAINT_REGION_BLOCK_SIZE ** 2
    block_truth = block_sums(truth_in_region, FAINT_REGION_BLOCK_SIZE)
    used_blocks = (coverage >= FAINT_REGION_MIN_BLOCK_COVERAGE) & (block_truth > 0)
    block_ratios = (block_sums(predicted_in_region, FAINT_REGION_BLOCK_SIZE)[used_blocks]
                    / block_truth[used_blocks])
    percentile_16, median, percentile_84 = np.percentile(block_ratios, (16, 50, 84))
    return {
        "faint_region_flux_ratio": float(predicted_in_region.sum() / truth_in_region.sum()),
        "faint_region_block_flux_ratio_median": float(median),
        "faint_region_block_flux_ratio_p16": float(percentile_16),
        "faint_region_block_flux_ratio_p84": float(percentile_84),
        "num_faint_region_blocks": int(used_blocks.sum()),
    }


def load_model(run_dir, data_dir=None):
    """(model, its map to the normalized ideal, config, device). The map uses the
    normalization of data_dir, by default the run's own dataset."""
    checkpoint = torch.load(run_dir / "checkpoint.pt", weights_only=False)
    config = checkpoint["config"]
    device = resolve_device(config.train.device)
    loss = load_loss(config.train.loss, data_dir or config.data.out_dir)
    model = build_model(config.model, loss.num_output_channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, loss.to_normalized_ideal, config, device


def style_axes(ax):
    ax.grid(True, color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(REFERENCE_LINE_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def select_rows(rows, predictor, aperture, binning):
    return [row for row in rows if row["predictor"] == predictor
            and row["aperture"] == f"{aperture}x{aperture}" and row["binning"] == binning]


def plot_dm_vs_magnitude(true_magnitudes, dm_by_aperture, rows, out_path):
    figure, axes = plt.subplots(1, len(APERTURE_SIZES), figsize=(12, 4.5), sharey=True)
    for ax, aperture in zip(axes, APERTURE_SIZES):
        dm = dm_by_aperture[aperture]
        finite = np.isfinite(dm)
        ax.hist2d(true_magnitudes[finite], np.clip(dm[finite], *DM_PLOT_RANGE), bins=(120, 130),
                  range=((true_magnitudes.min(), true_magnitudes.max()), DM_PLOT_RANGE),
                  cmap=SEQUENTIAL_BLUE, norm=LogNorm(), cmin=1)
        model_rows = select_rows(rows, "model", aperture, "magnitude")
        centers = np.array([(row["bin_low"] + row["bin_high"]) / 2 for row in model_rows])
        medians = np.array([row["median_dm"] for row in model_rows])
        scatters = np.array([row["robust_scatter_dm"] for row in model_rows])
        ax.axhline(0.0, color=REFERENCE_LINE_COLOR, linewidth=1)
        ax.plot(centers, medians, color="#eb6834", linewidth=2, label="binned median")
        ax.plot(centers, medians + scatters, color="#eb6834", linewidth=1, linestyle="--",
                label="median ± robust scatter")
        ax.plot(centers, medians - scatters, color="#eb6834", linewidth=1, linestyle="--")
        ax.set_title(f"{aperture}x{aperture} aperture  ({int((~finite).sum())} sources with "
                     "F_pred ≤ 0 not shown)", fontsize=10)
        ax.set_xlabel("true magnitude  (−2.5 log₁₀ F_true, brighter →left)", fontsize=9)
        style_axes(ax)
    axes[0].set_ylabel(f"Δm = −2.5 log₁₀(F_pred / F_true)  (clipped to {DM_PLOT_RANGE})", fontsize=9)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    figure.suptitle("Model photometric error vs true magnitude (val region; Δm > 0 = predicted too faint)",
                    fontsize=11)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"wrote {out_path}")


def plot_completeness(rows, out_path):
    metrics = (("frac_detected", "fraction detected (F_pred ≥ detection level)"),
               (f"frac_within_{RECOVERED_WITHIN_MAG}", f"fraction within {RECOVERED_WITHIN_MAG} mag"))
    figure, axes = plt.subplots(len(APERTURE_SIZES), len(metrics), figsize=(11, 7),
                                sharex=True, sharey=True)
    for index_row, aperture in enumerate(APERTURE_SIZES):
        for index_column, (key, label) in enumerate(metrics):
            ax = axes[index_row, index_column]
            for predictor, color in PREDICTOR_COLORS.items():
                series = select_rows(rows, predictor, aperture, "magnitude")
                centers = [(row["bin_low"] + row["bin_high"]) / 2 for row in series]
                ax.plot(centers, [row[key] for row in series], color=color, linewidth=2,
                        marker="o", markersize=4, label=predictor)
            ax.set_title(f"{aperture}x{aperture}: {label}", fontsize=10)
            ax.set_ylim(-0.03, 1.03)
            style_axes(ax)
            if index_row == len(APERTURE_SIZES) - 1:
                ax.set_xlabel("true magnitude (brighter →left)", fontsize=9)
    # Series overlap at the ends of every panel, so direct labels collide; one shared legend.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 1.0), ncol=len(labels),
                  fontsize=9, frameon=False)
    figure.suptitle("Source recovery vs true magnitude (val region)", fontsize=11, x=0.3)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"wrote {out_path}")


def plot_snr(rows, out_path):
    """1x1 recovery and efficiency vs per-source S/N (bin centres in log10 S/N)."""
    panels = ((f"frac_within_{RECOVERED_WITHIN_MAG}", f"fraction within {RECOVERED_WITHIN_MAG} mag", "linear", "top"),
              ("robust_efficiency", "robust efficiency  (Cramér–Rao σ_F² / robust flux error²; 1 = at the bound)",
               "log", "bottom"))
    figure, axes = plt.subplots(1, len(panels), figsize=(12, 4.5), sharex=True)
    for ax, (key, label, y_scale, note_position) in zip(axes, panels):
        for predictor, color in PREDICTOR_COLORS.items():
            series = select_rows(rows, predictor, 1, "snr")
            centers = [10 ** ((row["bin_low"] + row["bin_high"]) / 2) for row in series]
            ax.plot(centers, [row[key] for row in series], color=color, linewidth=2,
                    marker="o", markersize=4, label=predictor)
        ax.set_xscale("log")
        x_limits = ax.get_xlim()
        ax.axvspan(x_limits[0], 1.0, color="#f0efec", zorder=0)
        ax.set_xlim(x_limits)
        ax.text(0.02, 0.97 if note_position == "top" else 0.03,
                "below the noise (S/N < 1):\nefficiency > 1 here is not skill",
                transform=ax.transAxes, fontsize=8, color=TEXT_SECONDARY, va=note_position)
        for level in SNR_REFERENCE_LEVELS:
            ax.axvline(level, color=REFERENCE_LINE_COLOR, linewidth=1, linestyle=":")
        if key == "robust_efficiency":
            ax.axhline(1.0, color=REFERENCE_LINE_COLOR, linewidth=1)
        ax.set_yscale(y_scale)
        ax.set_title(f"1x1: {label}", fontsize=10)
        ax.set_xlabel(f"per-source S/N = F_true / Cramér–Rao σ_F  (dotted: S/N {SNR_REFERENCE_LEVELS})",
                      fontsize=9)
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 1.0), ncol=len(labels),
                  fontsize=9, frameon=False)
    figure.suptitle("Recovery and efficiency vs S/N (val region)", fontsize=11, x=0.3)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"wrote {out_path}")


def load_noise(data_dir):
    """(noise.json contents, per-pixel Cramer-Rao flux sigma map)."""
    if not (data_dir / "crlb_flux_sigma.npy").exists():
        raise FileNotFoundError(f"no crlb_flux_sigma.npy in {data_dir}; run dataset.estimate_psf first")
    return (json.loads((data_dir / "noise.json").read_text()),
            np.load(data_dir / "crlb_flux_sigma.npy").astype(np.float64))


def plot_flux_scatter(true_flux, predicted_flux, out_path):
    positive = predicted_flux > 0
    log_true = np.log10(true_flux[positive])
    log_predicted = np.log10(predicted_flux[positive])
    low = min(log_true.min(), np.percentile(log_predicted, 0.1))
    high = max(log_true.max(), log_predicted.max())
    figure, ax = plt.subplots(figsize=(6, 5.5))
    image = ax.hist2d(log_true, log_predicted, bins=150, range=((low, high), (low, high)),
                      cmap=SEQUENTIAL_BLUE, norm=LogNorm(), cmin=1)[3]
    ax.plot([low, high], [low, high], color=REFERENCE_LINE_COLOR, linewidth=1, label="F_pred = F_true")
    ax.set_xlabel("log₁₀ F_true  (ideal flux units)", fontsize=9)
    ax.set_ylabel("log₁₀ F_pred", fontsize=9)
    ax.set_title(f"Model vs truth, 1x1 aperture, val sources\n"
                 f"({int((~positive).sum())} of {positive.size} with F_pred ≤ 0 not shown)", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    style_axes(ax)
    figure.colorbar(image, ax=ax, label="sources per cell")
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"wrote {out_path}")


def main(run_dir, data_override=None):
    run_dir = Path(run_dir)
    model, to_normalized_ideal, config, device = load_model(run_dir, data_override)
    data = config.data if data_override is None else load_config(
        Path(data_override) / "resolved_config.yaml").data
    data_dir = Path(data.out_dir)
    ideal_norm = normalization_from_dict(json.loads((data_dir / "norm.json").read_text())["ideal"])
    noise, crlb_map = load_noise(data_dir)

    truth = load_fits(data.ideal_fits).astype(np.float64)
    val_mask = np.load(data_dir / "val_mask.npy")
    observed = np.load(data_dir / "observed.npy")
    model_flux = ideal_norm.inverse(
        predict_full_image(model, to_normalized_ideal, observed, device).astype(np.float64))

    train_sources = ~val_mask & (truth > 0)
    predictors = {
        "model": model_flux,
        "zero": np.zeros_like(truth),
        "train_mean": np.full_like(truth, truth[~val_mask].mean()),
    }

    sources = val_mask & (truth > 0)
    true_magnitudes = magnitude(truth[sources])
    crlb_sigma = crlb_map[sources]
    log_snr = np.log10(truth[sources] / crlb_sigma)
    binnings = {
        "magnitude": (true_magnitudes, np.arange(
            np.floor(true_magnitudes.min() / MAGNITUDE_BIN_WIDTH) * MAGNITUDE_BIN_WIDTH,
            true_magnitudes.max() + MAGNITUDE_BIN_WIDTH, MAGNITUDE_BIN_WIDTH)),
        "snr": (log_snr, np.arange(
            np.floor(log_snr.min() / SNR_BIN_WIDTH_DEX) * SNR_BIN_WIDTH_DEX,
            log_snr.max() + SNR_BIN_WIDTH_DEX, SNR_BIN_WIDTH_DEX)),
        "snr_coarse": (log_snr, np.log10(SNR_COARSE_EDGES)),
    }
    bright = truth >= np.median(truth[sources])
    detection_level = float(np.percentile(truth[train_sources], DETECTION_LEVEL_PERCENTILE))
    faint_region = faint_region_mask(truth, crlb_map, val_mask)

    rows = []
    summary = {"num_val_sources": int(sources.sum()), "detection_level_flux": detection_level,
               "noise": noise, "median_crlb_flux_sigma": float(np.median(crlb_sigma)),
               **{f"frac_val_sources_snr_above_{level}": float((log_snr >= np.log10(level)).mean())
                  for level in SNR_REFERENCE_LEVELS},
               "faint_region_fraction_of_val_area": float(faint_region.sum() / val_mask.sum()),
               "faint_region_fraction_of_val_true_flux": float(truth[faint_region].sum()
                                                               / truth[val_mask].sum()),
               "predictors": {}}
    model_dm_by_aperture = {}
    true_boxes = {aperture: box_sum(truth, aperture) for aperture in APERTURE_SIZES}
    for predictor, predicted in predictors.items():
        for aperture in APERTURE_SIZES:
            predicted_box = box_sum(predicted, aperture)[sources]
            dm = delta_magnitude(predicted_box, true_boxes[aperture][sources])
            detected = predicted_box >= detection_level
            flux_error = predicted_box - true_boxes[aperture][sources]
            for binning, (bin_values, bin_edges) in binnings.items():
                rows.extend(binned_rows(predictor, aperture, binning, bin_values, bin_edges, dm,
                                        detected, flux_error, crlb_sigma if aperture == 1 else None))
            if predictor == "model":
                model_dm_by_aperture[aperture] = dm
        summary["predictors"][predictor] = {
            **flux_summary(predicted, truth, val_mask, sources, bright, detection_level),
            **faint_region_summary(predicted, truth, faint_region)}

    eval_dir = run_dir / "eval" / data_dir.name
    eval_dir.mkdir(parents=True, exist_ok=True)
    with open(eval_dir / "photometry.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (eval_dir / "photometry_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {eval_dir / 'photometry.csv'} and photometry_summary.json")
    print(json.dumps(summary, indent=2))

    plot_dm_vs_magnitude(true_magnitudes, model_dm_by_aperture, rows, eval_dir / "photometry_dm.png")
    plot_completeness(rows, eval_dir / "photometry_completeness.png")
    plot_flux_scatter(truth[sources], model_flux[sources], eval_dir / "photometry_flux.png")
    plot_snr(rows, eval_dir / "photometry_snr.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--data", default=None,
                        help="dataset dir to score on (defaults to the run's own dataset)")
    arguments = parser.parse_args()
    main(arguments.run_dir, arguments.data)
