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

Two reference predictors (all zeros, and the train-region mean flux) are scored the same
way so every number has a floor to beat.

Writes into <run_dir>/eval/<dataset>/: photometry.csv (one row per predictor x aperture x
magnitude bin), photometry_summary.json (whole-region flux totals and spurious flux),
and three figures.
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
from core.normalize import normalization_from_dict
from core.runtime import resolve_device
from dataset.gen_data import load_fits
from model.espcn import build_model
from script.eval import predict_full_image

APERTURE_SIZES = (1, 3)
MAGNITUDE_BIN_WIDTH = 0.5
RECOVERED_WITHIN_MAG = 0.5
ROBUST_SCATTER_FACTOR = 1.4826  # MAD -> Gaussian sigma
DETECTION_LEVEL_PERCENTILE = 1.0  # of train-region source fluxes
DM_PLOT_RANGE = (-4.0, 9.0)

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


def binned_rows(predictor, aperture, true_magnitudes, dm, detected, bin_edges):
    rows = []
    bin_index = np.digitize(true_magnitudes, bin_edges) - 1
    for index_bin in range(len(bin_edges) - 1):
        in_bin = bin_index == index_bin
        count = int(in_bin.sum())
        if count == 0:
            continue
        dm_bin = dm[in_bin]
        finite = dm_bin[np.isfinite(dm_bin)]
        median_dm = float(np.median(finite)) if finite.size else float("nan")
        scatter = (ROBUST_SCATTER_FACTOR * float(np.median(np.abs(finite - median_dm)))
                   if finite.size else float("nan"))
        absolute = np.abs(np.nan_to_num(dm_bin, nan=np.inf))
        rows.append({
            "predictor": predictor,
            "aperture": f"{aperture}x{aperture}",
            "mag_low": float(bin_edges[index_bin]),
            "mag_high": float(bin_edges[index_bin + 1]),
            "num_sources": count,
            "median_dm": median_dm,
            "robust_scatter_dm": scatter,
            "frac_detected": float(detected[in_bin].mean()),
            "frac_within_0.1": float((absolute < 0.1).mean()),
            "frac_within_0.2": float((absolute < 0.2).mean()),
            f"frac_within_{RECOVERED_WITHIN_MAG}": float((absolute < RECOVERED_WITHIN_MAG).mean()),
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


def load_model(run_dir):
    checkpoint = torch.load(run_dir / "checkpoint.pt", weights_only=False)
    config = checkpoint["config"]
    device = resolve_device(config.train.device)
    model = build_model(config.model).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config, device


def style_axes(ax):
    ax.grid(True, color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(REFERENCE_LINE_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def plot_dm_vs_magnitude(true_magnitudes, dm_by_aperture, rows, out_path):
    figure, axes = plt.subplots(1, len(APERTURE_SIZES), figsize=(12, 4.5), sharey=True)
    for ax, aperture in zip(axes, APERTURE_SIZES):
        dm = dm_by_aperture[aperture]
        finite = np.isfinite(dm)
        ax.hist2d(true_magnitudes[finite], np.clip(dm[finite], *DM_PLOT_RANGE), bins=(120, 130),
                  range=((true_magnitudes.min(), true_magnitudes.max()), DM_PLOT_RANGE),
                  cmap=SEQUENTIAL_BLUE, norm=LogNorm(), cmin=1)
        model_rows = [row for row in rows
                      if row["predictor"] == "model" and row["aperture"] == f"{aperture}x{aperture}"]
        centers = np.array([(row["mag_low"] + row["mag_high"]) / 2 for row in model_rows])
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
                series = [row for row in rows
                          if row["predictor"] == predictor and row["aperture"] == f"{aperture}x{aperture}"]
                centers = [(row["mag_low"] + row["mag_high"]) / 2 for row in series]
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
    model, config, device = load_model(run_dir)
    data = config.data if data_override is None else load_config(
        Path(data_override) / "resolved_config.yaml").data
    data_dir = Path(data.out_dir)
    ideal_norm = normalization_from_dict(json.loads((data_dir / "norm.json").read_text())["ideal"])

    truth = load_fits(data.ideal_fits).astype(np.float64)
    val_mask = np.load(data_dir / "val_mask.npy")
    observed = np.load(data_dir / "observed.npy")
    model_flux = ideal_norm.inverse(predict_full_image(model, observed, device).astype(np.float64))

    train_sources = ~val_mask & (truth > 0)
    predictors = {
        "model": model_flux,
        "zero": np.zeros_like(truth),
        "train_mean": np.full_like(truth, truth[~val_mask].mean()),
    }

    sources = val_mask & (truth > 0)
    true_magnitudes = magnitude(truth[sources])
    bin_edges = np.arange(np.floor(true_magnitudes.min() / MAGNITUDE_BIN_WIDTH) * MAGNITUDE_BIN_WIDTH,
                          true_magnitudes.max() + MAGNITUDE_BIN_WIDTH, MAGNITUDE_BIN_WIDTH)
    bright = truth >= np.median(truth[sources])
    detection_level = float(np.percentile(truth[train_sources], DETECTION_LEVEL_PERCENTILE))

    rows = []
    summary = {"num_val_sources": int(sources.sum()), "detection_level_flux": detection_level,
               "predictors": {}}
    model_dm_by_aperture = {}
    true_boxes = {aperture: box_sum(truth, aperture) for aperture in APERTURE_SIZES}
    for predictor, predicted in predictors.items():
        for aperture in APERTURE_SIZES:
            predicted_box = box_sum(predicted, aperture)[sources]
            dm = delta_magnitude(predicted_box, true_boxes[aperture][sources])
            detected = predicted_box >= detection_level
            rows.extend(binned_rows(predictor, aperture, true_magnitudes, dm, detected, bin_edges))
            if predictor == "model":
                model_dm_by_aperture[aperture] = dm
        summary["predictors"][predictor] = flux_summary(predicted, truth, val_mask, sources,
                                                        bright, detection_level)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--data", default=None,
                        help="dataset dir to score on (defaults to the run's own dataset)")
    arguments = parser.parse_args()
    main(arguments.run_dir, arguments.data)
