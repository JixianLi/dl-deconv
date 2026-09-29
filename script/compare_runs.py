"""Compare the photometry of several runs on one dataset.

  uv run python -m script.compare_runs storage/runs/compare-weak-signal \
      storage/runs/baseline storage/runs/mse storage/runs/deep storage/runs/deep_mse

Reads each run's eval/<dataset>/photometry.csv and photometry_summary.json (run
script.photometry first), prints the faint-flux totals and the per-S/N-regime scores
of the model predictor, and writes compare_snr.png into the output dir. Runs are
named by their directory.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from script.photometry import (RECOVERED_WITHIN_MAG, REFERENCE_LINE_COLOR, SNR_REFERENCE_LEVELS,
                               TEXT_SECONDARY, style_axes)

# Categorical slots 1-4 of the dataviz palette, assigned in run order.
RUN_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
RECOVERED_KEY = f"frac_within_{RECOVERED_WITHIN_MAG}"
END_LABEL_MIN_GAP_POINTS = 10


def load_run(run_dir, dataset):
    eval_dir = Path(run_dir) / "eval" / dataset
    with open(eval_dir / "photometry.csv", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["predictor"] == "model" and row["aperture"] == "1x1"]
    summary = json.loads((eval_dir / "photometry_summary.json").read_text())
    return rows, summary


def rows_for(rows, binning):
    return [{key: value if key in ("predictor", "aperture", "binning") else float(value)
             for key, value in row.items()} for row in rows if row["binning"] == binning]


def regime_label(row):
    low, high = 10 ** row["bin_low"], 10 ** row["bin_high"]
    return f"S/N > {low:g}" if high == float("inf") else f"S/N {low:g}-{high:g}"


def print_tables(runs):
    name_width = max(len(name) for name in runs)
    print(f"\n{'run':<{name_width}}  faint-region ratio  block median [p16, p84]  "
          "faint-source ratio  spurious frac")
    for name, (_, summary) in runs.items():
        model = summary["predictors"]["model"]
        print(f"{name:<{name_width}}  {model['faint_region_flux_ratio']:18.3f}  "
              f"{model['faint_region_block_flux_ratio_median']:6.3f} "
              f"[{model['faint_region_block_flux_ratio_p16']:.3f}, "
              f"{model['faint_region_block_flux_ratio_p84']:.3f}]    "
              f"{model['faint_source_flux_ratio']:18.3f}  {model['spurious_flux_fraction']:13.4f}")
    first_summary = next(iter(runs.values()))[1]
    print(f"(faint region: {first_summary['faint_region_fraction_of_val_area']:.1%} of val area, "
          f"{first_summary['predictors']['model']['num_faint_region_blocks']} blocks)")

    print(f"\n{'run':<{name_width}}  {'regime':<14}  {'sources':>8}  "
          f"{f'within {RECOVERED_WITHIN_MAG} mag':>15}  {'median F error':>14}  {'efficiency':>10}")
    for name, (rows, _) in runs.items():
        for row in rows_for(rows, "snr_coarse"):
            print(f"{name:<{name_width}}  {regime_label(row):<14}  {int(row['num_sources']):8d}  "
                  f"{row[RECOVERED_KEY]:15.3f}  {row['median_flux_error']:14.5f}  "
                  f"{row['efficiency']:10.3f}")


def label_line_ends(figure, ax, line_ends):
    """Name each line at its end, skipping a label that would collide with one already placed.

    Converging lines would need their labels nudged apart, which detaches them from the
    lines; the legend identifies any line left unlabeled.
    """
    placed_heights = []
    points_per_pixel = 72 / figure.dpi
    for name, x, y in line_ends:
        height = ax.transData.transform((x, y))[1] * points_per_pixel
        if any(abs(height - placed) < END_LABEL_MIN_GAP_POINTS for placed in placed_heights):
            continue
        placed_heights.append(height)
        ax.annotate(name, (x, y), xytext=(6, 0), textcoords="offset points",
                    fontsize=8, color=TEXT_SECONDARY, va="center")


def plot_snr(runs, out_path):
    panels = ((RECOVERED_KEY, f"fraction within {RECOVERED_WITHIN_MAG} mag", "linear"),
              ("efficiency", "efficiency  (Cramér–Rao σ_F² / flux MSE; 1 = at the bound)", "log"))
    figure, axes = plt.subplots(1, len(panels), figsize=(12, 4.5), sharex=True)
    for ax, (key, label, y_scale) in zip(axes, panels):
        line_ends = []
        for (name, (rows, _)), color in zip(runs.items(), RUN_COLORS):
            series = rows_for(rows, "snr")
            centers = [10 ** ((row["bin_low"] + row["bin_high"]) / 2) for row in series]
            values = [row[key] for row in series]
            ax.plot(centers, values, color=color, linewidth=2, marker="o", markersize=4, label=name)
            line_ends.append((name, centers[-1], values[-1]))
        ax.set_yscale(y_scale)
        ax.set_xscale("log")
        x_limits = ax.get_xlim()
        ax.axvspan(x_limits[0], 1.0, color="#f0efec", zorder=0)
        ax.set_xlim(x_limits[0], x_limits[1] * 3)  # room for the end labels
        for level in SNR_REFERENCE_LEVELS:
            ax.axvline(level, color=REFERENCE_LINE_COLOR, linewidth=1, linestyle=":")
        if key == "efficiency":
            ax.axhline(1.0, color=REFERENCE_LINE_COLOR, linewidth=1)
        label_line_ends(figure, ax, line_ends)
        ax.set_title(f"1x1: {label}", fontsize=10)
        ax.set_xlabel(f"per-source S/N  (shaded: S/N < 1; dotted: S/N {SNR_REFERENCE_LEVELS})",
                      fontsize=9)
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 1.0), ncol=len(labels),
                  fontsize=9, frameon=False)
    figure.suptitle("Recovery and efficiency vs S/N by run (val region)", fontsize=11, x=0.3)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"\nwrote {out_path}")


def main(out_dir, run_dirs, dataset):
    if len(run_dirs) > len(RUN_COLORS):
        raise ValueError(f"at most {len(RUN_COLORS)} runs per comparison, got {len(run_dirs)}")
    runs = {Path(run_dir).name: load_run(run_dir, dataset) for run_dir in run_dirs}
    print_tables(runs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_snr(runs, out_dir / "compare_snr.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--dataset", default="baseline",
                        help="eval/<dataset> subdir to read from each run")
    arguments = parser.parse_args()
    main(arguments.out_dir, arguments.run_dirs, arguments.dataset)
