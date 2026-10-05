"""Compare the runs of a loss screen: final photometry, convergence and runtime.

  uv run python -m script.compare_losses storage/runs/loss_screen_compare \
      storage/runs/loss_screen/*/

Reads each run's metrics.csv and eval/<dataset>/photometry.csv + photometry_summary.json
(run script.train and script.photometry first). Runs are named by their directory; any
number of runs works (compare_runs is capped at 4 for its direct-labeled plots).

Writes into the output dir loss_screen.csv, one row per run:
  - final photometry of the model, 1x1 aperture: faint-region flux ratio with its block
    p16/median/p84, faint-source flux ratio, spurious flux fraction, and per coarse S/N
    regime the fraction within 0.5 mag, plain efficiency and robust efficiency;
  - convergence, per loss-agnostic val metric: the iteration from which the metric stays
    past 90% of its change from the first to the final validation, and its value at 15k
    and 50k iterations (linearly interpolated between validations; NaN past the run's end);
  - runtime: median train iterations per second over validation intervals.
For every run <name>_seed43 whose <name> is also present, a row seed_noise_<name> holds
|<name> - <name>_seed43| per column: differences between losses smaller than that are noise.

Also writes loss_screen_convergence.png: small multiples, one column per candidate (every
run but the l1 and mse references), one row per val metric, the references in gray.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from script.compare_runs import RECOVERED_KEY, load_run, regime_label, rows_for
from script.photometry import REFERENCE_LINE_COLOR, TEXT_SECONDARY, style_axes

CONVERGENCE_METRICS = (
    ("val_l1_normalized", "val L1 (normalized)"),
    ("val_mse_normalized", "val MSE (normalized)"),
    ("val_faint_flux_ratio", "faint-source flux ratio"),
    ("val_spurious_flux_fraction", "spurious flux fraction"),
    ("val_frac_within_0.5mag_snr1_10", "frac within 0.5 mag, S/N 1-10"),
)
CONVERGED_FRACTION = 0.9
CHECKPOINT_ITERATIONS = (15_000, 50_000)
Y_RANGE_FROM_ITERATION = 15_000  # early transients (e.g. faint ratio ~800) would flatten every panel
SEED_SUFFIX = "_seed43"
CANDIDATE_COLOR = "#2a78d6"  # categorical slot 1
REFERENCE_STYLES = {"l1": ("#8a8984", "-"), "mse": ("#52514e", "--")}


def load_metrics(run_dir):
    with open(Path(run_dir) / "metrics.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def iteration_converged(iterations, values):
    """First validation from which the metric stays past CONVERGED_FRACTION of its first-to-final change."""
    threshold = values[0] + CONVERGED_FRACTION * (values[-1] - values[0])
    direction = np.sign(values[-1] - values[0])
    not_past = np.flatnonzero((values - threshold) * direction < 0)
    return float(iterations[not_past[-1] + 1] if not_past.size else iterations[0])


def value_at(iterations, values, iteration):
    if iteration > iterations[-1]:
        return float("nan")
    return float(np.interp(iteration, iterations, values))


def regime_key(row):
    return regime_label(row).replace("S/N > ", "snr_above_").replace("S/N ", "snr_")


def table_row(rows, summary, metrics):
    model = summary["predictors"]["model"]
    row = {
        "faint_region_flux_ratio": model["faint_region_flux_ratio"],
        "faint_region_block_p16": model["faint_region_block_flux_ratio_p16"],
        "faint_region_block_median": model["faint_region_block_flux_ratio_median"],
        "faint_region_block_p84": model["faint_region_block_flux_ratio_p84"],
        "faint_source_flux_ratio": model["faint_source_flux_ratio"],
        "spurious_flux_fraction": model["spurious_flux_fraction"],
    }
    for regime in rows_for(rows, "snr_coarse"):
        key = regime_key(regime)
        row[f"{key}_{RECOVERED_KEY}"] = regime[RECOVERED_KEY]
        row[f"{key}_plain_efficiency"] = regime["plain_efficiency"]
        row[f"{key}_robust_efficiency"] = regime["robust_efficiency"]
    iterations = metrics["iteration"]
    for key, _ in CONVERGENCE_METRICS:
        row[f"{key}_converged_iteration"] = iteration_converged(iterations, metrics[key])
        for checkpoint in CHECKPOINT_ITERATIONS:
            row[f"{key}_at_{checkpoint // 1000}k"] = value_at(iterations, metrics[key], checkpoint)
    interval_iterations = np.diff(np.concatenate([[0.0], iterations]))
    row["median_train_iterations_per_second"] = float(np.median(interval_iterations / metrics["train_seconds"]))
    return row


def seed_noise_rows(table):
    noise = {}
    for name, row in table.items():
        base = name.removesuffix(SEED_SUFFIX)
        if name.endswith(SEED_SUFFIX) and base in table:
            noise[f"seed_noise_{base}"] = {key: abs(table[base][key] - value) for key, value in row.items()}
    return noise


def print_table(title, table, columns):
    name_width = max(len(name) for name in table)
    widths = [max(len(header), 9) for _, header in columns]
    print(f"\n{title}\n{'run':<{name_width}}  " + "  ".join(
        f"{header:>{width}}" for (_, header), width in zip(columns, widths)))
    for name, row in table.items():
        print(f"{name:<{name_width}}  " + "  ".join(
            f"{row.get(key, float('nan')):>{width}.4g}" for (key, _), width in zip(columns, widths)))


def print_tables(table):
    print_table("Final photometry (val region, model, 1x1)", table, [
        ("faint_region_flux_ratio", "faint region"), ("faint_region_block_p16", "block p16"),
        ("faint_region_block_median", "block median"), ("faint_region_block_p84", "block p84"),
        ("faint_source_flux_ratio", "faint sources"), ("spurious_flux_fraction", "spurious")])
    regime_keys = sorted({key.removesuffix(f"_{RECOVERED_KEY}") for row in table.values()
                          for key in row if key.endswith(f"_{RECOVERED_KEY}")},
                         key=lambda key: float(key.split("_")[-1].split("-")[0]))
    for suffix, title in ((RECOVERED_KEY, f"fraction within {RECOVERED_KEY.split('_')[-1]} mag"),
                          ("plain_efficiency", "plain efficiency"),
                          ("robust_efficiency", "robust efficiency")):
        print_table(f"Per S/N regime: {title}", table,
                    [(f"{regime}_{suffix}", regime) for regime in regime_keys])
    print_table("Convergence: iteration from which each val metric stays past 90% of its change", table,
                [(f"{key}_converged_iteration", key.removeprefix("val_")) for key, _ in CONVERGENCE_METRICS])
    for checkpoint in CHECKPOINT_ITERATIONS:
        print_table(f"Val metrics at {checkpoint // 1000}k iterations", table,
                    [(f"{key}_at_{checkpoint // 1000}k", key.removeprefix("val_")) for key, _ in CONVERGENCE_METRICS])
    print_table("Runtime", table, [("median_train_iterations_per_second", "train it/s")])


def write_csv(table, out_path):
    columns = list(dict.fromkeys(key for row in table.values() for key in row))
    with open(out_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", *columns])
        for name, row in table.items():
            writer.writerow([name, *(row.get(key, float("nan")) for key in columns)])
    print(f"\nwrote {out_path}")


def row_y_limits(metrics_by_run, key):
    values = []
    for metrics in metrics_by_run.values():
        late = metrics["iteration"] >= Y_RANGE_FROM_ITERATION
        values.append(metrics[key][late] if late.any() else metrics[key])
    values = np.concatenate(values)
    low, high = float(values.min()), float(values.max())
    pad = 0.05 * (high - low or abs(high) or 1.0)
    return low - pad, high + pad


def plot_convergence(metrics_by_run, out_path):
    candidates = [name for name in metrics_by_run if name not in REFERENCE_STYLES]
    references = [name for name in REFERENCE_STYLES if name in metrics_by_run]
    figure, axes = plt.subplots(len(CONVERGENCE_METRICS), len(candidates), sharex=True, sharey="row",
                                figsize=(2.1 * len(candidates) + 1.2, 1.8 * len(CONVERGENCE_METRICS) + 1.0),
                                squeeze=False)
    for index_row, (key, label) in enumerate(CONVERGENCE_METRICS):
        for index_column, candidate in enumerate(candidates):
            ax = axes[index_row, index_column]
            for reference in references:
                color, linestyle = REFERENCE_STYLES[reference]
                metrics = metrics_by_run[reference]
                ax.plot(metrics["iteration"], metrics[key], color=color, linestyle=linestyle, linewidth=1.5,
                        label=f"{reference} (reference)")
            metrics = metrics_by_run[candidate]
            ax.plot(metrics["iteration"], metrics[key], color=CANDIDATE_COLOR, linewidth=2, label="candidate")
            if key == "val_faint_flux_ratio":
                ax.axhline(1.0, color=REFERENCE_LINE_COLOR, linewidth=1)
            style_axes(ax)
            if index_row == 0:
                ax.set_title(candidate, fontsize=9)
            if index_column == 0:
                ax.set_ylabel(label, fontsize=8)
            if index_row == len(CONVERGENCE_METRICS) - 1:
                ax.set_xlabel("iteration", fontsize=8)
        axes[index_row, 0].set_ylim(row_y_limits(metrics_by_run, key))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", ncol=len(labels), fontsize=9, frameon=False)
    figure.suptitle(f"Loss screen: loss-agnostic val metrics per candidate (y range from iterations "
                    f"≥ {Y_RANGE_FROM_ITERATION // 1000}k; faint ratio target 1)",
                    fontsize=10, x=0.02, ha="left", color=TEXT_SECONDARY)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"wrote {out_path}")


def main(out_dir, run_dirs, dataset):
    names = [Path(run_dir).name for run_dir in run_dirs]
    metrics_by_run = {name: load_metrics(run_dir) for name, run_dir in zip(names, run_dirs)}
    table = {name: table_row(*load_run(run_dir, dataset), metrics_by_run[name])
             for name, run_dir in zip(names, run_dirs)}
    table.update(seed_noise_rows(table))
    print_tables(table)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(table, out_dir / "loss_screen.csv")
    plot_convergence(metrics_by_run, out_dir / "loss_screen_convergence.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--dataset", default="baseline",
                        help="eval/<dataset> subdir to read from each run")
    arguments = parser.parse_args()
    main(arguments.out_dir, arguments.run_dirs, arguments.dataset)
