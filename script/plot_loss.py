"""Plot a run's train and val loss on a log scale.

  uv run python -m script.plot_loss storage/runs/mse

Reads <run_dir>/metrics.csv and writes <run_dir>/figures/loss_curve_log.png. Each train
point is the mean loss over the preceding val_interval iterations; each val point is
measured at the end of that interval.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from script.photometry import style_axes

SERIES = (("train_loss", "train", "#2a78d6"), ("val_loss", "val", "#eb6834"))


def main(run_dir):
    run_dir = Path(run_dir)
    with open(run_dir / "metrics.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    iterations = [int(row["iteration"]) for row in rows]
    figure, ax = plt.subplots(figsize=(8, 4.5))
    for key, name, color in SERIES:
        values = [float(row[key]) for row in rows]
        ax.plot(iterations, values, color=color, linewidth=2, marker="o", markersize=3,
                label=f"{name}  (final {values[-1]:.2e})")
    ax.set_yscale("log")
    ax.set_xlabel("iteration", fontsize=9)
    ax.set_ylabel("loss (normalized space, log scale)", fontsize=9)
    # The lines converge, so end labels would collide; the legend carries identity and final values.
    ax.legend(fontsize=9, frameon=False)
    ax.set_title(f"{run_dir.name}: loss per validation interval", fontsize=10)
    style_axes(ax)
    figure.tight_layout()
    out_path = run_dir / "figures" / "loss_curve_log.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    main(parser.parse_args().run_dir)
