"""Render an export_fits output directory as browsable PNGs.

  uv run python -m script.visualize_export tmp-storage/baseline

Writes into <export_dir>/figures/:

- `input.png`, `prediction.png`, `target.png` — the three full images, one
  output pixel per image pixel. Resampling a sparse point-source field to a
  smaller figure drops faint single-pixel sources, so these are written at
  native size rather than laid out side by side.
- `patches-<split>.png` — a 3 x 6 grid; rows are input / prediction / target,
  columns are six randomly chosen patches.

Display stretch is asinh, matching the normalization the model trains under.
The input carries its own color scale because it sits in a different flux
regime (background near 86) from the ideal field (median 0); prediction and
target always share a scale, so those panels can be compared directly.

These scales are for display only. They are unrelated to the invertible
normalizations in core.normalize, which is why nothing here is called a
normalization.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize, ManualInterval

from dataset.gen_data import load_fits

KINDS = ("input", "prediction", "target")
CMAP = "inferno"
NUM_PATCH_COLUMNS = 6
LOW_PERCENTILE = 0.5
HIGH_PERCENTILE = 99.5
ASINH_LINEAR_FRACTION = 0.1  # AsinhStretch(a): share of the scaled range that stays ~linear


def build_display_scale(reference):
    """asinh scale spanning the reference image's central percentile range."""
    finite = reference[np.isfinite(reference)]
    low, high = (float(value) for value in np.percentile(finite, [LOW_PERCENTILE, HIGH_PERCENTILE]))
    # A flat image (an all-zero ideal patch, say) gives a zero-width interval, which
    # would divide by zero inside the stretch.
    if high <= low:
        high = low + 1.0
    return ImageNormalize(interval=ManualInterval(low, high),
                          stretch=AsinhStretch(ASINH_LINEAR_FRACTION))


def build_colormap():
    colormap = plt.get_cmap(CMAP).copy()
    colormap.set_bad("black")  # prediction is NaN wherever no patch covered the pixel
    return colormap


def paired_scales(images):
    """Own scale for the input, one shared scale for prediction and target."""
    target_scale = build_display_scale(images["target"])
    return {"input": build_display_scale(images["input"]),
            "prediction": target_scale,
            "target": target_scale}


def write_full_png(image, scale, colormap, path):
    scaled = np.ma.filled(np.ma.asarray(scale(image)).astype(float), np.nan)
    plt.imsave(path, scaled, cmap=colormap, vmin=0.0, vmax=1.0, origin="lower")


def render_full_images(export_dir, figures_dir, colormap):
    images = {kind: load_fits(export_dir / f"{kind}.fits") for kind in KINDS}
    scales = paired_scales(images)
    for kind in KINDS:
        path = figures_dir / f"{kind}.png"
        write_full_png(images[kind], scales[kind], colormap, path)
        height, width = images[kind].shape
        print(f"wrote {path}  {width}x{height}")


def patch_indices(split_dir):
    # macOS writes `._<name>` AppleDouble companions on non-HFS drives; they match the glob.
    return sorted(path.name.removesuffix("-input.fits") for path in split_dir.glob("*-input.fits")
                  if not path.name.startswith("."))


def render_patch_grid(split_dir, figures_dir, colormap, seed):
    indices = patch_indices(split_dir)
    if not indices:
        raise ValueError(f"no patch triplets found in {split_dir}")
    rng = np.random.default_rng(seed)
    columns = rng.choice(len(indices), size=min(NUM_PATCH_COLUMNS, len(indices)), replace=False)
    chosen = [indices[position] for position in columns]

    figure, axes = plt.subplots(len(KINDS), len(chosen),
                                figsize=(2.0 * len(chosen), 2.15 * len(KINDS)))
    axes = np.asarray(axes).reshape(len(KINDS), len(chosen))
    for column, index in enumerate(chosen):
        patches = {kind: load_fits(split_dir / f"{index}-{kind}.fits") for kind in KINDS}
        scales = paired_scales(patches)
        for row, kind in enumerate(KINDS):
            ax = axes[row, column]
            ax.imshow(patches[kind], origin="lower", cmap=colormap, norm=scales[kind])
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(index, fontsize=9)
            if column == 0:
                ax.set_ylabel(kind, fontsize=10)

    figure.suptitle(f"{split_dir.name} patches (asinh, scaled per patch; "
                    "prediction and target share each column's scale)", fontsize=10)
    figure.tight_layout()
    path = figures_dir / f"patches-{split_dir.name}.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    print(f"wrote {path}  patches {' '.join(chosen)}")


def main(export_dir, split, seed):
    export_dir = Path(export_dir)
    figures_dir = export_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    colormap = build_colormap()
    render_full_images(export_dir, figures_dir, colormap)
    render_patch_grid(export_dir / split, figures_dir, colormap, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("export_dir", help="an export_fits output dir, e.g. tmp-storage/baseline")
    parser.add_argument("--split", default="validation", choices=("training", "validation"))
    parser.add_argument("--seed", type=int, default=0, help="re-roll the random patch selection")
    arguments = parser.parse_args()
    main(arguments.export_dir, arguments.split, arguments.seed)
