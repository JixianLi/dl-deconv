"""Generate a deconvolution window dataset from a paired FITS image.

Pipeline (run as `uv run python -m dataset.gen_data config/baseline.yaml`):

  1. Load the observed (trimmed) and ideal (untrimmed) FITS; they share a pixel grid.
  2. Tile the image into spatial blocks (data.split_block_size) and send a seeded subset
     of whole blocks to val, giving a per-pixel val mask.
  3. Enumerate window top-left corners (step = data.stride). A window belongs to a split
     only if ALL its pixels lie in that split's blocks; windows straddling a train/val
     border are dropped, so train and val never share a pixel.
  4. Fit a normalization on the train-block pixels of the full image, separately for
     observed and ideal, since the two live in very different value regimes.
  5. Save the full normalized images, the per-split window corners, the val mask, the
     normalization params, and the resolved config into data.out_dir.

Windows are not materialized: at stride 1 they would be ~480 GB. PatchDataset crops
them from observed.npy / ideal.npy on demand. Images are float32, shape (H, W),
normalized to roughly [0, 1]; corners are int32, shape (N, 2) as (y, x).
"""

import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

from core.config import dump_config, load_config
from core.normalize import fit_normalization


def load_fits(path):
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim == 2:
                return np.ascontiguousarray(hdu.data, dtype=np.float32)
    raise ValueError(f"no 2-d image HDU found in {path}")


def block_split_mask(height, width, block_size, val_fraction, rng):
    """Per-pixel bool image, True where the pixel's block was sent to val (seeded)."""
    num_blocks_y = -(-height // block_size)
    num_blocks_x = -(-width // block_size)
    blocks = [(block_y, block_x) for block_y in range(num_blocks_y) for block_x in range(num_blocks_x)]
    num_val = round(val_fraction * len(blocks))
    val_mask = np.zeros((height, width), dtype=bool)
    for position in rng.permutation(len(blocks))[:num_val]:
        block_y, block_x = blocks[position]
        val_mask[block_y * block_size:(block_y + 1) * block_size,
                 block_x * block_size:(block_x + 1) * block_size] = True
    return val_mask


def window_corners(val_mask, patch_size, stride):
    """(train_corners, val_corners) for windows lying wholly in one split, on a `stride` grid.

    The val-pixel count of every window comes from a summed-area table: with
    S[y, x] = sum of val_mask[:y, :x], the window at (y, x) holds
    S[y+P, x+P] - S[y, x+P] - S[y+P, x] + S[y, x] val pixels, so all windows cost O(H*W).
    """
    height, width = val_mask.shape
    summed = np.zeros((height + 1, width + 1), dtype=np.int64)
    summed[1:, 1:] = val_mask.cumsum(axis=0).cumsum(axis=1)
    size = patch_size
    val_counts = (summed[size:, size:] - summed[:-size, size:]
                  - summed[size:, :-size] + summed[:-size, :-size])

    on_grid = np.zeros_like(val_counts, dtype=bool)
    on_grid[::stride, ::stride] = True
    train_corners = np.argwhere(on_grid & (val_counts == 0)).astype(np.int32)
    val_corners = np.argwhere(on_grid & (val_counts == size * size)).astype(np.int32)
    return train_corners, val_corners


def main(config_path):
    config = load_config(config_path)
    data = config.data
    out_dir = Path(data.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.seed)
    observed_image = load_fits(data.observed_fits)
    ideal_image = load_fits(data.ideal_fits)
    if observed_image.shape != ideal_image.shape:
        raise ValueError(f"observed {observed_image.shape} and ideal {ideal_image.shape} "
                         "FITS must share a pixel grid")
    height, width = observed_image.shape

    val_mask = block_split_mask(height, width, data.split_block_size, data.val_fraction, rng)
    train_corners, val_corners = window_corners(val_mask, data.patch_size, data.stride)

    observed_norm = fit_normalization(observed_image[~val_mask], data.observed_normalize,
                                      asinh_softening=data.asinh_softening)
    ideal_norm = fit_normalization(ideal_image[~val_mask], data.ideal_normalize,
                                   asinh_softening=data.asinh_softening)

    np.save(out_dir / "observed.npy", observed_norm.forward(observed_image).astype(np.float32))
    np.save(out_dir / "ideal.npy", ideal_norm.forward(ideal_image).astype(np.float32))
    np.save(out_dir / "train_corners.npy", train_corners)
    np.save(out_dir / "val_corners.npy", val_corners)
    np.save(out_dir / "val_mask.npy", val_mask)
    print(f"image {height}x{width}  val pixels {int(val_mask.sum())}/{val_mask.size}")
    print(f"train: {len(train_corners)} windows  val: {len(val_corners)} windows  "
          f"(patch {data.patch_size}, stride {data.stride})")

    (out_dir / "norm.json").write_text(json.dumps(
        {"observed": observed_norm.to_dict(), "ideal": ideal_norm.to_dict()}, indent=2))
    dump_config(config, out_dir / "resolved_config.yaml")
    print(f"wrote dataset to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
