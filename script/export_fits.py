"""Export baseline-model images as physical-flux FITS for inspection in a FITS viewer.

  uv run python -m script.export_fits runs/baseline

Writes into <run_dir>/result/<dataset>/ (see the README it emits for the layout):
full input/target/prediction images, plus per-patch input/target/prediction FITS
for every train and val patch, indexed by global manifest index. Everything is
inverse-normalized back to physical flux, so a FITS viewer sees real values, not
the [0, 1] training scale.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

from core.config import load_config
from core.normalize import normalization_from_dict
from core.runtime import resolve_device
from dataset.gen_data import load_fits
from model.espcn import build_model
from script.eval import predict_patches, reconstruct

README = """\
# Baseline model images ({dataset})

Physical-flux FITS exported from run `{run}` on dataset `{dataset}`. Values are
inverse-normalized back to the original flux scale (not the [0, 1] training
scale), so any FITS viewer applies its own stretch to real numbers.

## Full images

- `input.fits`      full observed image (the model's input)
- `target.fits`     full ideal ground truth
- `prediction.fits` full prediction, tiled from the per-patch predictions

`prediction.fits` is NaN wherever no patch covered a pixel (image edges the patch
grid does not reach); `input.fits` and `target.fits` are the untiled originals.

## Per-patch images

`training/` and `validation/` hold one triplet per patch:

- `NNNN-input.fits`      observed patch (model input)
- `NNNN-target.fits`     ideal patch (ground truth)
- `NNNN-prediction.fits` model output for that patch

`NNNN` is the **global manifest index**: row `NNNN` of `{dataset}/manifest.csv`
in the dataset dir carries that patch's split and corner (corner_y, corner_x)
in the full image. Indices are not contiguous within a folder — train and val
patches are interleaved across the image, so each folder holds its own subset.
"""


def gather_patches(data_dir, manifest):
    """Observed+ideal patch stacks in manifest order, drawn from the per-split arrays."""
    observed = {split: np.load(Path(data_dir) / f"{split}_observed.npy") for split in ("train", "val")}
    ideal = {split: np.load(Path(data_dir) / f"{split}_ideal.npy") for split in ("train", "val")}
    cursor = {"train": 0, "val": 0}
    observed_ordered, ideal_ordered = [], []
    for row in manifest:
        split = row["split"]
        observed_ordered.append(observed[split][cursor[split]])
        ideal_ordered.append(ideal[split][cursor[split]])
        cursor[split] += 1
    return np.stack(observed_ordered), np.stack(ideal_ordered)


def write_fits(path, image):
    fits.PrimaryHDU(np.ascontiguousarray(image, dtype=np.float32)).writeto(path, overwrite=True)


def main(run_dir, data_override=None):
    run_dir = Path(run_dir)
    checkpoint = torch.load(run_dir / "checkpoint.pt", weights_only=False)
    config = checkpoint["config"]
    data = config.data if data_override is None else load_config(
        Path(data_override) / "resolved_config.yaml").data
    device = resolve_device(config.train.device)

    model = build_model(config.model).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    data_dir = Path(data.out_dir)
    norms = json.loads((data_dir / "norm.json").read_text())
    observed_norm = normalization_from_dict(norms["observed"])
    ideal_norm = normalization_from_dict(norms["ideal"])
    with open(data_dir / "manifest.csv") as handle:
        manifest = list(csv.DictReader(handle))

    observed, ideal = gather_patches(data_dir, manifest)
    corners = [(int(row["corner_y"]), int(row["corner_x"])) for row in manifest]
    predictions = predict_patches(model, observed, device)

    observed_flux = observed_norm.inverse(observed)
    target_flux = ideal_norm.inverse(ideal)
    prediction_flux = ideal_norm.inverse(predictions)

    result_dir = run_dir / "result" / data_dir.name
    (result_dir / "training").mkdir(parents=True, exist_ok=True)
    (result_dir / "validation").mkdir(parents=True, exist_ok=True)

    height = load_fits(data.observed_fits).shape[0]
    write_fits(result_dir / "input.fits", load_fits(data.observed_fits))
    write_fits(result_dir / "target.fits", load_fits(data.ideal_fits))
    prediction_full = ideal_norm.inverse(reconstruct(predictions, corners, height, data.patch_size))
    write_fits(result_dir / "prediction.fits", prediction_full)

    pad = max(4, len(str(len(manifest) - 1)))
    counts = {"train": 0, "val": 0}
    for row, obs, tgt, pred in zip(manifest, observed_flux, target_flux, prediction_flux):
        folder = "training" if row["split"] == "train" else "validation"
        stem = result_dir / folder / f"{int(row['index']):0{pad}d}"
        write_fits(f"{stem}-input.fits", obs[0])
        write_fits(f"{stem}-target.fits", tgt[0])
        write_fits(f"{stem}-prediction.fits", pred[0])
        counts[row["split"]] += 1

    (result_dir / "README.md").write_text(README.format(run=run_dir.name, dataset=data_dir.name))
    print(f"wrote {counts['train']} train + {counts['val']} val patch triplets "
          f"and 3 full images to {result_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--data", default=None,
                        help="dataset dir to export (defaults to the run's own dataset)")
    arguments = parser.parse_args()
    main(arguments.run_dir, arguments.data)
