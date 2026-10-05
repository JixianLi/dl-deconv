"""Export baseline-model images as physical-flux FITS for inspection in a FITS viewer.

  uv run python -m script.export_fits runs/baseline

Writes into <run_dir>/result/<dataset>/ (see the README it emits for the layout):
full input/target/prediction images, plus per-patch input/target/prediction FITS
for every non-overlapping (stride = patch_size) train and val window, named by its
corner in the full image. Everything is inverse-normalized back to physical flux,
so a FITS viewer sees real values, not the [0, 1] training scale.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

from core.config import load_config
from core.losses import load_loss
from core.normalize import normalization_from_dict
from core.runtime import resolve_device
from dataset.gen_data import load_fits
from dataset.patch_dataset import PatchDataset
from model import build_model
from script.eval import predict_full_image, predict_patches

README = """\
# Baseline model images ({dataset})

Physical-flux FITS exported from run `{run}` on dataset `{dataset}`. Values are
inverse-normalized back to the original flux scale (not the [0, 1] training
scale), so any FITS viewer applies its own stretch to real numbers.

## Full images

- `input.fits`      full observed image (the model's input)
- `target.fits`     full ideal ground truth
- `prediction.fits` full prediction, one convolutional pass over the whole input

## Per-patch images

`training/` and `validation/` hold one triplet per non-overlapping window
(corners on a {patch_size}px grid) lying wholly inside that split's blocks:

- `yYYYY-xXXXX-input.fits`      observed patch (model input)
- `yYYYY-xXXXX-target.fits`     ideal patch (ground truth)
- `yYYYY-xXXXX-prediction.fits` model output for that patch alone

`YYYY`, `XXXX` are the patch's top-left corner (row, column) in the full image.
The patch prediction is the model run on the patch by itself, so its edges see
zero padding; the same pixels in `prediction.fits` see the real neighbours and
can differ within 4px of the patch border.
"""


def write_fits(path, image):
    fits.PrimaryHDU(np.ascontiguousarray(image, dtype=np.float32)).writeto(path, overwrite=True)


def main(run_dir, data_override=None):
    run_dir = Path(run_dir)
    checkpoint = torch.load(run_dir / "checkpoint.pt", weights_only=False)
    config = checkpoint["config"]
    data = config.data if data_override is None else load_config(
        Path(data_override) / "resolved_config.yaml").data
    device = resolve_device(config.train.device)

    data_dir = Path(data.out_dir)
    loss = load_loss(config.train.loss, data_dir)
    model = build_model(config.model, loss.num_output_channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    norms = json.loads((data_dir / "norm.json").read_text())
    observed_norm = normalization_from_dict(norms["observed"])
    ideal_norm = normalization_from_dict(norms["ideal"])

    result_dir = run_dir / "result" / data_dir.name
    result_dir.mkdir(parents=True, exist_ok=True)
    write_fits(result_dir / "input.fits", load_fits(data.observed_fits))
    write_fits(result_dir / "target.fits", load_fits(data.ideal_fits))
    observed_full = np.load(data_dir / "observed.npy")
    write_fits(result_dir / "prediction.fits",
               ideal_norm.inverse(predict_full_image(model, loss.to_normalized_ideal, observed_full, device)))

    counts = {}
    for split, folder in (("train", "training"), ("val", "validation")):
        (result_dir / folder).mkdir(parents=True, exist_ok=True)
        dataset = PatchDataset(data_dir, split, stride=data.patch_size)
        observed = np.stack([dataset[index][0].numpy() for index in range(len(dataset))])
        ideal = np.stack([dataset[index][1].numpy() for index in range(len(dataset))])
        predictions = predict_patches(model, loss.to_normalized_ideal, observed, device)
        triplets = zip(dataset.corners, observed_norm.inverse(observed), ideal_norm.inverse(ideal),
                       ideal_norm.inverse(predictions))
        for (y, x), obs, tgt, pred in triplets:
            stem = result_dir / folder / f"y{y:04d}-x{x:04d}"
            write_fits(f"{stem}-input.fits", obs[0])
            write_fits(f"{stem}-target.fits", tgt[0])
            write_fits(f"{stem}-prediction.fits", pred[0])
        counts[split] = len(dataset)

    (result_dir / "README.md").write_text(README.format(run=run_dir.name, dataset=data_dir.name,
                                                       patch_size=data.patch_size))
    print(f"wrote {counts['train']} train + {counts['val']} val patch triplets "
          f"and 3 full images to {result_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--data", default=None,
                        help="dataset dir to export (defaults to the run's own dataset)")
    arguments = parser.parse_args()
    main(arguments.run_dir, arguments.data)
