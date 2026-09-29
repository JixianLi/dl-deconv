"""Evaluate a trained deconvolution run on train and val, and visualize.

  uv run python -m script.eval runs/baseline

Reads the checkpoint (which carries its own config), scores the non-overlapping
(stride = patch_size) train and val windows (loss + PSNR), saves Observed / Ideal /
Predicted comparison grids per split, and predicts the full image in one
convolutional pass, into <run_dir>/eval/. No held-out set: we report on both splits.
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from core.config import load_config
from core.runtime import psnr, resolve_device, set_seed
from dataset.patch_dataset import PatchDataset
from model import build_model

NUM_EXAMPLES = 6
CMAP = "inferno"
FULL_IMAGE_TILE_SIZE = 512


def score(model, dataset, device):
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    loss_fn = torch.nn.L1Loss()
    total_loss = 0.0
    total_psnr = 0.0
    with torch.no_grad():
        for observed, ideal in loader:
            observed, ideal = observed.to(device), ideal.to(device)
            prediction = model(observed)
            total_loss += loss_fn(prediction, ideal).item() * observed.size(0)
            total_psnr += psnr(prediction, ideal) * observed.size(0)
    return total_loss / len(dataset), total_psnr / len(dataset)


def visualize(model, dataset, device, indices, out_path):
    rows = len(indices)
    fig, axes = plt.subplots(rows, 3, figsize=(7, 2.3 * rows))
    axes = np.atleast_2d(axes)
    titles = ["Observed (input)", "Ideal (target)", "Predicted"]
    for row, index in enumerate(indices):
        observed, ideal = dataset[index]
        with torch.no_grad():
            prediction = model(observed.unsqueeze(0).to(device)).squeeze(0).cpu()
        sample_psnr = psnr(prediction, ideal)
        for col, image in enumerate((observed, ideal, prediction)):
            ax = axes[row, col]
            ax.imshow(image.squeeze(0).numpy(), origin="lower", cmap=CMAP, vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[col], fontsize=9)
        axes[row, 2].set_ylabel(f"PSNR {sample_psnr:.2f}", fontsize=8)
        axes[row, 2].yaxis.set_label_position("right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def predict_patches(model, observed, device, batch_size=256):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(observed), batch_size):
            chunk = torch.from_numpy(observed[start:start + batch_size]).to(device)
            outputs.append(model(chunk).cpu().numpy())
    return np.concatenate(outputs)


def predict_full_image(model, observed, device, tile_size=FULL_IMAGE_TILE_SIZE):
    """Model output over the whole (H, W) image, identical to a single forward pass.

    One pass at 4040x4040 would need ~4 GB of activations, so we run tiles. Each tile's
    input is extended by the receptive-field radius (a halo) on every side that is not an
    image border; the zero padding at the halo edge then only corrupts halo outputs, which
    are cropped. At true image borders the tile edge IS the image edge, so the model's
    zero padding matches the one-pass result there too.
    """
    halo = model.receptive_field_radius
    height, width = observed.shape
    prediction = np.empty((height, width), dtype=np.float32)
    with torch.no_grad():
        for y0 in range(0, height, tile_size):
            for x0 in range(0, width, tile_size):
                y1, x1 = min(y0 + tile_size, height), min(x0 + tile_size, width)
                in_y0, in_x0 = max(y0 - halo, 0), max(x0 - halo, 0)
                in_y1, in_x1 = min(y1 + halo, height), min(x1 + halo, width)
                tile = np.ascontiguousarray(observed[in_y0:in_y1, in_x0:in_x1], dtype=np.float32)
                output = model(torch.from_numpy(tile)[None, None].to(device))[0, 0].cpu().numpy()
                prediction[y0:y1, x0:x1] = output[y0 - in_y0:y1 - in_y0, x0 - in_x0:x1 - in_x0]
    return prediction


def visualize_full(model, data, device, out_path):
    data_dir = Path(data.out_dir)
    observed_full = np.load(data_dir / "observed.npy")
    ideal_full = np.load(data_dir / "ideal.npy")
    predicted_full = predict_full_image(model, observed_full, device)

    mse = float(np.mean((predicted_full - ideal_full) ** 2))
    full_psnr = float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
    print(f"full image: psnr={full_psnr:.3f}")

    # observed and ideal use different normalizations, so each panel gets its own scale;
    # the prediction shares the ideal scale since the model outputs in ideal space.
    observed_vmax = float(np.nanpercentile(observed_full, 99.5))
    ideal_vmax = float(np.nanpercentile(ideal_full, 99.5))
    panels = [
        (observed_full, observed_vmax, f"Observed ({observed_full.shape[0]}x{observed_full.shape[1]})"),
        (ideal_full, ideal_vmax, f"Ideal ground truth ({ideal_full.shape[0]}x{ideal_full.shape[1]})"),
        (predicted_full, ideal_vmax,
         f"Predicted ({predicted_full.shape[0]}x{predicted_full.shape[1]})"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for ax, (image, vmax, title) in zip(axes, panels):
        ax.imshow(image, origin="lower", cmap=CMAP, vmin=0.0, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"wrote {out_path}")


def main(run_dir, data_override=None):
    run_dir = Path(run_dir)
    checkpoint = torch.load(run_dir / "checkpoint.pt", weights_only=False)
    config = checkpoint["config"]
    # Model comes from the checkpoint; the data spec can be a different dataset (e.g. another
    # band) via --data, so we can test how the trained weights transfer cross-domain.
    data = config.data if data_override is None else load_config(
        Path(data_override) / "resolved_config.yaml").data
    set_seed(config.seed)
    device = resolve_device(config.train.device)

    model = build_model(config.model).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    eval_dir = run_dir / "eval" / Path(data.out_dir).name
    eval_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    for split in ("train", "val"):
        dataset = PatchDataset(data.out_dir, split, stride=data.patch_size)
        loss, mean_psnr = score(model, dataset, device)
        print(f"{split}: n={len(dataset)}  l1={loss:.5f}  psnr={mean_psnr:.3f}")
        indices = rng.choice(len(dataset), size=min(NUM_EXAMPLES, len(dataset)), replace=False)
        visualize(model, dataset, device, indices, eval_dir / f"{split}_examples.png")

    visualize_full(model, data, device, eval_dir / "full_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--data", default=None,
                        help="dataset dir to evaluate on (defaults to the run's own dataset)")
    arguments = parser.parse_args()
    main(arguments.run_dir, arguments.data)
