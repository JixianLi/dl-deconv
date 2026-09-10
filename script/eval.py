"""Evaluate a trained deconvolution run over ALL train and val samples, and visualize.

  uv run python -m script.eval runs/baseline

Reads the checkpoint (which carries its own config), scores every train and val
patch (loss + PSNR), and saves Observed / Ideal / Predicted comparison grids per
split into <run_dir>/eval/. No held-out set: we report on both splits.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from core.config import load_config
from core.normalize import normalization_from_dict
from core.runtime import psnr, resolve_device, set_seed
from dataset.gen_data import load_fits
from dataset.patch_dataset import PatchDataset
from model.espcn import build_model

NUM_EXAMPLES = 6
CMAP = "inferno"


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


def gather_observed_patches(data_dir, manifest):
    """Return (observed stack, corners) in manifest order, drawing from the per-split arrays."""
    arrays = {split: np.load(Path(data_dir) / f"{split}_observed.npy") for split in ("train", "val")}
    cursor = {"train": 0, "val": 0}
    patches = []
    corners = []
    for row in manifest:
        split = row["split"]
        patches.append(arrays[split][cursor[split]])
        cursor[split] += 1
        corners.append((int(row["corner_y"]), int(row["corner_x"])))
    return np.stack(patches), corners


def predict_patches(model, observed, device, batch_size=256):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(observed), batch_size):
            chunk = torch.from_numpy(observed[start:start + batch_size]).to(device)
            outputs.append(model(chunk).cpu().numpy())
    return np.concatenate(outputs)


def reconstruct(predictions, corners, size, patch_size):
    """Place predicted patches into a full canvas; overlaps resolve by max (fmax skips nan)."""
    canvas = np.full((size, size), np.nan, dtype=np.float32)
    for (y, x), prediction in zip(corners, predictions):
        view = canvas[y:y + patch_size, x:x + patch_size]
        canvas[y:y + patch_size, x:x + patch_size] = np.fmax(view, prediction[0])
    return canvas


def visualize_full(model, data, device, out_path):
    data_dir = Path(data.out_dir)
    norms = json.loads((data_dir / "norm.json").read_text())
    observed_norm = normalization_from_dict(norms["observed"])
    ideal_norm = normalization_from_dict(norms["ideal"])
    with open(data_dir / "manifest.csv") as handle:
        manifest = list(csv.DictReader(handle))

    observed_patches, corners = gather_observed_patches(data_dir, manifest)
    predictions = predict_patches(model, observed_patches, device)

    observed_raw = load_fits(data.observed_fits)
    ideal_raw = load_fits(data.ideal_fits)
    height = observed_raw.shape[0]
    predicted_full = reconstruct(predictions, corners, height, data.patch_size)
    observed_full = observed_norm.forward(observed_raw)
    ideal_full = ideal_norm.forward(ideal_raw)

    covered = ~np.isnan(predicted_full)
    mse = float(np.mean((predicted_full[covered] - ideal_full[covered]) ** 2))
    full_psnr = float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
    print(f"full image: covered={int(covered.sum())}/{covered.size}  psnr={full_psnr:.3f}")

    # observed and ideal use different normalizations, so each panel gets its own scale;
    # the prediction shares the ideal scale since the model outputs in ideal space.
    observed_vmax = float(np.nanpercentile(observed_full, 99.5))
    ideal_vmax = float(np.nanpercentile(ideal_full, 99.5))
    panels = [
        (observed_full, observed_vmax, f"Observed ({observed_full.shape[0]}x{observed_full.shape[1]})"),
        (ideal_full, ideal_vmax, f"Ideal ground truth ({ideal_full.shape[0]}x{ideal_full.shape[1]})"),
        (np.nan_to_num(predicted_full), ideal_vmax,
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
        dataset = PatchDataset(data.out_dir, split)
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
