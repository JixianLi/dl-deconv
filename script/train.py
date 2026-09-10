"""Train the deconvolution baseline from a config file.

  uv run python -m script.train config/baseline.yaml

Writes into train.out_dir: checkpoint.pt (weights + config), metrics.csv (per
epoch), and resolved_config.yaml. Assumes the dataset already exists at
data.out_dir (run dataset.gen_data first).
"""

import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from core.config import dump_config, load_config
from core.runtime import psnr, resolve_device, set_seed
from dataset.patch_dataset import PatchDataset
from model.espcn import build_model

LOSSES = {"l1": torch.nn.L1Loss, "mse": torch.nn.MSELoss}


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    count = 0
    with torch.no_grad():
        for observed, ideal in loader:
            observed, ideal = observed.to(device), ideal.to(device)
            prediction = model(observed)
            batch = observed.size(0)
            total_loss += loss_fn(prediction, ideal).item() * batch
            total_psnr += psnr(prediction, ideal) * batch
            count += batch
    return total_loss / count, total_psnr / count


def main(config_path):
    config = load_config(config_path)
    train_config = config.train
    set_seed(config.seed)
    device = resolve_device(train_config.device)
    out_dir = Path(train_config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = PatchDataset(config.data.out_dir, "train")
    val_set = PatchDataset(config.data.out_dir, "val")
    train_loader = DataLoader(train_set, batch_size=train_config.batch_size, shuffle=True,
                              num_workers=train_config.num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=train_config.batch_size, shuffle=False,
                            num_workers=train_config.num_workers)

    model = build_model(config.model).to(device)
    loss_fn = LOSSES[train_config.loss]()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)

    print(f"device={device}  train={len(train_set)}  val={len(val_set)}")
    metrics = []
    for epoch in range(1, train_config.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for observed, ideal in train_loader:
            observed, ideal = observed.to(device), ideal.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(observed), ideal)
            loss.backward()
            optimizer.step()
            running += loss.item() * observed.size(0)
            seen += observed.size(0)
        train_loss = running / seen
        val_loss, val_psnr = evaluate(model, val_loader, loss_fn, device)
        metrics.append((epoch, train_loss, val_loss, val_psnr))
        print(f"epoch {epoch:3d}  train_loss={train_loss:.5f}  "
              f"val_loss={val_loss:.5f}  val_psnr={val_psnr:.3f}")

    torch.save({"model_state": model.state_dict(), "config": config}, out_dir / "checkpoint.pt")
    with open(out_dir / "metrics.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_psnr"])
        writer.writerows(metrics)
    dump_config(config, out_dir / "resolved_config.yaml")
    print(f"wrote run to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
