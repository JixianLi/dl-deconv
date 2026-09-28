"""Train the deconvolution baseline from a config file.

  uv run python -m script.train config/baseline.yaml

Writes into train.out_dir: checkpoint.pt (weights + config), metrics.csv (one row
per validation, every train.val_interval iterations), and resolved_config.yaml.
Assumes the dataset already exists at data.out_dir (run dataset.gen_data first).
"""

import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

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
    val_set = PatchDataset(config.data.out_dir, "val", stride=config.data.patch_size)
    # One pass over this sampler is the whole run. Without replacement, windows repeat only
    # once iterations * batch_size exceeds the train set (it then draws fresh permutations).
    train_sampler = RandomSampler(train_set,
                                  num_samples=train_config.iterations * train_config.batch_size)
    train_loader = DataLoader(train_set, batch_size=train_config.batch_size, sampler=train_sampler,
                              num_workers=train_config.num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=train_config.batch_size, shuffle=False,
                            num_workers=train_config.num_workers)

    model = build_model(config.model).to(device)
    loss_fn = LOSSES[train_config.loss]()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)

    print(f"device={device}  train={len(train_set)}  val={len(val_set)}  "
          f"iterations={train_config.iterations}")
    metrics = []
    running = 0.0
    seen = 0
    model.train()
    for iteration, (observed, ideal) in enumerate(train_loader, start=1):
        observed, ideal = observed.to(device), ideal.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(observed), ideal)
        loss.backward()
        optimizer.step()
        running += loss.item() * observed.size(0)
        seen += observed.size(0)
        if iteration % train_config.val_interval == 0 or iteration == train_config.iterations:
            train_loss = running / seen
            val_loss, val_psnr = evaluate(model, val_loader, loss_fn, device)
            model.train()
            metrics.append((iteration, train_loss, val_loss, val_psnr))
            print(f"iteration {iteration:7d}  train_loss={train_loss:.5f}  "
                  f"val_loss={val_loss:.5f}  val_psnr={val_psnr:.3f}")
            running = 0.0
            seen = 0

    torch.save({"model_state": model.state_dict(), "config": config}, out_dir / "checkpoint.pt")
    with open(out_dir / "metrics.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "train_loss", "val_loss", "val_psnr"])
        writer.writerows(metrics)
    dump_config(config, out_dir / "resolved_config.yaml")
    print(f"wrote run to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
