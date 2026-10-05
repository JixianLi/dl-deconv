"""Train the deconvolution baseline from a config file.

  uv run python -m script.train config/baseline.yaml

Writes into train.out_dir: checkpoint.pt (weights + config), metrics.csv (one row
per validation, every train.val_interval iterations), and resolved_config.yaml.
Assumes the dataset already exists at data.out_dir (run dataset.gen_data, then
dataset.estimate_psf for noise.json).

train_loss and val_loss are in the run's own loss, so they can't be compared across
losses. The other val columns are computed on the 1-channel normalized-ideal prediction
of the val patches and can:

  val_psnr, val_l1_normalized, val_mse_normalized    in normalized space
  val_faint_flux_ratio        predicted / true flux on source pixels fainter than the
                              median val source (photometry's faint_source_flux_ratio)
  val_spurious_flux_fraction  positive predicted flux on true-zero pixels / true flux
  val_frac_within_0.5mag_snr1_10
                              fraction of sources with true flux in [1, 10] x the median
                              Cramér–Rao sigma that are predicted within 0.5 mag; about
                              S/N 1-10, since the sigma map is near uniform
  train_seconds               wall time of the interval's training steps, data loading
                              included, validation excluded (the first one includes
                              DataLoader worker startup)
  elapsed_seconds             wall time since training started, validation included
"""

import csv
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

from core.config import dump_config, load_config
from core.losses import flux_from_normalized, load_ideal_norm, load_loss, load_median_flux_sigma
from core.runtime import psnr, resolve_device, set_seed
from dataset.patch_dataset import PatchDataset
from model import build_model


RECOVERED_WITHIN_MAG = 0.5
SNR_WINDOW = (1, 10)  # true flux range of val_frac_within_0.5mag_snr1_10, in median sigmas


def median_source_flux(loader, ideal_norm):
    fluxes = [flux_from_normalized(ideal_norm, ideal) for _, ideal in loader]
    fluxes = torch.cat([flux[flux > 0] for flux in fluxes])
    return float(fluxes.median())


def evaluate(model, loader, loss_fn, ideal_norm, faint_flux_limit, median_flux_sigma, device):
    model.eval()
    sums = dict.fromkeys(("loss", "psnr", "l1", "mse", "faint_predicted", "faint_true",
                          "spurious", "true_total", "window_recovered", "window_sources"), 0.0)
    count = 0
    ratio_low, ratio_high = 10 ** (-0.4 * RECOVERED_WITHIN_MAG), 10 ** (0.4 * RECOVERED_WITHIN_MAG)
    with torch.no_grad():
        for observed, ideal in loader:
            observed, ideal = observed.to(device), ideal.to(device)
            raw_output = model(observed)
            prediction = loss_fn.to_normalized_ideal(raw_output)
            batch = observed.size(0)
            sums["loss"] += loss_fn(raw_output, ideal).item() * batch
            sums["psnr"] += psnr(prediction, ideal) * batch
            sums["l1"] += torch.mean(torch.abs(prediction - ideal)).item() * batch
            sums["mse"] += torch.mean((prediction - ideal) ** 2).item() * batch
            count += batch

            predicted_flux = flux_from_normalized(ideal_norm, prediction)
            true_flux = flux_from_normalized(ideal_norm, ideal)
            sources = true_flux > 0
            faint = sources & (true_flux < faint_flux_limit)
            sums["faint_predicted"] += predicted_flux[faint].sum().item()
            sums["faint_true"] += true_flux[faint].sum().item()
            sums["spurious"] += predicted_flux[~sources].clamp(min=0.0).sum().item()
            sums["true_total"] += true_flux.sum().item()
            in_window = (sources & (true_flux >= SNR_WINDOW[0] * median_flux_sigma)
                         & (true_flux <= SNR_WINDOW[1] * median_flux_sigma))
            ratio = predicted_flux[in_window] / true_flux[in_window]
            sums["window_recovered"] += ((ratio > ratio_low) & (ratio < ratio_high)).sum().item()
            sums["window_sources"] += in_window.sum().item()
    return {
        "val_loss": sums["loss"] / count,
        "val_psnr": sums["psnr"] / count,
        "val_l1_normalized": sums["l1"] / count,
        "val_mse_normalized": sums["mse"] / count,
        "val_faint_flux_ratio": sums["faint_predicted"] / sums["faint_true"],
        "val_spurious_flux_fraction": sums["spurious"] / sums["true_total"],
        f"val_frac_within_{RECOVERED_WITHIN_MAG}mag_snr{SNR_WINDOW[0]}_{SNR_WINDOW[1]}":
            sums["window_recovered"] / sums["window_sources"],
    }


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

    loss_fn = load_loss(train_config.loss, config.data.out_dir)
    model = build_model(config.model, loss_fn.num_output_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)
    ideal_norm = load_ideal_norm(config.data.out_dir)
    median_flux_sigma = load_median_flux_sigma(config.data.out_dir)
    faint_flux_limit = median_source_flux(val_loader, ideal_norm)

    print(f"device={device}  train={len(train_set)}  val={len(val_set)}  "
          f"iterations={train_config.iterations}  loss={train_config.loss}")
    metrics = []
    running = 0.0
    seen = 0
    model.train()
    start_time = time.perf_counter()
    interval_start_time = start_time
    for iteration, (observed, ideal) in enumerate(train_loader, start=1):
        observed, ideal = observed.to(device), ideal.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(observed), ideal)
        loss.backward()
        optimizer.step()
        running += loss.item() * observed.size(0)
        seen += observed.size(0)
        if iteration % train_config.val_interval == 0 or iteration == train_config.iterations:
            train_seconds = time.perf_counter() - interval_start_time
            train_loss = running / seen
            val_metrics = evaluate(model, val_loader, loss_fn, ideal_norm, faint_flux_limit,
                                   median_flux_sigma, device)
            model.train()
            metrics.append({"iteration": iteration, "train_loss": train_loss, **val_metrics,
                            "train_seconds": train_seconds,
                            "elapsed_seconds": time.perf_counter() - start_time})
            print(f"iteration {iteration:7d}  train_loss={train_loss:.5g}  "
                  f"val_loss={val_metrics['val_loss']:.5g}  val_psnr={val_metrics['val_psnr']:.3f}  "
                  f"faint_ratio={val_metrics['val_faint_flux_ratio']:.3f}  "
                  f"it/s={seen / train_config.batch_size / train_seconds:.1f}")
            running = 0.0
            seen = 0
            interval_start_time = time.perf_counter()

    torch.save({"model_state": model.state_dict(), "config": config}, out_dir / "checkpoint.pt")
    with open(out_dir / "metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    dump_config(config, out_dir / "resolved_config.yaml")
    print(f"wrote run to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
