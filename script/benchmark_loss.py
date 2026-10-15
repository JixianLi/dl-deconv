"""Time a training step's forward pass, loss and backward pass for several losses.

  uv run python -m script.benchmark_loss config/deep_screen/mse.yaml config/loss_screen/mse.yaml \
      --losses mse tweedie beta_nll

For each config (its model, batch size and dataset) and each loss, runs warm-up steps,
then times steps on a fixed set of train batches already on the device, so data loading
is excluded. The device is synchronized around each section, so asynchronous GPU work is
counted in the section that queued it. The loss's own gradient is computed in the
backward pass, so a loss's full cost is the difference in total step time, reported
relative to the first loss listed. Run it alone on the GPU; anything sharing it inflates
the times. Prints a table and writes nothing.
"""

import argparse
import time

import numpy as np
import torch

from core.config import load_config
from core.losses import load_loss
from core.runtime import resolve_device, set_seed
from dataset.patch_dataset import PatchDataset
from model import build_model

NUM_BATCHES = 16
SECTIONS = ("forward", "loss", "backward + step", "total")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def fixed_batches(config, device):
    dataset = PatchDataset(config.data.out_dir, "train")
    rng = np.random.default_rng(config.seed)
    batches = []
    for _ in range(NUM_BATCHES):
        indices = rng.choice(len(dataset), size=config.train.batch_size, replace=False)
        observed, ideal = zip(*(dataset[index] for index in indices))
        batches.append((torch.stack(observed).to(device), torch.stack(ideal).to(device)))
    return batches


def time_steps(config, loss_name, batches, device, warmup_steps, timed_steps):
    """Median milliseconds per section over timed_steps."""
    set_seed(config.seed)
    loss_fn = load_loss(loss_name, config.data.out_dir)
    model = build_model(config.model, loss_fn.num_output_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    times = {section: [] for section in SECTIONS}
    for step in range(warmup_steps + timed_steps):
        observed, ideal = batches[step % len(batches)]
        optimizer.zero_grad()
        synchronize(device)
        start = time.perf_counter()
        raw_output = model(observed)
        synchronize(device)
        after_forward = time.perf_counter()
        loss = loss_fn(raw_output, ideal)
        synchronize(device)
        after_loss = time.perf_counter()
        loss.backward()
        optimizer.step()
        synchronize(device)
        end = time.perf_counter()
        if step >= warmup_steps:
            for section, seconds in zip(SECTIONS, (after_forward - start, after_loss - after_forward,
                                                   end - after_loss, end - start)):
                times[section].append(seconds * 1e3)
    return {section: float(np.median(values)) for section, values in times.items()}


def main(config_files, losses, warmup_steps, timed_steps):
    for config_file in config_files:
        config = load_config(config_file)
        device = resolve_device(config.train.device)
        batches = fixed_batches(config, device)
        name = device.type if device.type != "cuda" else torch.cuda.get_device_name(device)
        print(f"\n{config.model.name} (batch {config.train.batch_size}) on {name}, "
              f"median ms per step over {timed_steps} steps")
        print(f"{'loss':12s}" + "".join(f"{section:>17s}" for section in SECTIONS)
              + f"{'loss share':>12s}{'vs ' + losses[0]:>12s}")
        reference_total = None
        for loss_name in losses:
            medians = time_steps(config, loss_name, batches, device, warmup_steps, timed_steps)
            reference_total = reference_total or medians["total"]
            print(f"{loss_name:12s}" + "".join(f"{medians[section]:17.3f}" for section in SECTIONS)
                  + f"{medians['loss'] / medians['total']:12.1%}{medians['total'] / reference_total:12.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", help="configs whose model, batch size and data to use")
    parser.add_argument("--losses", nargs="+", default=["mse", "tweedie", "beta_nll"])
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    arguments = parser.parse_args()
    main(arguments.configs, arguments.losses, arguments.warmup_steps, arguments.steps)
