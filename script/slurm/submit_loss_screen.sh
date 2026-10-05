#!/bin/bash
# Submit one job per loss-screen config, so each run fails and resubmits on its own.
#
#   bash script/slurm/submit_loss_screen.sh
#
# Run from the repo root. Resubmit a single run with
#   sbatch -J loss-<name> script/slurm/train.sbatch config/loss_screen/<name>.yaml

set -euo pipefail

[[ -d config/loss_screen ]] || { echo "error: run from the repo root" >&2; exit 1; }
for config in config/loss_screen/*.yaml; do
    name=$(basename "$config" .yaml)
    sbatch -J "loss-$name" script/slurm/train.sbatch "$config"
done
