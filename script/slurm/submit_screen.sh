#!/bin/bash
# Submit one job per group of configs in a folder: <name>.yaml plus its seed repeats
# <name>_seed<N>.yaml share a job and train in parallel on one GPU. Extra arguments go to
# sbatch, e.g. a partition and time limit for a bigger model:
#
#   bash script/slurm/submit_screen.sh config/loss_screen
#   bash script/slurm/submit_screen.sh config/deep_screen -p gh -t 06:00:00
#
# Run from the repo root. Resubmit a single group with
#   sbatch -J <folder>-<name> script/slurm/train.sbatch config/<folder>/<name>*.yaml

set -euo pipefail

config_dir="${1:-}"
[[ -n "$config_dir" && -d "$config_dir" ]] || { echo "usage: $0 <config dir> [sbatch args...]" >&2; exit 1; }
shift
[[ -f pyproject.toml && -d script ]] || { echo "error: run from the repo root" >&2; exit 1; }

screen=$(basename "$config_dir")
for base in $(ls "$config_dir"/*.yaml | xargs -n 1 basename | sed -E 's/(_seed[0-9]+)?\.yaml$//' | sort -u); do
    group=("$config_dir/$base.yaml")
    for repeat in "$config_dir/${base}"_seed*.yaml; do
        [[ -f "$repeat" ]] && group+=("$repeat")
    done
    sbatch -J "$screen-$base" "$@" script/slurm/train.sbatch "${group[@]}"
done
