#!/bin/bash
# Submit one job per group of configs in a folder: <name>.yaml plus its seed repeats
# <name>_seed<N>.yaml share a job and train in parallel on one GPU. --runs-per-job N
# splits a group into jobs of at most N runs, for groups too big for one GPU. Extra
# arguments go to sbatch, e.g. a partition and time limit for a bigger model:
#
#   bash script/slurm/submit_screen.sh config/loss_screen
#   bash script/slurm/submit_screen.sh config/deep_screen -p gh -t 06:00:00
#   bash script/slurm/submit_screen.sh config/stability_screen --runs-per-job 5 -p gh -t 03:00:00
#
# Run from the repo root. Resubmit a single group with
#   sbatch -J <folder>-<name> script/slurm/train.sbatch config/<folder>/<name>*.yaml

set -euo pipefail

config_dir="${1:-}"
[[ -n "$config_dir" && -d "$config_dir" ]] || {
    echo "usage: $0 <config dir> [--runs-per-job N] [sbatch args...]" >&2; exit 1; }
shift
runs_per_job=0
if [[ "${1:-}" == "--runs-per-job" ]]; then
    runs_per_job="${2:-}"
    [[ "$runs_per_job" =~ ^[1-9][0-9]*$ ]] || { echo "error: --runs-per-job needs a positive integer" >&2; exit 1; }
    shift 2
fi
[[ -f pyproject.toml && -d script ]] || { echo "error: run from the repo root" >&2; exit 1; }

screen=$(basename "$config_dir")
for base in $(ls "$config_dir"/*.yaml | xargs -n 1 basename | sed -E 's/(_seed[0-9]+)?\.yaml$//' | sort -u); do
    group=("$config_dir/$base.yaml")
    for repeat in "$config_dir/${base}"_seed*.yaml; do
        [[ -f "$repeat" ]] && group+=("$repeat")
    done
    if (( runs_per_job == 0 || ${#group[@]} <= runs_per_job )); then
        sbatch -J "$screen-$base" "$@" script/slurm/train.sbatch "${group[@]}"
        continue
    fi
    for (( start = 0, part = 1; start < ${#group[@]}; start += runs_per_job, part++ )); do
        sbatch -J "$screen-$base-$part" "$@" script/slurm/train.sbatch "${group[@]:start:runs_per_job}"
    done
done
