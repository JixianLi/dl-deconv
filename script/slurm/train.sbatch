#!/bin/bash
# Train, evaluate and score one or more configs on a TACC Vista Grace Hopper node (1 GPU).
#
#   sbatch -J <name> script/slurm/train.sbatch <config.yaml> [<config.yaml> ...]
#
# Several configs run in parallel on the node's one GPU, each as train -> eval ->
# photometry; a small model leaves most of the GPU idle, so sharing it raises throughput,
# and on a time limit every run stops at about the same iteration. Each run's output goes
# to <run_dir>/log/run-<job id>.out as it runs, so it survives a time limit (train saves
# its checkpoint, metrics and summary at every validation). The job's own log, with the
# checks and each run's exit status, is written to <job name>-<job id>.out in the repo
# root and moved when the job ends: into <run_dir>/log/ for one config, into a log/ folder
# next to the run dirs for several. A job that fails before its configs are read leaves
# it in the repo root. The job fails if any run fails.
#
# Submit from the repo root. If your login has several projects, export
# SBATCH_ACCOUNT=<project> first (sbatch reads it as -A). Sets up no environment: it
# expects uv on PATH and storage/ to resolve to the data dir (e.g. a symlink to
# $SCRATCH/dl-deconv), and fails fast otherwise.
#
# The defaults fit the ESPCN screen, whose runs finish in about 8 minutes: 2 hours on
# whichever of gh and gh-dev starts the job first. For a bigger model, override on the
# command line, e.g. sbatch -p gh -t 06:00:00 ... (gh-dev caps jobs at 2 hours).
#SBATCH -p gh,gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o %x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jixianli@tacc.utexas.edu

set -euo pipefail

fail() {
    echo "error: $*" >&2
    exit 1
}

configs=("$@")
(( ${#configs[@]} > 0 )) || fail "usage: sbatch script/slurm/train.sbatch <config.yaml> [<config.yaml> ...]"
[[ -f pyproject.toml && -d script ]] || fail "not in the repo root: $(pwd)"
[[ -n "${SCRATCH:-}" && -d "$SCRATCH" ]] || fail "\$SCRATCH is unset or not a directory"
[[ -d storage ]] || fail "storage/ is missing or does not resolve to a directory"
for config in "${configs[@]}"; do
    [[ -f "$config" ]] || fail "config not found: $config (submit from the repo root)"
done

uv sync --locked || fail "uv sync --locked failed (is uv.lock in sync with pyproject.toml?)"

run_dirs=()
for config in "${configs[@]}"; do
    run_dir=$(uv run --no-sync python - "$config" <<'EOF'
import sys
from pathlib import Path

from core.config import load_config

config = load_config(sys.argv[1])
data_dir = Path(config.data.out_dir)
needed = [data_dir / name for name in (
    "observed.npy", "ideal.npy", "val_mask.npy", "norm.json", "noise.json", "crlb_flux_sigma.npy",
    "train_corners.npy", "val_corners.npy", "resolved_config.yaml")]
needed.append(Path(config.data.ideal_fits))
missing = [str(path) for path in needed if not path.exists()]
if missing:
    sys.exit("error: missing data files:\n  " + "\n  ".join(missing))
print(config.train.out_dir)
EOF
    ) || fail "data check failed for $config"
    run_dirs+=("$run_dir")
done

if (( ${#run_dirs[@]} == 1 )); then
    job_log_dir="${run_dirs[0]}/log"
else
    job_log_dir="$(dirname "${run_dirs[0]}")/log"
fi
# Renaming keeps SLURM's open handle on the log valid, so lines written after this script
# exits still land in the moved file, as long as the repo and storage/ share a filesystem.
# At a time limit SLURM sends TERM first; turning it into a normal exit runs the EXIT trap.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    slurm_log=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *StdOut=//p')
    trap 'mkdir -p "$job_log_dir" && mv "$slurm_log" "$job_log_dir/"' EXIT
    trap 'echo "received TERM (time limit?); runs keep what they saved at their last validation"; exit 143' TERM
fi

uv run --no-sync python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || fail "torch sees no CUDA device; refusing to train on CPU"

run_pipeline() {
    local config=$1 run_dir=$2
    uv run --no-sync python -m script.train "$config" \
        && uv run --no-sync python -m script.eval "$run_dir" \
        && uv run --no-sync python -m script.photometry "$run_dir"
}

job_id="${SLURM_JOB_ID:-local}"
echo "host=$(hostname) job=$job_id runs=${#configs[@]}"
pids=()
for index in "${!configs[@]}"; do
    run_dir="${run_dirs[$index]}"
    mkdir -p "$run_dir/log"
    run_pipeline "${configs[$index]}" "$run_dir" > "$run_dir/log/run-$job_id.out" 2>&1 &
    pids+=($!)
    echo "started config=${configs[$index]} run_dir=$run_dir pid=$!"
done

failed=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "finished run_dir=${run_dirs[$index]}"
    else
        echo "FAILED (exit $?) run_dir=${run_dirs[$index]}, see ${run_dirs[$index]}/log/run-$job_id.out"
        failed=1
    fi
done
exit "$failed"
