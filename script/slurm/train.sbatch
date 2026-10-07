#!/bin/bash
# Train, evaluate and score one config on a TACC Vista Grace Hopper node (1 GPU).
#
#   sbatch -J <name> script/slurm/train.sbatch config/loss_screen/<name>.yaml
#
# Submit from the repo root; the log goes to <job name>-<job id>.out there. If your login
# has several projects, export SBATCH_ACCOUNT=<project> first (sbatch reads it as -A).
# Creates nothing and sets up no environment: it expects uv on PATH and storage/ to
# resolve to the data dir (e.g. a symlink to $SCRATCH/dl-deconv), and fails fast otherwise.
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o %x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jixianli@tacc.utexas.edu

set -euo pipefail

fail() {
    echo "error: $*" >&2
    exit 1
}

config="${1:-}"
[[ -n "$config" ]] || fail "usage: sbatch script/slurm/train.sbatch <config.yaml>"
[[ -f "$config" ]] || fail "config not found: $config (submit from the repo root)"
[[ -f pyproject.toml && -d script ]] || fail "not in the repo root: $(pwd)"
[[ -n "${SCRATCH:-}" && -d "$SCRATCH" ]] || fail "\$SCRATCH is unset or not a directory"
[[ -d storage ]] || fail "storage/ is missing or does not resolve to a directory"

uv sync --locked || fail "uv sync --locked failed (is uv.lock in sync with pyproject.toml?)"

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

uv run --no-sync python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || fail "torch sees no CUDA device; refusing to train on CPU"

echo "config=$config run_dir=$run_dir host=$(hostname) job=${SLURM_JOB_ID:-none}"
uv run --no-sync python -m script.train "$config"
uv run --no-sync python -m script.eval "$run_dir"
uv run --no-sync python -m script.photometry "$run_dir"
uv run --no-sync python -m script.plot_loss "$run_dir"
