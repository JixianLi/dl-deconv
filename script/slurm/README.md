# Running on TACC Vista

`train.sbatch` trains one config on a Grace Hopper node (`gh` queue, 1 GPU per node,
whole node allocated), then runs `script.eval`, `script.photometry` and
`script.plot_loss` on the run. `submit_loss_screen.sh` submits one job per config in
`config/loss_screen/`.

Setup, done once by hand (the scripts create nothing):

- the repo is checked out on Vista and uv is on PATH;
- `storage/` in the repo resolves to the data dir, e.g. `ln -s $SCRATCH/dl-deconv storage`,
  laid out like local `storage/` (`patches/baseline/`, `M31/`), so config paths match;
- if your login has several projects, `export SBATCH_ACCOUNT=<project>`.

Before the full screen, try one short job on `gh-dev` with a smoke config (a copy with
`iterations: 60`, `val_interval: 30` and an out_dir of its own):

    sbatch -p gh-dev -t 00:30:00 -J smoke script/slurm/train.sbatch <smoke config>

Each job checks, in order, that it is in the repo root, `$SCRATCH` exists, `storage/`
resolves, `uv sync --locked` succeeds (a no-op when `.venv` matches `uv.lock`), the
config's data files exist, and torch sees a GPU, and stops with a message at the first
failure.

**`$SCRATCH` is purged of files not accessed for 10 days.** Copy
`storage/runs/loss_screen/` back to local storage as soon as the screen finishes.
