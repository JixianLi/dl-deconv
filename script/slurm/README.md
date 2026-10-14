# Running on TACC Vista

`train.sbatch` trains one or more configs on a Grace Hopper node (1 GPU per node, whole
node allocated), then runs `script.eval` and `script.photometry` on each run. Several
configs train in parallel on the one GPU. Each run folder follows the layout in
`core/run_layout.py`, and each run's output is written to its `log/` folder as it goes.
Training saves at every validation, so a run cut off by the time limit keeps its metrics,
latest checkpoint and summary; run eval and photometry on it afterwards.

`submit_screen.sh <config dir> [sbatch args]` submits one job per config group: a base
config and its `_seed<N>` repeats share a job.

    bash script/slurm/submit_screen.sh config/loss_screen
    bash script/slurm/submit_screen.sh config/deep_screen -p gh -t 06:00:00

The default is 2 hours on `gh` or `gh-dev`, enough for ESPCN. `gh-dev` caps jobs at
2 hours, so longer jobs must name `-p gh`.

Setup, done once by hand (the scripts set none of it up):

- the repo is checked out on Vista and uv is on PATH;
- `storage/` in the repo resolves to the data dir, e.g. `ln -s $SCRATCH/dl-deconv storage`,
  laid out like local `storage/` (`patches/baseline/`, `M31/`), so config paths match;
- if your login has several projects, `export SBATCH_ACCOUNT=<project>`.

Before the full screen, try one short job on `gh-dev` with a smoke config (a copy with
`iterations: 60`, `val_interval: 30` and an out_dir of its own):

    sbatch -p gh-dev -t 00:30:00 -J smoke script/slurm/train.sbatch <smoke config>

Each job checks, in order, that it is in the repo root, `$SCRATCH` exists, `storage/`
resolves, `uv sync --locked` succeeds (a no-op when `.venv` matches `uv.lock`), the
configs' data files exist, and torch sees a GPU, and stops with a message at the first
failure.

**`$SCRATCH` is purged of files not accessed for 10 days.** Copy each screen's
`storage/runs/<screen>/` back to local storage as soon as it finishes.
