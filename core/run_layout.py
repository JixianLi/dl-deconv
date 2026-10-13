"""Layout of an experiment folder, the train.out_dir of a run.

  <run_dir>/
    config.yaml                the resolved config
    summary.txt                metadata table (write_summary)
    checkpoint/checkpoint.pt   weights + config
    training/metrics.csv       one row per validation interval
    loss_curve/loss_log.png    train and val loss, log scale
    loss_curve/loss_linear.png the same, linear scale (losses such as an NLL go negative)
    evaluation/                eval and photometry on the run's own dataset
    evaluation/export/         physical-flux FITS, only if script.export_fits ran
    evaluation-<dataset>/      the same for another dataset, via --data
    log/                       the SLURM log, moved here by script/slurm/train.sbatch

Every Python script that reads or writes a run goes through these functions, so a
rename happens here once; train.sbatch hard-codes log/.
"""

from pathlib import Path


def config_path(run_dir):
    return Path(run_dir) / "config.yaml"


def summary_path(run_dir):
    return Path(run_dir) / "summary.txt"


def checkpoint_path(run_dir):
    return Path(run_dir) / "checkpoint" / "checkpoint.pt"


def metrics_path(run_dir):
    return Path(run_dir) / "training" / "metrics.csv"


def loss_curve_path(run_dir, scale):
    return Path(run_dir) / "loss_curve" / f"loss_{scale}.png"


def evaluation_dir(run_dir, other_dataset=None):
    """evaluation/ for the run's own dataset, evaluation-<other_dataset>/ for another one."""
    return Path(run_dir) / ("evaluation" if other_dataset is None else f"evaluation-{other_dataset}")


def export_dir(run_dir, other_dataset=None):
    return evaluation_dir(run_dir, other_dataset) / "export"


def other_dataset_name(run_data_dir, evaluated_data_dir):
    """None when evaluating on the run's own dataset, else the evaluated dataset's dir name."""
    if Path(evaluated_data_dir).resolve() == Path(run_data_dir).resolve():
        return None
    return Path(evaluated_data_dir).name


def write_summary(path, fields):
    """Write (label, value) pairs as two aligned plain-text columns."""
    width = max(len(label) for label, _ in fields) + 2
    lines = [f"{label:<{width}}{value}" for label, value in fields]
    Path(path).write_text("\n".join(lines) + "\n")
