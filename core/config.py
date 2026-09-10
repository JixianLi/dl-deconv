"""Config schema and (de)serialization for the SISR baseline.

A run is fully specified by one YAML file with three sections (`data`, `model`,
`train`) plus a top-level `seed`. Defaults live here in the dataclasses, so the
*resolved* config (defaults filled in) is what we dump alongside every run's
outputs. Reproducing a run means reproducing these parameters, not bytes.
"""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    observed_fits: str  # input: the trimmed, PSF-blurred observed image
    ideal_fits: str  # target: the untrimmed ideal point-source intensity field
    out_dir: str
    patch_size: int = 64  # observed and ideal are the same size; the model maps one to the other
    stride: int = 64  # patch grid step over the source image (>= patch_size => no overlap)
    observed_normalize: str = "asinh"  # asinh | log | linear — applied to the observed image
    ideal_normalize: str = "asinh"  # normalization for the sparse ideal field
    asinh_softening: float = 3.0  # asinh transition width in units of the (sigma-clipped) noise std
    split_block_size: int = 505  # side length (px) of the spatial blocks assigned wholesale to train/val
    val_fraction: float = 0.2


@dataclass
class ModelConfig:
    name: str = "espcn"
    channels: int = 64


@dataclass
class TrainConfig:
    out_dir: str
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1.0e-3
    loss: str = "l1"  # l1 | mse
    num_workers: int = 2
    device: str = "auto"  # auto | cpu | cuda | mps


@dataclass
class Config:
    seed: int
    data: DataConfig
    model: ModelConfig
    train: TrainConfig


def _build(cls, values):
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**values)


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text())
    return Config(
        seed=raw["seed"],
        data=_build(DataConfig, raw.get("data", {})),
        model=_build(ModelConfig, raw.get("model", {})),
        train=_build(TrainConfig, raw.get("train", {})),
    )


def dump_config(config, path):
    Path(path).write_text(yaml.safe_dump(asdict(config), sort_keys=False))
