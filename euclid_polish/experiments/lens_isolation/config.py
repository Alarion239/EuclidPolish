"""Configuration and filesystem safety for the lens-isolation experiment."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from euclid_polish.config import Config
from euclid_polish.ensemble_registry import default_ensemble_dir

EXPERIMENT_NAME = "lens_isolation"
SCHEMA_VERSION = 2
_MEMBER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class ExperimentPaths:
    """All artifacts owned by the experiment beneath one isolated root."""

    root: str = os.path.join(Config.DATA_DIR, "experiments", EXPERIMENT_NAME)

    @property
    def records(self) -> str:
        return os.path.join(self.root, "records")

    @property
    def ensemble(self) -> str:
        return os.path.join(self.root, "ensemble")

    @property
    def evaluation(self) -> str:
        return os.path.join(self.root, "evaluation")


@dataclass(frozen=True)
class DatasetConfig:
    """Scientific settings for ordinary pure-TNG lens-isolation fields."""

    n_train: int = 6400
    n_validate: int = 100
    n_test: int = 100
    image_size: int = 510
    seed: int = -1
    galaxy_density_arcmin2: float = 60.0
    lens_density_arcmin2: float = 10.0

    def __post_init__(self) -> None:
        counts = (self.n_train, self.n_validate, self.n_test)
        if any(int(n) < 0 for n in counts):
            raise ValueError("split counts must be non-negative")
        if int(self.image_size) <= 0 or int(self.image_size) % 2:
            raise ValueError("image_size must be a positive even integer")
        densities = (
            self.galaxy_density_arcmin2,
            self.lens_density_arcmin2,
        )
        if any(float(value) < 0.0 for value in densities):
            raise ValueError("population densities must be non-negative")
        if float(self.galaxy_density_arcmin2) != 60.0:
            raise ValueError("lens isolation requires galaxy_density_arcmin2=60")
        if float(self.lens_density_arcmin2) != 10.0:
            raise ValueError("lens isolation requires lens_density_arcmin2=10")

    def scientific_config(self) -> dict[str, object]:
        """Return the versioned generation inputs persisted with the dataset."""
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}

    def fingerprint(self, *, extra: Mapping[str, object] | None = None) -> str:
        """Return the stable identity of scientific and runtime generation inputs."""
        payload_data: dict[str, object] = dict(self.scientific_config())
        if extra:
            payload_data["runtime"] = dict(extra)
        payload = json.dumps(payload_data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TrainConfig:
    """Normal fixed-record training controls shared by experimental members."""

    sources: tuple[str, ...]
    steps: int = 50_000
    batch_size: int = 16
    evaluate_every: int = 500
    lr_peak: float = 1e-5
    lr_final: float = 1e-6
    lr_warmup_steps: int = 500
    loss_norm: str = "l1"
    noise_aug: float = 0.0
    bootstrap: float | None = None
    base_seed: int = 0

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("at least one source member is required")
        if any(not _MEMBER_RE.fullmatch(str(source)) for source in self.sources):
            raise ValueError("source member names must be simple path-free names")
        if int(self.steps) < 1:
            raise ValueError("steps must be >= 1")
        if int(self.batch_size) < 1 or int(self.evaluate_every) < 1:
            raise ValueError("batch_size and evaluate_every must be >= 1")
        if not (0 < float(self.lr_final) <= float(self.lr_peak)):
            raise ValueError("learning rates must satisfy 0 < final <= peak")
        if self.loss_norm not in {"l1", "l2", "l3", "mse", "berhu"}:
            raise ValueError("loss_norm must be one of l1, l2, l3, mse, or berhu")
        if float(self.noise_aug) < 0.0:
            raise ValueError("noise_aug must be non-negative")
        if self.bootstrap is not None and not (0.0 < float(self.bootstrap) <= 1.0):
            raise ValueError("bootstrap must be in (0, 1]")


def _contains(root: str, path: str) -> bool:
    root_real = os.path.realpath(root)
    path_real = os.path.realpath(path)
    try:
        return os.path.commonpath((root_real, path_real)) == root_real
    except ValueError:
        return False


def assert_safe_output(
    path: str,
    *,
    source: str | None = None,
    protected_roots: tuple[str, ...] | None = None,
) -> str:
    """Return ``path`` normalized, rejecting production/source overlap."""
    if not str(path).strip():
        raise ValueError("output path must not be empty")
    normalized = os.path.abspath(os.path.expanduser(str(path)))
    roots = protected_roots or (
        Config.RECORDS_DIR_V2,
        default_ensemble_dir(),
        Config.DEFAULT_CHECKPOINT_DIR,
    )
    for protected in roots:
        if protected and _contains(protected, normalized):
            raise ValueError(f"output is inside protected production path {protected!r}")
    if source:
        source_abs = os.path.abspath(os.path.expanduser(str(source)))
        if _contains(source_abs, normalized) or _contains(normalized, source_abs):
            raise ValueError("output path overlaps its read-only source checkpoint")
    return normalized
