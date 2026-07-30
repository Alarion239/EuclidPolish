#!/usr/bin/env python3
"""Generate unbiased normal-field dirty/lens pairs for lens isolation."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import secrets
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

# Keep generation workers one-CPU each, just as ``scripts/run_pipeline.py``
# does before importing NumPy. An explicit cluster override still wins.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.experiments.lens_isolation.config import (
    SCHEMA_VERSION,
    DatasetConfig,
    ExperimentPaths,
    assert_safe_output,
)


@dataclass(frozen=True)
class LensWorkerRuntime:
    """Pickle-safe process-local construction inputs for one shard worker."""

    records_dir: str
    image_size: int
    psf_dir: str
    tng_dir: str
    galaxy_density_arcmin2: float
    lens_density_arcmin2: float
    config_fingerprint: str


_WORKER_GENERATOR: Any | None = None
_WORKER_RECORDS_DIR = ""
_WORKER_CONFIG_FINGERPRINT = ""
_GENERATION_STATE = "generation_state.json"


def _init_lens_worker(runtime: LensWorkerRuntime) -> None:
    """Build one complete normal-field pair generator in each process."""
    global _WORKER_GENERATOR, _WORKER_RECORDS_DIR, _WORKER_CONFIG_FINGERPRINT

    from euclid_polish.experiments.lens_isolation.generation import LensCaptureAdapter
    from euclid_polish.psf.psf_library import load_all_band_psf_sets
    from euclid_polish.sky.generation.cosmos_tng_prior import CosmosTngPrior
    from euclid_polish.sky.generation.sky_simulator import SkySimulator, SkySimulatorConfig
    from euclid_polish.sky.observation.observation_simulator import (
        ObservationSimulator,
        ObservationSimulatorConfig,
    )

    sky = SkySimulator(
        CosmosTngPrior(Config.COSMOS_TNG_PRIOR_PATH),
        SkySimulatorConfig(
            image_size=runtime.image_size,
            pixel_scale=Config.DEFAULT_PIXEL_SCALE,
            galaxy_density_arcmin2=runtime.galaxy_density_arcmin2,
            tng_galaxy_dir=runtime.tng_dir,
            lens_density_arcmin2=runtime.lens_density_arcmin2,
        ),
    )
    observation = ObservationSimulator(
        psf_sets_by_band=load_all_band_psf_sets(
            psf_dir=runtime.psf_dir,
            target_pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        ),
        config=ObservationSimulatorConfig(),
    )
    _WORKER_GENERATOR = LensCaptureAdapter(sky, observation)
    _WORKER_RECORDS_DIR = runtime.records_dir
    _WORKER_CONFIG_FINGERPRINT = runtime.config_fingerprint


def _generate_lens_shard(shard):
    """Process-pool entry point; no simulator state crosses worker boundaries."""
    from euclid_polish.experiments.lens_isolation.records import write_shard

    if _WORKER_GENERATOR is None or not _WORKER_RECORDS_DIR or not _WORKER_CONFIG_FINGERPRINT:
        raise RuntimeError("lens-isolation worker was not initialized")
    return write_shard(
        _WORKER_GENERATOR,
        _WORKER_RECORDS_DIR,
        shard,
        config_fingerprint=_WORKER_CONFIG_FINGERPRINT,
    )


def _generate_split_parallel(
    runtime: LensWorkerRuntime,
    *,
    subset: str,
    count: int,
    seed: int,
    workers: int,
    config_fingerprint: str,
    force: bool,
):
    """Follow the normal Sky process-shard lifecycle for one paired split."""
    from euclid_polish.experiments.lens_isolation.records import (
        SplitSummary,
        clear_split,
        completed_shards,
        make_shards,
        merge_shards,
        missing_shards,
        validate_split,
    )
    from euclid_polish.observability import Reporter

    if not force and validate_split(
        runtime.records_dir,
        subset,
        count,
        config_fingerprint=config_fingerprint,
    ):
        return SplitSummary(subset=subset, count=count, reused=True)
    if not make_shards(subset, count=count, workers=workers, seed=seed):
        raise ValueError("lens-isolation splits must contain at least one field")
    if force:
        clear_split(runtime.records_dir, subset)
        summaries = []
    else:
        summaries = completed_shards(
            runtime.records_dir,
            subset,
            total_count=count,
            split_seed=seed,
            config_fingerprint=config_fingerprint,
        )
    pending = missing_shards(
        subset,
        count=count,
        workers=workers,
        seed=seed,
        completed=summaries,
    )
    active_workers = min(workers, len(pending))
    reporter = Reporter.from_env()
    if pending:
        remaining = sum(shard.count for shard in pending)
        reporter.set_stage(f"lens isolation: generate {subset} (×{active_workers})")
        reporter.set_parallel(remaining, active_workers, label=subset)
        with ProcessPoolExecutor(
            max_workers=active_workers,
            initializer=_init_lens_worker,
            initargs=(runtime,),
        ) as pool:
            summaries.extend(pool.map(_generate_lens_shard, pending))
    return merge_shards(
        runtime.records_dir,
        subset,
        summaries,
        config_fingerprint=config_fingerprint,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=ExperimentPaths().records)
    parser.add_argument("--ntrain", type=int, default=6400)
    parser.add_argument("--nvalid", type=int, default=100)
    parser.add_argument("--ntest", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=510)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--psf-dir", default=None)
    parser.add_argument("--tng-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _source_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", _PROJECT_ROOT, "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _published_config_mismatch(records_dir: str, subset: str, fingerprint: str) -> bool:
    path = os.path.join(records_dir, f"split_{subset}.json")
    try:
        with open(path, encoding="utf-8") as handle:
            metadata = json.load(handle)
    except FileNotFoundError:
        return False
    except (OSError, ValueError):
        return True
    return (
        metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("config_fingerprint") != fingerprint
    )


def _generation_state_path(records_dir: str) -> str:
    return os.path.join(records_dir, _GENERATION_STATE)


def _read_generation_state(records_dir: str) -> dict[str, Any] | None:
    try:
        with open(_generation_state_path(records_dir), encoding="utf-8") as handle:
            state = json.load(handle)
    except (FileNotFoundError, OSError, ValueError):
        return None
    return state if isinstance(state, dict) else None


def _write_generation_state(
    records_dir: str,
    *,
    config_fingerprint: str,
    counts: dict[str, int],
    master_seed: int,
) -> None:
    os.makedirs(records_dir, exist_ok=True)
    path = _generation_state_path(records_dir)
    fd, temporary = tempfile.mkstemp(prefix=f"{_GENERATION_STATE}.tmp-", dir=records_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": SCHEMA_VERSION,
                    "config_fingerprint": config_fingerprint,
                    "counts": counts,
                    "master_seed": int(master_seed),
                },
                handle,
                indent=2,
                sort_keys=True,
            )
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def _published_master_seed(records_dir: str, fingerprint: str) -> int | None:
    for offset, subset in enumerate(("train", "validate", "test")):
        path = os.path.join(records_dir, f"split_{subset}.json")
        try:
            with open(path, encoding="utf-8") as handle:
                metadata = json.load(handle)
            if (
                metadata.get("schema_version") == SCHEMA_VERSION
                and metadata.get("config_fingerprint") == fingerprint
            ):
                return int(metadata["seed"]) - offset
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
            continue
    return None


def _master_seed_for_run(
    records_dir: str,
    *,
    config_fingerprint: str,
    counts: dict[str, int],
    requested_seed: int,
) -> int:
    """Persist an entropy-derived seed before workers can publish a shard."""
    state = _read_generation_state(records_dir)
    if state is not None:
        try:
            state_seed = int(state["master_seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid partial lens-isolation generation state; rerun with --force") from error
        if (
            state.get("schema_version") != SCHEMA_VERSION
            or state.get("config_fingerprint") != config_fingerprint
            or state.get("counts") != counts
        ):
            raise ValueError(
                "incompatible partial lens-isolation artifacts found; rerun generation with --force"
            )
        return state_seed
    master_seed = _published_master_seed(records_dir, config_fingerprint)
    if master_seed is None:
        master_seed = requested_seed if requested_seed >= 0 else secrets.randbits(63)
    _write_generation_state(
        records_dir,
        config_fingerprint=config_fingerprint,
        counts=counts,
        master_seed=master_seed,
    )
    return master_seed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be >= 1")
    config = DatasetConfig(
        n_train=args.ntrain,
        n_validate=args.nvalid,
        n_test=args.ntest,
        image_size=args.image_size,
        seed=args.seed,
    )
    out_dir = assert_safe_output(args.out_dir)
    counts = {"train": config.n_train, "validate": config.n_validate, "test": config.n_test}
    source_commit = _source_commit()
    runtime_config = {
        "tng_dir": os.path.abspath(args.tng_dir or Config.TNG_SKIRT_DIR),
        "psf_dir": os.path.abspath(args.psf_dir or Config.EUCLID_PSF_DIR),
        "source_commit": source_commit,
    }
    config_fingerprint = config.fingerprint(extra=runtime_config)
    plan = {
        "experiment": "lens_isolation",
        "schema_version": SCHEMA_VERSION,
        "out_dir": os.path.abspath(out_dir),
        "counts": counts,
        "image_size": config.image_size,
        "population": {
            "galaxy_density_arcmin2": config.galaxy_density_arcmin2,
            "lens_density_arcmin2": config.lens_density_arcmin2,
        },
        "config_fingerprint": config_fingerprint,
        "seed": config.seed,
        "workers": args.workers,
        "force": bool(args.force),
    }
    if args.dry_run:
        print(json.dumps(plan, sort_keys=True))
        return 0

    from euclid_polish.experiments.lens_isolation.records import clear_split, write_dataset_metadata

    for subset in counts:
        if not args.force and _published_config_mismatch(out_dir, subset, config_fingerprint):
            raise ValueError(
                "incompatible lens-isolation artifacts found; rerun generation with --force "
                "to replace only data/experiments/lens_isolation records"
            )
    if args.force:
        for subset in counts:
            clear_split(out_dir, subset)
        for name in ("dataset.json", _GENERATION_STATE):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(os.path.join(out_dir, name))
    master_seed = _master_seed_for_run(
        out_dir,
        config_fingerprint=config_fingerprint,
        counts=counts,
        requested_seed=config.seed,
    )

    # Catalog=None enforces pure-TNG fields. Each process constructs its own
    # simulator and capture adapter, matching normal Sky process sharding.
    runtime = LensWorkerRuntime(
        records_dir=out_dir,
        image_size=config.image_size,
        psf_dir=args.psf_dir or Config.EUCLID_PSF_DIR,
        tng_dir=args.tng_dir or Config.TNG_SKIRT_DIR,
        galaxy_density_arcmin2=config.galaxy_density_arcmin2,
        lens_density_arcmin2=config.lens_density_arcmin2,
        config_fingerprint=config_fingerprint,
    )
    summaries = {}
    for offset, (subset, count) in enumerate(counts.items()):
        summaries[subset] = _generate_split_parallel(
            runtime,
            subset=subset,
            count=count,
            seed=master_seed + offset,
            workers=args.workers,
            config_fingerprint=config_fingerprint,
            force=args.force,
        )
        status = "reused" if summaries[subset].reused else "generated"
        print(f"{subset}: {count} normal fields ({status})")
    metadata_path = write_dataset_metadata(
        out_dir,
        config={
            **config.scientific_config(),
            **runtime_config,
        },
        master_seed=master_seed,
        split_summaries=summaries,
        source_commit=source_commit,
    )
    print(json.dumps({**plan, "seed": master_seed, "dataset_json": metadata_path}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
