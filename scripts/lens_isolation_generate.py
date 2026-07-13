#!/usr/bin/env python3
"""Generate unbiased normal-field dirty/lens pairs for lens isolation."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys

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
            "sersic_density_arcmin2": config.sersic_density_arcmin2,
            "tng_density_arcmin2": config.tng_density_arcmin2,
            "tng_redshift_mode": config.tng_redshift_mode,
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

    from euclid_polish.experiments.lens_isolation.generation import LensCaptureAdapter
    from euclid_polish.experiments.lens_isolation.records import (
        generate_split,
        write_dataset_metadata,
    )
    from euclid_polish.psf.psf_library import load_all_band_psf_sets
    from euclid_polish.sky.generation.sky_simulator import SkySimulator, SkySimulatorConfig
    from euclid_polish.sky.observation.observation_simulator import (
        ObservationSimulator,
        ObservationSimulatorConfig,
    )

    for subset in counts:
        if not args.force and _published_config_mismatch(out_dir, subset, config_fingerprint):
            raise ValueError(
                "incompatible lens-isolation artifacts found; rerun generation with --force "
                "to replace only data/experiments/lens_isolation records"
            )

    # Catalog=None enforces the pure-TNG field population.  SkySimulator fails
    # before any record writer opens when the required TNG atlas is missing.
    sky = SkySimulator(
        None,
        SkySimulatorConfig(
            image_size=config.image_size,
            pixel_scale=Config.DEFAULT_PIXEL_SCALE,
            sersic_density_arcmin2=config.sersic_density_arcmin2,
            tng_density_arcmin2=config.tng_density_arcmin2,
            tng_redshift_mode=config.tng_redshift_mode,
            tng_galaxy_dir=args.tng_dir or Config.TNG_SKIRT_DIR,
            lens_density_arcmin2=config.lens_density_arcmin2,
        ),
    )
    psf_sets = load_all_band_psf_sets(
        psf_dir=args.psf_dir or Config.EUCLID_PSF_DIR,
        target_pixel_scale=Config.DEFAULT_PIXEL_SCALE,
    )
    observation = ObservationSimulator(psf_sets_by_band=psf_sets, config=ObservationSimulatorConfig())
    generator = LensCaptureAdapter(sky, observation)
    master_seed = config.seed if config.seed >= 0 else secrets.randbits(63)
    summaries = {}
    for offset, (subset, count) in enumerate(counts.items()):
        summaries[subset] = generate_split(
            generator,
            out_dir,
            subset,
            count=count,
            seed=master_seed + offset,
            config_fingerprint=config_fingerprint,
            force=args.force,
            workers=args.workers,
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
