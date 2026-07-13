#!/usr/bin/env python3
"""Generate isolated paired records for the lens-isolation experiment."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import tempfile

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.experiments.lens_isolation.config import (
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
    parser.add_argument("--max-lens-retries", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--catalog", default=None)
    parser.add_argument("--psf-dir", default=None)
    parser.add_argument("--sersic-density", type=float, default=None)
    parser.add_argument("--tng-density", type=float, default=0.0)
    parser.add_argument("--star-density", type=float, default=None)
    parser.add_argument("--tng-dir", default=None)
    return parser.parse_args(argv)


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
        max_lens_retries=args.max_lens_retries,
    )
    out_dir = assert_safe_output(args.out_dir)
    plan = {
        "experiment": "lens_isolation",
        "out_dir": os.path.abspath(out_dir),
        "counts": {
            "train": config.n_train,
            "validate": config.n_validate,
            "test": config.n_test,
        },
        "image_size": config.image_size,
        "positive_fraction": config.positive_fraction,
        "seed": config.seed,
        "workers": args.workers,
        "force": bool(args.force),
    }
    if args.dry_run:
        print(json.dumps(plan, sort_keys=True))
        return 0

    # Heavy simulation imports are intentionally deferred so WebUI dry-runs
    # and command validation remain cheap and work without data dependencies.
    from euclid_polish.config import Config
    from euclid_polish.experiments.lens_isolation.generation import (
        LensIsolationGenerator,
    )
    from euclid_polish.experiments.lens_isolation.records import (
        dataset_fingerprint,
        generate_split,
    )
    from euclid_polish.psf.psf_library import load_all_band_psf_sets
    from euclid_polish.sky.generation.cosmos2025 import (
        ensure_prefiltered_catalog,
        open_cosmos2025,
    )
    from euclid_polish.sky.generation.sky_simulator import (
        SkySimulator,
        SkySimulatorConfig,
    )
    from euclid_polish.sky.observation.observation_simulator import (
        ObservationSimulator,
        ObservationSimulatorConfig,
    )

    sersic_density = (
        Config.DEFAULT_GAL_DENSITY_ARCMIN2 if args.sersic_density is None else args.sersic_density
    )
    catalog = None
    if sersic_density > 0:
        catalog_path = args.catalog or Config.COSMOS2025_CATALOG_PATH
        catalog = open_cosmos2025(path=ensure_prefiltered_catalog(catalog_path))
    sky_config = SkySimulatorConfig(
        image_size=config.image_size,
        pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        sersic_density_arcmin2=sersic_density,
        tng_density_arcmin2=args.tng_density,
        tng_galaxy_dir=args.tng_dir or Config.TNG_SKIRT_DIR,
        star_density_arcmin2=(
            Config.DEFAULT_STAR_DENSITY_ARCMIN2 if args.star_density is None else args.star_density
        ),
        # Positives explicitly request one lens; the density is irrelevant.
        lens_density_arcmin2=0.0,
    )
    sky = SkySimulator(catalog, sky_config)
    psf_sets = load_all_band_psf_sets(
        psf_dir=args.psf_dir or Config.EUCLID_PSF_DIR,
        target_pixel_scale=Config.DEFAULT_PIXEL_SCALE,
    )
    observation = ObservationSimulator(
        psf_sets_by_band=psf_sets,
        config=ObservationSimulatorConfig(add_noise=True),
    )
    generator = LensIsolationGenerator(
        sky,
        observation,
        crop_size=Config.DEFAULT_HR_CROP_SIZE,
        max_lens_retries=config.max_lens_retries,
    )
    run_seed = config.seed if config.seed >= 0 else secrets.randbits(63)
    summaries = {}
    for offset, (subset, count) in enumerate(plan["counts"].items()):
        summaries[subset] = generate_split(
            generator,
            out_dir,
            subset,
            count=count,
            seed=run_seed + offset,
            force=args.force,
            workers=args.workers,
        )
        print(f"{subset}: {count} examples ({'reused' if summaries[subset].reused else 'generated'})")
    metadata = {
        **plan,
        "seed": run_seed,
        "fingerprint": dataset_fingerprint(out_dir),
        "splits": {
            name: {
                "count": summary.count,
                "n_positive": summary.n_positive,
                "n_negative": summary.n_negative,
            }
            for name, summary in summaries.items()
        },
    }
    os.makedirs(out_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix="dataset.json.tmp-", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        os.replace(temp_path, os.path.join(out_dir, "dataset.json"))
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
