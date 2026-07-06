#!/usr/bin/env python
"""Precompute the pre-rotated PSF kernel pools (one-time, per PSF extraction).

For every cluster ePSF of every band, draws K RANDOM telescope-roll angles
(default 12; the same angle table is shared across bands so a pool index is
one physical pointing in all four channels), rotates with the same order-3
spline generation would use, and streams the results to
``<psf_dir>/euclid_psf_rotpool_<BAND>.fits``. The unrotated original is
included too, and every kernel carries its source cluster index, roll angle
and NSTARS weight in the HDU header.

This amortises the ~92 ms/kernel rotation the FASRC benchmark measured to a
one-time build, and is the substrate for (a) rotation augmentation in
generation — which currently applies NO roll at all (psf_unrotated_prob=1.0)
— and (b) per-member PSF bagging: training a member against a seeded random
subset of clusters via euclid_polish.psf.rotpool.load_all_band_rotpools.

Sized for a FASRC login node — CPU-only, parallel across --workers:

    python scripts/pregenerate_psf_rotations.py                 # full build
    python scripts/pregenerate_psf_rotations.py --rotations 16 --workers 16
    python scripts/pregenerate_psf_rotations.py --crop 257      # 4x smaller

Re-running overwrites (build is deterministic for a fixed --seed).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from euclid_polish.config import Config  # noqa: E402
from euclid_polish.psf.psf_library import load_all_band_psf_sets  # noqa: E402
from euclid_polish.psf.rotpool import (  # noqa: E402
    DEFAULT_ROTATIONS,
    build_rotation_pool,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--psf-dir", default=Config.EUCLID_PSF_DIR,
                   help="Dir with the extracted band ePSF FITS files; the "
                        "pools are written next to them.")
    p.add_argument("--rotations", type=int, default=DEFAULT_ROTATIONS,
                   help="Random roll angles per cluster kernel (the "
                        "unrotated original is always included as well). "
                        "10–20 is the useful range.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the shared random angle table — fixed "
                        "seed ⇒ reproducible pool.")
    p.add_argument("--crop", type=int, default=0,
                   help="Centre-crop stored kernels to this odd side "
                        "(0 = keep full support). 257 keeps 99.4%% of the "
                        "VIS flux at 4x less disk/RAM — enough for the "
                        "crop-local on-the-fly forward; keep full size if "
                        "the pool must also serve full-field generation.")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                   help="Parallel rotation workers (default: half the cores "
                        "— polite on a login node).")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    crop = int(args.crop) or None

    print("Loading band ePSF sets (cleaned + resampled — the pool stores "
          "these, so the loader never re-cleans)…")
    t0 = time.perf_counter()
    psf_sets = load_all_band_psf_sets(psf_dir=args.psf_dir,
                                      target_pixel_scale=Config.DEFAULT_PIXEL_SCALE)
    print(f"  loaded in {time.perf_counter() - t0:.1f} s")

    n_kernels = sum(p.n for p in psf_sets.values())
    side = crop or max(p.shape[0] for p in psf_sets.values())
    total = n_kernels * (args.rotations + 1)
    gb = total * side * side * 4 / 1e9
    for name, pset in psf_sets.items():
        print(f"  {name}: {pset.n} kernels {pset.shape}")
    print(f"\nBuilding: {n_kernels} kernels × ({args.rotations} rolls + "
          f"original) = {total} pool kernels @ {side}² ≈ {gb:.1f} GB, "
          f"{args.workers} workers, seed {args.seed}")

    t0 = time.perf_counter()
    last = [0.0]

    def _prog(done, band_total, label):
        if time.perf_counter() - last[0] > 10:
            last[0] = time.perf_counter()
            print(f"  … {label}: {done}/{band_total} "
                  f"({time.perf_counter() - t0:.0f} s elapsed)", flush=True)

    paths = build_rotation_pool(
        psf_sets, psf_dir=args.psf_dir, rotations=int(args.rotations),
        seed=int(args.seed), crop_to=crop, workers=int(args.workers),
        on_progress=_prog)

    print(f"\n✓ pools built in {time.perf_counter() - t0:.0f} s:")
    for name, path in paths.items():
        print(f"  {name}: {path} ({os.path.getsize(path) / 1e9:.2f} GB)")
    print("\nNext: load with euclid_polish.psf.rotpool.load_all_band_rotpools"
          "(subset_clusters=N, subset_seed=member_seed) for per-member PSF "
          "bagging.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
