#!/usr/bin/env python
"""Download multi-band Euclid sky cutouts for round-trip training.

The HST path (``scripts/fasrc_generate_hst_tfrecords.py``) gives us
forward-modelled HST→Euclid pairs with HR ground truth. Round-trip
training adds a self-supervised signal on *real* Euclid observations:
``loss = |Conv(M(LR_real)) - LR_real|`` where ``Conv`` is the
deterministic Euclid forward operator (PSF + rebin, no noise).

This script handles the data-acquisition half of that path:

  1. Generate ``N`` random sky positions inside a circular footprint
     (default: 2° radius around RA=270°, Dec=66° — a deep Euclid
     coverage region). Positions outside Euclid coverage are filtered
     downstream when the cutout service can't find a covering mosaic
     tile, so we don't need an explicit footprint mask.
  2. Write them as a sky catalog CSV in the same on-disk format
     :class:`euclid_polish.euclid.catalog.StarCatalog` already uses for
     ePSF stars — same loader, same flag tracking, separate file so
     the star catalog stays clean.
  3. Run :class:`euclid_polish.euclid.downloader.EuclidCutoutDownloader`
     once per band, with a *large* cutout size (default 512 VIS px =
     51.2″) so the TFRecord-generation step can chop each download into
     many smaller training stamps.

Per-band cutouts land in ``<output_dir>/cutouts/<band>/``, same layout
as the star path. Downstream, ``fasrc_generate_euclid_roundtrip_tfrecords.py``
stacks VIS + NISP into 4-channel cubes on the shared 0.10″ grid.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.euclid.catalog import StarCatalog
from euclid_polish.euclid.downloader import (
    DownloadConfig, EuclidCutoutDownloader,
)


# Default sky-catalog location. Kept separate from the star catalog
# (``Config.DEFAULT_OUTPUT_DIR``) so the per-(band, size) flag columns
# can't collide and the two pipelines can be re-run independently.
DEFAULT_SKY_OUTPUT_DIR = os.path.join(Config.DATA_DIR, "euclid_sky")


def _uniform_disk_positions(
    ra_centre_deg:  float,
    dec_centre_deg: float,
    radius_deg:     float,
    n_positions:    int,
    *,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Uniform random (RA, Dec) inside a small spherical disk.

    Uses a flat-sky rejection sample on a 2R × 2R square — fine for
    radii ≲ a few degrees, where the deviation from a true geodesic
    disk is sub-arcsec. The RA offset is divided by ``cos(dec_centre)``
    so the area density on the sphere stays uniform at high declination
    (at dec=66° this is a ~2.5× horizontal stretch compared to a naive
    flat sample, which would otherwise clump positions toward the poles
    of the local tangent plane).

    Returns a DataFrame with the same ``(id, ra, dec, magnitude)``
    columns :class:`StarCatalog` expects so the existing CSV reader
    accepts it verbatim. ``magnitude`` is NaN — irrelevant for sky
    cutouts, the catalog reader doesn't require it.
    """
    cos_dec = float(np.cos(np.deg2rad(dec_centre_deg)))
    # cos(90°) is ~6e-17 (positive due to float rounding), so a bare
    # ``cos_dec <= 0`` check would let pole-centred calls through and
    # blow up RA by 1e16×. Reject anything within ~3° of the poles.
    if cos_dec < 1e-3:
        raise ValueError(
            f"dec_centre_deg={dec_centre_deg} too close to a pole "
            "for the flat-sky approximation"
        )

    accepted: list = []
    # Acceptance rate of disk-in-square is π/4 ≈ 0.785, so 1.4× over-
    # sampling clears the budget in one round with high probability.
    while len(accepted) < n_positions:
        n_try = max(8, int((n_positions - len(accepted)) * 1.4))
        dx = rng.uniform(-radius_deg, radius_deg, n_try)
        dy = rng.uniform(-radius_deg, radius_deg, n_try)
        keep = (dx ** 2 + dy ** 2) <= radius_deg ** 2
        for x, y in zip(dx[keep], dy[keep]):
            ra  = (ra_centre_deg + x / cos_dec) % 360.0
            dec = dec_centre_deg + y
            accepted.append((ra, dec))
            if len(accepted) >= n_positions:
                break

    df = pd.DataFrame(accepted, columns=["ra", "dec"])
    df.insert(0, "id", range(len(df)))
    df["magnitude"] = np.nan
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=DEFAULT_SKY_OUTPUT_DIR,
                   help="Root for the sky catalog CSV and per-band "
                        "cutout directories. Default: "
                        f"{DEFAULT_SKY_OUTPUT_DIR}")
    p.add_argument("--n-positions", type=int, default=100,
                   help="Number of random sky positions to generate. "
                        "After per-position 4-band download some will "
                        "drop out (off-coverage, tile boundary, NISP "
                        "missing) — over-sample by ~2× the count of "
                        "fully-multi-band positions you need.")
    p.add_argument("--ra-centre", type=float, default=270.0,
                   help="Disk centre RA (deg). Default 270.")
    p.add_argument("--dec-centre", type=float, default=66.0,
                   help="Disk centre Dec (deg). Default 66 — a deep "
                        "Euclid coverage region near the NEP.")
    p.add_argument("--radius-deg", type=float, default=2.0,
                   help="Disk radius (deg). Default 2.")
    p.add_argument("--vis-pixels", type=int, default=512,
                   help="Cutout size in VIS pixels (= 0.10\"/pix). "
                        "Default 512 (= 51.2\") gives plenty of room "
                        "for the TFRecord chopper to extract many "
                        "training stamps per position; the same "
                        "angular extent gets requested from each band "
                        "at its own native pixel scale.")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel downloads per band")
    p.add_argument("--bands",
                   default=",".join(b.name for b in Config.BANDS),
                   help="Comma-separated band list; default = all "
                        "four (VIS + Y/J/H).")
    p.add_argument("--regenerate-catalog", action="store_true",
                   help="Overwrite the existing sky catalog CSV. "
                        "Without this flag, an existing catalog is "
                        "reused (positions stay fixed across runs so "
                        "the per-band download flag columns line up).")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for position generation. Only "
                        "matters on the first run (or with "
                        "--regenerate-catalog).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be done and exit.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    band_names = [n.strip() for n in args.bands.split(",") if n.strip()]
    arcsec_side = args.vis_pixels * Config.BAND_VIS.pixel_scale_lr_arcsec

    print("=" * 64)
    print(f"  Euclid sky cutout download (for round-trip training)")
    print("=" * 64)
    print(f"  output dir       = {args.output_dir}")
    print(f"  sky disk         = (RA={args.ra_centre:.3f}°, "
          f"Dec={args.dec_centre:.3f}°, r={args.radius_deg:.2f}°)")
    print(f"  n_positions      = {args.n_positions}")
    print(f"  cutout size      = {args.vis_pixels} VIS px "
          f"(= {arcsec_side:.1f}\")")
    print(f"  bands            = {band_names}")
    print(f"  workers / band   = {args.workers}")
    print()

    t0 = time.perf_counter()

    # ---- 1. Sky catalog: generate or reuse ----
    os.makedirs(args.output_dir, exist_ok=True)
    catalog_path = os.path.join(args.output_dir, Config.CATALOG_FILE)
    if os.path.isfile(catalog_path) and not args.regenerate_catalog:
        existing = pd.read_csv(catalog_path)
        print(f"[1/2] reusing sky catalog: {len(existing)} positions "
              f"in {catalog_path}")
    else:
        rng = np.random.default_rng(args.seed)
        df = _uniform_disk_positions(
            args.ra_centre, args.dec_centre, args.radius_deg,
            args.n_positions, rng=rng,
        )
        if args.dry_run:
            print(f"[1/2] DRY RUN — would generate {len(df)} positions "
                  f"and write to {catalog_path}")
        else:
            df.to_csv(catalog_path, index=False)
            print(f"[1/2] generated sky catalog: {len(df)} positions → "
                  f"{catalog_path}")
            # Show a few for sanity (RA/Dec inside the requested disk).
            for _, row in df.head(3).iterrows():
                print(f"        id={int(row['id']):03d}  "
                      f"RA={row['ra']:9.5f}°  Dec={row['dec']:+9.5f}°")

    if args.dry_run:
        print()
        print(f"  DRY RUN — would download {len(band_names)} bands × "
              f"{args.n_positions} positions at "
              f"{args.vis_pixels} VIS px each.")
        runtime = time.perf_counter() - t0
        print(f"\nRUNTIME_SECONDS={runtime:.1f}")
        return 0

    cat = StarCatalog(args.output_dir)
    if not cat.exists():
        print(f"ERROR: catalog write at {catalog_path} did not "
              "produce a readable file; check disk + permissions.")
        return 1

    # ---- 2. Per-band download ----
    summary: dict = {}
    for band_name in band_names:
        band = Config.get_band(band_name)
        native = band.cutout_size_for_arcsec(arcsec_side)
        print(f"\n=== {band_name}  (instrument={band.archive_instrument}"
              f"{('/' + band.archive_filter) if band.archive_filter else ''}, "
              f"native_size={native} px) ===")
        cfg = DownloadConfig.for_band(
            band_name,
            cutout_size_vis_pixels=args.vis_pixels,
            max_workers=args.workers,
        )
        downloader = EuclidCutoutDownloader(cat, cfg)
        t_band = time.perf_counter()
        result = downloader.download(show_progress=True)
        summary[band_name] = result
        print(f"  → {band_name}: downloaded={result['downloaded']}, "
              f"valid={result['valid']}, "
              f"corrupted={result['corrupted']}, "
              f"failed={result.get('failed', 0)}  "
              f"[{time.perf_counter() - t_band:.0f}s]")

    print()
    print("=" * 64)
    print(f"Summary  ({(time.perf_counter() - t0) / 60:.1f} min total):")
    for name, r in summary.items():
        print(f"  {name:5s}  +{r['downloaded']:4d}  "
              f"valid={r['valid']}  corrupted={r['corrupted']}  "
              f"failed={r.get('failed', 0)}")

    # How many positions have ALL four bands? That's the upper bound
    # for the round-trip TFRecord generator.
    valid_per_band = {n: int(r["valid"]) for n, r in summary.items()}
    if valid_per_band:
        worst = min(valid_per_band.values())
        print(f"\n  positions valid in *every* band (≤ min): "
              f"~{worst} — these are what the TFRecord step will use.")

    runtime = time.perf_counter() - t0
    print(f"\nRUNTIME_SECONDS={runtime:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
