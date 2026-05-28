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
  2. Write the positions to ``$output_dir/sky_positions.csv`` (columns
     ``id, ra, dec``).
  3. For each position, fetch all four Euclid bands (VIS + NISP Y/J/H)
     via :func:`euclid_polish.euclid.downloader.fetch_cutout_at` and
     bundle them into a single multi-HDU FITS at
     ``$output_dir/cutouts/sky_NNNN.fits`` with one ``ImageHDU`` per
     band (``EXTNAME`` in ``VIS``, ``Y_E``, ``J_E``, ``H_E``).

A position is only written to disk if **all four bands** succeed —
half-bundled files would force the downstream TFRecord step to grep
for coverage across four directories. Better to lose a position than
to silently corrupt the 4-band invariant.

Migration notes — the on-disk layout used to be
``$output_dir/cutouts/<band>/star_NNNN_<size>.fits`` with a catalogue
called ``stars.csv``. To re-download cleanly under the new layout,
remove the old artefacts first::

    rm -rf $DATA_DIR/euclid_sky/cutouts/{VIS,Y_E,J_E,H_E}
    rm -f  $DATA_DIR/euclid_sky/stars.csv

Then re-run this script.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import tempfile
import time
from typing import List, Optional

import numpy as np
import pandas as pd

from astropy.io import fits

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.euclid.downloader import fetch_cutout_at
from euclid_polish.observability.reporter import Reporter


# Default sky-catalog location. Kept separate from the ePSF star catalog
# (``Config.DEFAULT_OUTPUT_DIR``) so the two pipelines can be re-run
# independently.
DEFAULT_SKY_OUTPUT_DIR = os.path.join(Config.DATA_DIR, "euclid_sky")
# Sky-catalogue filename — distinct from ``Config.CATALOG_FILE``
# (``stars.csv``) to make it obvious these are sky-region cutouts for
# self-supervised training, NOT ePSF star cutouts.
SKY_CATALOG_FILENAME = "sky_positions.csv"
# Subdir for the bundled FITS. Lives directly under the output root so
# the catalogue and cutouts share one easily-rsynced tree.
CUTOUTS_SUBDIR = "cutouts"


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

    Returns a DataFrame with ``(id, ra, dec, magnitude)`` columns.
    ``magnitude`` is NaN — irrelevant for sky cutouts, kept for legacy
    test compatibility.
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


def bundle_path_for_id(output_dir: str, pos_id: int) -> str:
    """Return the absolute path of the bundled FITS for a given position id."""
    return os.path.join(output_dir, CUTOUTS_SUBDIR, f"sky_{pos_id:04d}.fits")


def _write_bundle(
    target_path: str,
    *,
    pos_id: int,
    ra: float,
    dec: float,
    vis_pixels: int,
    arcsec_side: float,
    band_files: dict,
) -> None:
    """Combine per-band tempfile FITS into a single multi-HDU bundle.

    ``band_files`` is a ``{band_name: tempfile_path}`` map; one entry
    per :data:`Config.LR_INPUT_BAND_NAMES` band. The output has a data-
    less ``PrimaryHDU`` carrying position metadata in its header plus
    one ``ImageHDU`` per band (``EXTNAME = band_name``).
    """
    primary_hdr = fits.Header()
    primary_hdr["POS_ID"]  = (int(pos_id),         "Sky position id (matches sky_positions.csv)")
    primary_hdr["RA"]      = (float(ra),           "ICRS right ascension (deg)")
    primary_hdr["DEC"]     = (float(dec),          "ICRS declination (deg)")
    primary_hdr["VIS_PIX"] = (int(vis_pixels),     "Cutout side in VIS pixels (0.10\"/pix)")
    primary_hdr["ARCSEC"]  = (float(arcsec_side),  "Cutout side on sky (arcsec)")

    hdul = fits.HDUList([fits.PrimaryHDU(header=primary_hdr)])
    for band_name in Config.LR_INPUT_BAND_NAMES:
        src = band_files[band_name]
        with fits.open(src, memmap=False) as src_hdul:
            data = np.asarray(src_hdul[0].data)
            band_hdr = src_hdul[0].header.copy(strip=True)
        band_hdr["EXTNAME"] = band_name
        hdul.append(fits.ImageHDU(data=data, header=band_hdr, name=band_name))

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    hdul.writeto(target_path, overwrite=True)


def _fetch_position_bundle(
    *,
    pos_id: int,
    ra: float,
    dec: float,
    vis_pixels: int,
    arcsec_side: float,
    output_dir: str,
) -> dict:
    """Fetch all 4 bands for one position and write the bundled FITS.

    Returns a dict with ``{"status": "written" | "cached" | "failed",
    "id": pos_id, "errors": [...]}`` so the caller can tally outcomes.
    Per-band tempfiles are removed in a ``finally`` block so a crash
    mid-write doesn't strand them on disk.
    """
    target = bundle_path_for_id(output_dir, pos_id)
    if os.path.isfile(target) and os.path.getsize(target) > 0:
        return {"status": "cached", "id": pos_id, "errors": []}

    band_names: List[str] = list(Config.LR_INPUT_BAND_NAMES)
    band_files: dict = {}
    errors: list = []
    tmp_dir = tempfile.mkdtemp(prefix=f"euclid_sky_{pos_id:04d}_")
    try:
        for band_name in band_names:
            tmp_path = os.path.join(tmp_dir, f"{band_name}.fits")
            ok, err = fetch_cutout_at(
                ra=ra, dec=dec, band_name=band_name,
                output_file=tmp_path,
                cutout_size_vis_pixels=vis_pixels,
            )
            if not ok:
                errors.append(f"{band_name}: {err}")
                return {"status": "failed", "id": pos_id, "errors": errors}
            band_files[band_name] = tmp_path

        _write_bundle(
            target,
            pos_id=pos_id, ra=ra, dec=dec,
            vis_pixels=vis_pixels, arcsec_side=arcsec_side,
            band_files=band_files,
        )
        return {"status": "written", "id": pos_id, "errors": []}
    finally:
        for p in band_files.values():
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=DEFAULT_SKY_OUTPUT_DIR,
                   help="Root for the sky catalogue CSV and bundled "
                        "FITS cutouts. Default: "
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
                   help="Parallel positions in flight at once.")
    p.add_argument("--regenerate-catalog", action="store_true",
                   help="Overwrite the existing sky-positions CSV. "
                        "Without this flag, an existing catalogue is "
                        "reused (positions stay fixed across runs so "
                        "the bundle ids line up).")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for position generation. Only "
                        "matters on the first run (or with "
                        "--regenerate-catalog).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be done and exit.")
    return p.parse_args()


def _dir_size_bytes(path: str) -> int:
    """Sum of file sizes under ``path``; returns 0 if the dir is absent."""
    total = 0
    if not os.path.isdir(path):
        return 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                continue
    return total


def main() -> int:
    args = parse_args()
    reporter = Reporter.from_env()
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
    print(f"  bands (bundled)  = {list(Config.LR_INPUT_BAND_NAMES)}")
    print(f"  workers          = {args.workers}")
    print()

    t0 = time.perf_counter()

    # ---- 1. Sky catalogue: generate or reuse ----
    os.makedirs(args.output_dir, exist_ok=True)
    catalog_path = os.path.join(args.output_dir, SKY_CATALOG_FILENAME)
    if os.path.isfile(catalog_path) and not args.regenerate_catalog:
        positions = pd.read_csv(catalog_path)
        reporter.set_stage("reusing sky catalog")
        print(f"[1/2] reusing sky catalogue: {len(positions)} positions "
              f"in {catalog_path}")
    else:
        rng = np.random.default_rng(args.seed)
        positions = _uniform_disk_positions(
            args.ra_centre, args.dec_centre, args.radius_deg,
            args.n_positions, rng=rng,
        )
        if args.dry_run:
            print(f"[1/2] DRY RUN — would generate {len(positions)} positions "
                  f"and write to {catalog_path}")
        else:
            reporter.set_stage("generating sky catalog")
            positions.to_csv(catalog_path, index=False)
            print(f"[1/2] generated sky catalogue: {len(positions)} positions → "
                  f"{catalog_path}")
            for _, row in positions.head(3).iterrows():
                print(f"        id={int(row['id']):03d}  "
                      f"RA={row['ra']:9.5f}°  Dec={row['dec']:+9.5f}°")

    if args.dry_run:
        print()
        print(f"  DRY RUN — would fetch {len(Config.LR_INPUT_BAND_NAMES)} bands × "
              f"{len(positions)} positions at "
              f"{args.vis_pixels} VIS px each, bundling per position.")
        runtime = time.perf_counter() - t0
        print(f"\nRUNTIME_SECONDS={runtime:.1f}")
        return 0

    # ---- 2. Per-position bundled download ----
    reporter.set_stage("downloading bundles")
    cutouts_dir = os.path.join(args.output_dir, CUTOUTS_SUBDIR)
    os.makedirs(cutouts_dir, exist_ok=True)

    n_positions = len(positions)
    n_written  = 0
    n_cached   = 0
    n_failed   = 0

    # Materialise the (id, ra, dec) work-list up front so the
    # ThreadPoolExecutor can dispatch from a simple iterable.
    work = [
        (int(row["id"]), float(row["ra"]), float(row["dec"]))
        for _, row in positions.iterrows()
    ]

    completed_i = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_to_id = {
            pool.submit(
                _fetch_position_bundle,
                pos_id=pid, ra=ra, dec=dec,
                vis_pixels=args.vis_pixels, arcsec_side=arcsec_side,
                output_dir=args.output_dir,
            ): pid
            for (pid, ra, dec) in work
        }
        for fut in concurrent.futures.as_completed(future_to_id):
            pid = future_to_id[fut]
            completed_i += 1
            reporter.set_step(completed_i, n_positions, f"sky_{pid:04d}")
            try:
                result = fut.result()
            except Exception as e:
                n_failed += 1
                reporter.warn(f"position {pid} crashed: {type(e).__name__}: {e}")
                continue
            status = result["status"]
            if status == "written":
                n_written += 1
            elif status == "cached":
                n_cached += 1
            else:
                n_failed += 1
                for err in result["errors"]:
                    reporter.warn(f"position {pid} failed: {err}")

    # ---- 3. Summary ----
    total_bytes = _dir_size_bytes(cutouts_dir)
    print()
    print("=" * 64)
    print(f"Summary  ({(time.perf_counter() - t0) / 60:.1f} min total):")
    print(f"  positions written  = {n_written}")
    print(f"  positions cached   = {n_cached}  (already on disk; skipped)")
    print(f"  positions failed   = {n_failed}  (couldn't get all 4 bands)")
    print(f"  bundle dir size    = {total_bytes / 1e9:.2f} GB  ({cutouts_dir})")

    runtime = time.perf_counter() - t0
    print(f"\nRUNTIME_SECONDS={runtime:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
