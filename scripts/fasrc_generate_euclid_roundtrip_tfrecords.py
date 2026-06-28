#!/usr/bin/env python
"""Stack 4-band Euclid sky cutouts into LR-only TFRecords for round-trip training.

Reads bundled multi-HDU cutouts written by
``scripts/fasrc_download_euclid_sky_cutouts.py`` — one
``sky_NNNN.fits`` per position, with four ``ImageHDU``s named
``VIS``, ``Y_E``, ``J_E``, ``H_E`` all delivered by the Euclid archive
on the shared 0.10″/pix mosaic grid (see ``config.py:222-223``). The
generator pulls the four images by ``EXTNAME``, stacks them into
``(H, W, 4)`` cubes in the canonical
:attr:`~euclid_polish.config.Config.LR_INPUT_BAND_NAMES` order, chops
each cube into many smaller training stamps, and writes them as
:class:`~euclid_polish.image.Image` records
(``is_clean=False``).

These records have *no HR side*. The round-trip training path
(:class:`~euclid_polish.training.trainer.Trainer`) detects them via a
per-example source tag and computes
``loss = |asinh(Conv(M(lr_real)) / scale) - lr_real|`` instead of the
supervised L1.

Output: ``$DATA_DIR/images/records_v2_euclid_roundtrip/{dirty_train,dirty_validate}.tfrecord``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from astropy.io import fits

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.catalog.photometry import adu_per_s_to_electrons_factor
from euclid_polish.observability.reporter import Reporter
from euclid_polish.image.tfio import open_writer
from euclid_polish.image import Image


# Input dir (bundled per-position cutouts), output records dir, and the
# sky-catalogue filename now live in Config — see
# Config.EUCLID_SKY_CUTOUTS_DIR, Config.ROUNDTRIP_RECORDS_DIR, and
# Config.EuclidSky.SKY_CATALOG_FILENAME.


def bundle_path_for_id(input_dir: str, pos_id: int) -> str:
    """Return the absolute path of the bundled FITS for a given position id."""
    return os.path.join(input_dir, f"sky_{pos_id:04d}.fits")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default=Config.EUCLID_SKY_CUTOUTS_DIR,
                   help="Directory containing the bundled per-position "
                        "FITS (``sky_NNNN.fits``) from the sky-download "
                        "step. Default: " + Config.EUCLID_SKY_CUTOUTS_DIR)
    p.add_argument("--output-dir", default=Config.ROUNDTRIP_RECORDS_DIR,
                   help="Where to write the TFRecord shards. "
                        "Default: " + Config.ROUNDTRIP_RECORDS_DIR)
    p.add_argument("--vis-pixels", type=int, default=512,
                   help="Expected cutout side (in 0.10\"/pix grid "
                        "pixels) from the download step. Used only as "
                        "a sanity print; bundled FITS carry the true "
                        "shape in their VIS HDU.")
    p.add_argument("--stamp-size", type=int, default=128,
                   help="Side length (LR pixels at 0.10\"/pix) of the "
                        "training stamps to chop each large cutout "
                        "into. Each large cutout yields "
                        "``(vis_pixels // stamp_size) ** 2`` stamps. "
                        "Default 128 (= 12.8\") fits a few galaxies "
                        "per stamp and matches typical training-time "
                        "crop sizes; smaller stamps = more records "
                        "but less effective augmentation room.")
    p.add_argument("--valid-fraction", type=float, default=0.1,
                   help="Fraction of positions held out for validation. "
                        "Split is at the *position* level, not the "
                        "stamp level, so stamps from the same large "
                        "cutout don't leak across train/validate.")
    p.add_argument("--max-zero-fraction", type=float, default=0.05,
                   help="Reject any stamp where more than this "
                        "fraction of pixels are exactly zero — "
                        "indicates the stamp clipped a mosaic edge "
                        "(zero-padded by the cutout service). Default "
                        "5%%.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the train/validate position "
                        "split.")
    p.add_argument("--scale-to-electrons", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Convert each band's archive pixels to electrons "
                        "over the stack using the SAME zeropoint factor as "
                        "every other Euclid-data path: "
                        "``10**((band.sim_zeropoint_e − MAGZERO)/2.5)`` with "
                        "MAGZERO read from each band's header (the cutout "
                        "bundle preserves it). This is the conversion "
                        "verify_star_photometry.py validates (measured/"
                        "catalog ratio ≈ 1), so the round-trip LR lands on "
                        "the identical electron scale as the synthetic, "
                        "star-anchor and direct-cutout lanes — the shared "
                        "asinh stretch (knee 1000 e⁻) then balances the "
                        "round-trip vs supervised loss. (The previous "
                        "``× t_total_s`` factor was ~2.3× too faint because "
                        "it assumed the archive ZP equalled "
                        "VIS_AB_ZP_E_PER_S=25.50 rather than the header's "
                        "MAGZERO≈24.6.) Use ``--no-scale-to-electrons`` only "
                        "to leave the raw archive units untouched.")
    p.add_argument("--dry-run", action="store_true",
                   help="Walk inputs and report counts, don't write.")
    return p.parse_args()


def _load_4band_cube(
    input_dir: str, position_id: int, vis_pixels: int,
    *, scale_to_electrons: bool = True,
) -> Optional[np.ndarray]:
    """Load and stack VIS + Y_E + J_E + H_E images for one position.

    Opens ``<input_dir>/sky_NNNN.fits`` once and pulls the four
    ``ImageHDU`` arrays by ``EXTNAME``. Returns ``(H, W, 4)`` float32
    in the canonical ``LR_INPUT_BAND_NAMES`` order, or ``None`` if the
    bundle is missing, the file is unreadable/corrupt, an ``EXTNAME``
    is missing, or shapes disagree.

    The ``vis_pixels`` argument is kept for backwards-compatible call
    sites but is no longer used to derive paths — bundle ids are flat
    (``sky_NNNN.fits``) regardless of cutout size.

    Per ``config.py:222``, the Euclid archive delivers all four bands
    on the same 0.10″/pix mosaic grid, so no on-disk resampling is
    needed here — just verify shapes agree and stack.

    **Units conversion** — ``scale_to_electrons=True`` (default) is what
    you want for training. The Euclid mosaic bands are delivered in
    archive **e⁻/s** calibrated to each header's ``MAGZERO`` (≈24.6 for
    VIS). To land on the synthetic/HST/star-anchor **total-electron**
    scale we apply the band's zeropoint factor
    ``10**((band.sim_zeropoint_e − MAGZERO)/2.5)`` (≈5130 for VIS) — the
    exact conversion ``verify_star_photometry.py`` validates against
    catalogue fluxes (measured/catalog ratio ≈ 1) and the same factor
    the direct-cutout reconstruct uses. This replaces the earlier
    ``× t_total_s`` (≈2242) approximation, which was ~2.3× too faint
    because it implicitly assumed the archive ZP equalled
    ``VIS_AB_ZP_E_PER_S=25.50`` rather than the header MAGZERO. With the
    shared asinh stretch (knee 1000 e⁻) the round-trip and supervised
    loss magnitudes then stay balanced. VIS keeps its sky-subtracted
    offset — irrelevant for the round-trip loss, which only checks
    ``Conv(M(lr)) ≈ lr`` on VIS and is sky-bias-invariant.
    """
    del vis_pixels  # bundle layout no longer needs the size for path lookup

    path = bundle_path_for_id(input_dir, position_id)
    if not os.path.isfile(path):
        return None
    try:
        with fits.open(path, memmap=False) as hdul:
            extnames = {hdu.name for hdu in hdul}
            channels: list = []
            for band_name in Config.LR_INPUT_BAND_NAMES:
                if band_name not in extnames:
                    return None
                hdu = hdul[band_name]
                arr = np.asarray(hdu.data, dtype=np.float32)
                if scale_to_electrons:
                    band = Config.get_band(band_name)
                    # Same archive e⁻/s → total-e⁻ conversion as the
                    # direct-cutout / star-anchor lanes: zeropoint factor
                    # from each band header's MAGZERO (preserved in the
                    # bundle). Falls back to sim_zeropoint_e (factor 1) only
                    # if a header somehow lacks MAGZERO.
                    magzero = float(hdu.header.get("MAGZERO",
                                                   band.sim_zeropoint_e))
                    arr = arr * adu_per_s_to_electrons_factor(magzero, band)
                channels.append(arr)
    except Exception:
        return None

    shapes = {c.shape for c in channels}
    if len(shapes) != 1:
        # Mismatched shapes typically mean one band's tile boundary
        # produced a smaller cutout. Drop the whole position — the
        # training-time crop assumes all bands align pixel-for-pixel.
        return None

    return np.stack(channels, axis=-1)


def _chop_cube(cube: np.ndarray, stamp_size: int) -> Iterator[np.ndarray]:
    """Non-overlapping ``stamp_size × stamp_size`` sub-cutouts.

    Trailing pixels (when ``H % stamp_size != 0``) are silently
    discarded. That's fine — the downloader produces fixed sizes and
    we pick stamp_size to divide vis_pixels cleanly.
    """
    H, W, _ = cube.shape
    ny = H // stamp_size
    nx = W // stamp_size
    for iy in range(ny):
        for ix in range(nx):
            y0 = iy * stamp_size
            x0 = ix * stamp_size
            yield cube[y0:y0 + stamp_size, x0:x0 + stamp_size, :]


def _stamp_is_usable(stamp: np.ndarray, *, max_zero_fraction: float) -> bool:
    """Reject stamps that hit a mosaic edge or have NaN pixels.

    The Euclid cutout service zero-pads outside the mosaic, so a high
    zero-fraction in any band is a strong signal we sliced into the
    pad region. NaNs (rare but possible from masking) would silently
    propagate through asinh and the loss; safer to drop the stamp.
    """
    if not np.isfinite(stamp).all():
        return False
    # Check per-band: any single band hitting the threshold disqualifies
    # the whole stamp (don't want NISP-padded VIS-real stamps either —
    # the round-trip Conv applies VIS only, but the model takes all 4
    # bands as input so any band with junk poisons the input).
    n_per_band = stamp.shape[0] * stamp.shape[1]
    for c in range(stamp.shape[-1]):
        zero_frac = float((stamp[..., c] == 0).sum()) / n_per_band
        if zero_frac > max_zero_fraction:
            return False
    return True


def main() -> int:
    args = parse_args()
    reporter = Reporter.from_env()
    if args.vis_pixels % args.stamp_size != 0:
        reporter.warn(
            f"vis_pixels ({args.vis_pixels}) is not a multiple of "
            f"stamp_size ({args.stamp_size}); trailing strip discarded"
        )
        print(f"WARNING: vis_pixels ({args.vis_pixels}) is not a multiple "
              f"of stamp_size ({args.stamp_size}); trailing strip will "
              "be discarded.")

    print("=" * 64)
    print(f"  Euclid round-trip TFRecord generation")
    print("=" * 64)
    print(f"  input dir       = {args.input_dir}")
    print(f"  output dir      = {args.output_dir}")
    print(f"  vis pixels      = {args.vis_pixels}  (LR grid 0.10\"/pix)")
    print(f"  stamp size      = {args.stamp_size} → "
          f"{(args.vis_pixels // args.stamp_size) ** 2} stamps / position")
    print(f"  valid fraction  = {args.valid_fraction}")
    print(f"  max zero frac   = {args.max_zero_fraction}")
    print(f"  scale to e⁻     = {args.scale_to_electrons}")
    print()

    t0 = time.time()

    # The sky-positions catalogue lives one level above the bundled
    # cutouts (under ``$DATA_DIR/euclid_sky/``); the bundles themselves
    # live in ``$DATA_DIR/euclid_sky/cutouts/``. Look for the CSV in
    # ``input_dir`` first (a user can point both at the same root), then
    # fall back to its parent so the default tree just works.
    cat_candidates = [
        os.path.join(args.input_dir, Config.EuclidSky.SKY_CATALOG_FILENAME),
        os.path.join(os.path.dirname(os.path.abspath(args.input_dir)),
                     Config.EuclidSky.SKY_CATALOG_FILENAME),
    ]
    cat_path = next((p for p in cat_candidates if os.path.isfile(p)), None)
    if cat_path is None:
        msg = (
            f"sky positions catalogue not found at any of: "
            f"{cat_candidates}"
        )
        reporter.error(msg)
        print(f"ERROR: {msg}")
        print("       Run scripts/fasrc_download_euclid_sky_cutouts.py first.")
        return 1
    positions = pd.read_csv(cat_path)
    reporter.set_stage("reading sky catalog")
    print(f"[1/3] sky catalogue: {len(positions)} positions  ({cat_path})")

    # Train/validate split at the position level so stamps from one
    # large cutout don't leak across the split.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(positions))
    n_valid = int(round(len(positions) * args.valid_fraction))
    valid_pos_ids = set(int(positions.iloc[i]["id"]) for i in perm[:n_valid])
    train_pos_ids = set(int(positions.iloc[i]["id"]) for i in perm[n_valid:])
    print(f"      split: {len(train_pos_ids)} train / "
          f"{len(valid_pos_ids)} validate positions")

    if args.dry_run:
        print()
        print(f"DRY RUN — would scan {len(positions)} positions for "
              f"4-band coverage and write up to "
              f"{len(positions) * (args.vis_pixels // args.stamp_size) ** 2} "
              "training stamps.")
        runtime = time.time() - t0
        print(f"\nRUNTIME_SECONDS={runtime:.1f}")
        return 0

    reporter.set_stage("streaming records")
    print(f"[2/3] streaming records to {args.output_dir} ...")
    counts: dict = {"train": 0, "validate": 0}
    dropped_no_4band = 0
    dropped_bad_stamp = 0

    splits = [
        ("train",    train_pos_ids),
        ("validate", valid_pos_ids),
    ]

    for subset, pos_ids in splits:
        with open_writer(f"dirty_{subset}", args.output_dir) as w:
            sorted_pos_ids = sorted(pos_ids)
            n_pos = len(sorted_pos_ids)
            for pos_i, pid in enumerate(sorted_pos_ids):
                reporter.set_step(pos_i + 1, n_pos, subset)
                cube = _load_4band_cube(
                    args.input_dir, pid, args.vis_pixels,
                    scale_to_electrons=args.scale_to_electrons,
                )
                if cube is None:
                    dropped_no_4band += 1
                    reporter.warn(
                        f"position {pid} bundle missing/corrupt/incomplete "
                        f"({bundle_path_for_id(args.input_dir, pid)})"
                    )
                    continue
                for stamp in _chop_cube(cube, args.stamp_size):
                    if not _stamp_is_usable(
                        stamp, max_zero_fraction=args.max_zero_fraction,
                    ):
                        dropped_bad_stamp += 1
                        continue
                    sky = Image(
                        data=stamp.astype(np.float32, copy=False),
                        pixel_scale_arcsec=Config.BAND_VIS.pixel_scale_lr_arcsec,
                        band_names=Config.LR_INPUT_BAND_NAMES,
                        is_clean=False,
                        index=counts[subset],
                        subset=subset,
                    )
                    w.write(sky, index=counts[subset])
                    counts[subset] += 1
        print(f"      {subset:8s}: wrote {counts[subset]} stamps")

    print()
    reporter.set_stage("done")
    print(f"[3/3] done.")
    print(f"      positions missing one or more bands : "
          f"{dropped_no_4band} / {len(positions)}")
    print(f"      stamps rejected (edge/NaN)          : {dropped_bad_stamp}")
    total = counts["train"] + counts["validate"]
    print(f"      total stamps written                : {total}  "
          f"({counts['train']} train + {counts['validate']} validate)")

    runtime = time.time() - t0
    print(f"\nRUNTIME_SECONDS={runtime:.1f}")
    print(f"N_TRAIN={counts['train']}")
    print(f"N_VALIDATE={counts['validate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
