#!/usr/bin/env python
"""Stack 4-band Euclid sky cutouts into LR-only TFRecords for round-trip training.

Reads per-band cutouts written by
``scripts/fasrc_download_euclid_sky_cutouts.py`` (one large stamp per
position per band, all delivered by the Euclid archive on the shared
0.10″/pix mosaic grid — see ``config.py:222-223``), stacks them into
``(H, W, 4)`` cubes in the canonical
:attr:`~euclid_polish.config.Config.LR_INPUT_BAND_NAMES` order, chops
each cube into many smaller training stamps, and writes them as
:class:`~euclid_polish.sky.types.MultiBandSkyImage` records
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

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config


# Input: where the sky downloader put the per-band cutouts.
DEFAULT_INPUT_DIR = os.path.join(Config.DATA_DIR, "euclid_sky")
# Output: separate records directory so the dataset loader can point
# at it independently of the synthetic/HST stores.
DEFAULT_OUTPUT_DIR = os.path.join(
    Config.DATA_DIR, "images", "records_v2_euclid_roundtrip",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default=DEFAULT_INPUT_DIR,
                   help="Root that holds the sky catalog + per-band "
                        "cutout directories from the sky-download "
                        "step. Default: " + DEFAULT_INPUT_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help="Where to write the TFRecord shards. "
                        "Default: " + DEFAULT_OUTPUT_DIR)
    p.add_argument("--vis-pixels", type=int, default=512,
                   help="Cutout size (in 0.10\"/pix grid pixels) used "
                        "by the download step. Used to locate the "
                        "FITS files on disk (filenames embed the "
                        "size). Must match what was downloaded.")
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
                   help="Multiply each band's pixel values by its "
                        "``t_total_s`` before writing. Euclid archive "
                        "mosaics arrive in e⁻/s; the synthetic and "
                        "HST records are in total electrons over the "
                        "full stack. Scaling here puts everything on "
                        "the same e⁻ scale so the shared asinh stretch "
                        "(knee 1000 e⁻) compresses all three sources "
                        "to comparable signal ranges and the round-"
                        "trip vs supervised loss magnitudes stay "
                        "balanced. Verified empirically: NISP medians "
                        "match expected per-pixel sky to ~10 % in "
                        "e⁻/s, off by 400× in total e⁻. Use "
                        "``--no-scale-to-electrons`` only if you've "
                        "confirmed the archive changed units.")
    p.add_argument("--dry-run", action="store_true",
                   help="Walk inputs and report counts, don't write.")
    return p.parse_args()


def _band_cutout_path(input_dir: str, band_name: str, position_id: int,
                      vis_pixels: int) -> str:
    """Match the filename convention from the downloader (``star_NNNN_SIZE.fits``)."""
    cutout_dir = Config.cutout_dir_for_band(
        band_name, root=os.path.join(input_dir, Config.CUTOUTS_SUBDIR),
    )
    return os.path.join(cutout_dir, f"star_{position_id:04d}_{vis_pixels}.fits")


def _load_4band_cube(
    input_dir: str, position_id: int, vis_pixels: int,
    *, scale_to_electrons: bool = True,
) -> Optional[np.ndarray]:
    """Load and stack VIS + Y_E + J_E + H_E cutouts for one position.

    Returns ``(H, W, 4)`` float32 cube in the canonical
    ``LR_INPUT_BAND_NAMES`` order, or ``None`` if any band is missing,
    unreadable, or has a mismatched shape (e.g. truncated download).

    Per ``config.py:222``, the Euclid archive delivers all four bands
    on the same 0.10″/pix mosaic grid, so no on-disk resampling is
    needed here — just verify shapes agree and stack.

    **Units conversion** — ``scale_to_electrons=True`` (default) is what
    you want for training. Empirical inspection of downloaded cutouts
    (see chunk-A units check in the PR description) shows all four
    Euclid mosaic bands are delivered in **e⁻/s**: NISP medians match
    expected per-pixel sky brightness to ~10 %, VIS medians sit near 0
    (sky-subtracted by the archive). The synthetic/HST records are in
    **total electrons** over the full integration stack. Multiplying
    each band by its ``t_total_s`` puts the round-trip records on the
    same scale, so the same asinh stretch (knee 1000 e⁻) compresses
    everything to comparable ranges and the supervised vs round-trip
    loss magnitudes stay balanced. VIS keeps its sky-subtracted offset
    — irrelevant for the round-trip loss, which only checks
    ``Conv(M(lr)) ≈ lr`` on VIS and is sky-bias-invariant.
    """
    from astropy.io import fits

    channels: list = []
    for band_name in Config.LR_INPUT_BAND_NAMES:
        path = _band_cutout_path(input_dir, band_name, position_id, vis_pixels)
        if not os.path.isfile(path):
            return None
        try:
            with fits.open(path, memmap=False) as hdul:
                arr = np.asarray(hdul[0].data, dtype=np.float32)
        except Exception:
            return None
        if scale_to_electrons:
            band = Config.get_band(band_name)
            arr = arr * float(band.t_total_s)
        channels.append(arr)

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
    if args.vis_pixels % args.stamp_size != 0:
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

    cat_path = os.path.join(args.input_dir, Config.CATALOG_FILE)
    if not os.path.isfile(cat_path):
        print(f"ERROR: sky catalog not found at {cat_path}")
        print("       Run scripts/fasrc_download_euclid_sky_cutouts.py first.")
        return 1
    positions = pd.read_csv(cat_path)
    print(f"[1/3] sky catalog: {len(positions)} positions")

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

    from euclid_polish.sky.tfrecord import open_multiband_writer
    from euclid_polish.sky.types import MultiBandSkyImage

    print(f"[2/3] streaming records to {args.output_dir} ...")
    counts: dict = {"train": 0, "validate": 0}
    dropped_no_4band = 0
    dropped_bad_stamp = 0

    splits = [
        ("train",    train_pos_ids),
        ("validate", valid_pos_ids),
    ]

    for subset, pos_ids in splits:
        with open_multiband_writer(f"dirty_{subset}", args.output_dir) as w:
            for pid in sorted(pos_ids):
                cube = _load_4band_cube(
                    args.input_dir, pid, args.vis_pixels,
                    scale_to_electrons=args.scale_to_electrons,
                )
                if cube is None:
                    dropped_no_4band += 1
                    continue
                for stamp in _chop_cube(cube, args.stamp_size):
                    if not _stamp_is_usable(
                        stamp, max_zero_fraction=args.max_zero_fraction,
                    ):
                        dropped_bad_stamp += 1
                        continue
                    sky = MultiBandSkyImage(
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
