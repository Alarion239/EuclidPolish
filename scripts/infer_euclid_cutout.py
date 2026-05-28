#!/usr/bin/env python
"""Download a real Euclid 4-band cutout, run it through a trained WDSR
checkpoint, and write two FITS: the super-resolved VIS (SR, 0.05"/pix)
and the original VIS LR cutout (0.10"/pix).

The model takes the 4-band Euclid LR cube (VIS + NIR Y/J/H) as input and
emits a single-band VIS HR image, so all four bands are fetched at the
same sky footprint, converted from the archive's ADU/s to electrons via
each band's MAGZERO, stacked, and run through ``reconstruct``.

Usage:
    python scripts/infer_euclid_cutout.py \
        --ra 52.5389 --dec -30.5 --vis-pixels 2048 \
        --ckpt-dir ckpt/wdsr --out-dir data/euclid_inference/adhoc
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.euclid.downloader import fetch_cutout_at
from euclid_polish.training.inference import (
    load_model_from_checkpoint, reconstruct,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ra", type=float, required=True, help="ICRS RA (deg).")
    p.add_argument("--dec", type=float, required=True, help="ICRS Dec (deg).")
    p.add_argument("--vis-pixels", type=int, default=2048,
                   help="VIS cutout side in 0.10\"/pix pixels (default 2048).")
    p.add_argument("--ckpt-dir", default=Config.DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--num-res-blocks", type=int,
                   default=Config.DEFAULT_NUM_RES_BLOCKS)
    p.add_argument("--out-dir",
                   default=os.path.join(Config.DATA_DIR,
                                        "euclid_inference", "adhoc"))
    return p.parse_args()


def _scaled_wcs_header(vis_header, scale: int):
    """Return a copy of the VIS header with WCS magnified by ``scale``.

    SR has ``scale``× more pixels per side at ``1/scale`` the pixel
    scale. Standard magnify transform: ``CRPIX → scale·CRPIX − (scale−1)/2``
    and the pixel-scale terms (CD matrix or CDELT) divided by ``scale``.
    Falls back to the unscaled header if the WCS can't be parsed.
    """
    hdr = vis_header.copy()
    try:
        w = WCS(vis_header)
        if not w.has_celestial:
            return hdr
        w = w.deepcopy()
        off = (scale - 1) / 2.0
        w.wcs.crpix = [c * scale - off for c in w.wcs.crpix]
        if w.wcs.has_cd():
            w.wcs.cd = w.wcs.cd / scale
        else:
            w.wcs.cdelt = [d / scale for d in w.wcs.cdelt]
        scaled = w.to_header()
        # Drop the old WCS keys before merging the scaled ones so stale
        # CD/CDELT/CRPIX don't linger.
        for key in list(hdr.keys()):
            if key.startswith(("CRPIX", "CRVAL", "CDELT", "CD1_", "CD2_",
                               "PC1_", "PC2_", "CTYPE", "CUNIT")):
                del hdr[key]
        hdr.update(scaled)
    except Exception as e:  # noqa: BLE001 — WCS is a bonus, never fatal
        hdr["WCSNOTE"] = f"SR WCS rescale skipped: {type(e).__name__}"
    return hdr


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import tensorflow as tf

    latest = tf.train.latest_checkpoint(args.ckpt_dir)
    if not latest:
        print(f"ERROR: no checkpoint in {args.ckpt_dir}")
        return 1
    scale = Config.DEFAULT_REBIN_FACTOR
    model = load_model_from_checkpoint(
        args.ckpt_dir, scale, args.num_res_blocks,
        nchan_in=Config.NUM_LR_CHANNELS, nchan_out=Config.NUM_HR_CHANNELS,
    )
    print(f"loaded checkpoint: {latest}")
    print(f"position: RA={args.ra:.5f}  Dec={args.dec:+.5f}  "
          f"VIS size={args.vis_pixels} px")

    bands_e: dict = {}
    vis_header = None
    for band_name in Config.LR_INPUT_BAND_NAMES:
        outf = os.path.join(args.out_dir, f"raw_{band_name}.fits")
        if os.path.exists(outf) and os.path.getsize(outf) > 0:
            print(f"  {band_name}: cached")
        else:
            print(f"  {band_name}: downloading …")
            ok, err = fetch_cutout_at(
                ra=args.ra, dec=args.dec, band_name=band_name,
                output_file=outf, cutout_size_vis_pixels=args.vis_pixels,
            )
            if not ok:
                print(f"ERROR downloading {band_name}: {err}")
                return 1
        band = Config.get_band(band_name)
        with fits.open(outf) as hdul:
            arr = hdul[0].data.astype(np.float32)
            header = hdul[0].header
        if band_name == "VIS":
            vis_header = header.copy()
        magzero = float(header.get("MAGZERO", band.sim_zeropoint_e))
        adu_to_e = 10 ** ((band.sim_zeropoint_e - magzero) / 2.5)
        bands_e[band_name] = (arr * adu_to_e).astype(np.float32)
        print(f"    shape={bands_e[band_name].shape}  MAGZERO={magzero:.3f}"
              f"  ADU/s→e⁻={adu_to_e:.1f}")

    shapes = {n: bands_e[n].shape for n in Config.LR_INPUT_BAND_NAMES}
    if len(set(shapes.values())) != 1:
        print(f"ERROR: per-band shapes disagree: {shapes}")
        return 1

    cube = np.stack([bands_e[n] for n in Config.LR_INPUT_BAND_NAMES], axis=-1)
    print(f"running model on cube {cube.shape} …")
    lr_vis, sr = reconstruct(model, cube)
    print(f"  LR VIS: {lr_vis.shape}   SR: {sr.shape}")

    # The ESA cutout header carries an EXTNAME (and possibly other keys)
    # that are invalid on a PrimaryHDU; drop EXTNAME and let astropy
    # silently fix the rest so the write doesn't trip FITS verification.
    def _clean(hdr):
        if hdr is None:
            return None
        h = hdr.copy()
        for k in ("EXTNAME", "XTENSION"):
            if k in h:
                del h[k]
        return h

    vis_path = os.path.join(args.out_dir, "original_VIS.fits")
    sr_path = os.path.join(args.out_dir, "SR.fits")
    fits.PrimaryHDU(lr_vis.astype(np.float32),
                    header=_clean(vis_header)).writeto(
        vis_path, overwrite=True, output_verify="silentfix")
    sr_header = (_clean(_scaled_wcs_header(vis_header, scale))
                 if vis_header is not None else None)
    fits.PrimaryHDU(sr.astype(np.float32),
                    header=sr_header).writeto(
        sr_path, overwrite=True, output_verify="silentfix")
    print(f"wrote {vis_path}  ({lr_vis.shape}, 0.10\"/pix)")
    print(f"wrote {sr_path}  ({sr.shape}, 0.05\"/pix)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
