#!/usr/bin/env python
"""Download a real Euclid 4-band cutout, run it through the production model,
and write two FITS to your local disk:

  * ``original_stack.fits`` — the stacked 4-band original LR cube
    (VIS, Y_E, J_E, H_E) at 0.10"/pix, in electrons, stored as one image
    plane per band (band 0 = VIS, directly comparable to SR).
  * ``SR.fits`` — the super-resolved 4-band cube at 0.05"/pix, one image
    plane per band in the same order (band 0 = VIS); its header names the
    model (combiner and members).

The production model is the ensemble (``load_eval_ensemble``): the spatial
gate over the STARFULL members it reads, or their plain mean — logged — when
no current gate loads. It takes the 4-band Euclid LR cube (VIS + NIR Y/J/H)
and emits the 4-band HR cube, so all four bands are fetched at the same sky
footprint, converted from the archive's ADU/s to electrons via each band's
MAGZERO, stacked, and run through ``sr_from_model``. The raw per-band archive
cutouts are also kept as ``raw_<band>.fits``.

Usage:
    python scripts/infer_euclid_cutout.py \
        --ra 267.4229 --dec 64.8873 --vis-pixels 1024 \
        --out-dir data/euclid_inference/local_cutout
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from astropy.io import fits

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.catalog.downloader import fetch_cutout_at
from euclid_polish.config import Config
from euclid_polish.eval.ensemble_infer import load_eval_ensemble, sr_from_model
from euclid_polish.photometry import (
    adu_per_s_to_electrons,
    adu_per_s_to_electrons_factor,
    header_magzero,
)
from euclid_polish.training.inference import scaled_wcs_header


def _ascii(value) -> str:
    """FITS header text must be ASCII (member labels carry a '·')."""
    return (str(value).replace("·", ".")
            .encode("ascii", "replace").decode("ascii"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ra", type=float, required=True, help="ICRS RA (deg).")
    p.add_argument("--dec", type=float, required=True, help="ICRS Dec (deg).")
    p.add_argument("--vis-pixels", type=int, default=2048,
                   help="VIS cutout side in 0.10\"/pix pixels (default 2048).")
    p.add_argument("--ensemble-dir", default=None,
                   help="Ensemble base dir (default: <ckpt parent>/ensemble). "
                        "The production gate picks the members it reads.")
    p.add_argument("--num-res-blocks", type=int,
                   default=Config.DEFAULT_NUM_RES_BLOCKS)
    p.add_argument("--out-dir",
                   default=os.path.join(Config.DATA_DIR,
                                        "euclid_inference", "adhoc"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    scale = Config.DEFAULT_REBIN_FACTOR
    model = load_eval_ensemble(args.ensemble_dir, args.num_res_blocks, log=print)
    print(f"model: {model.label}")
    print(f"position: RA={args.ra:.5f}  Dec={args.dec:+.5f}  "
          f"VIS size={args.vis_pixels} px")

    bands_e: dict = {}
    vis_header = None
    for band_name in Config.LR_INPUT_BAND_NAMES:
        # Always download fresh — reusing a cached raw file would silently
        # serve a *previous* position's cutout when --out-dir is reused.
        outf = os.path.join(args.out_dir, f"raw_{band_name}.fits")
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
        magzero = header_magzero(header, source=f"{band_name} cutout")
        adu_to_e = adu_per_s_to_electrons_factor(magzero, band)
        bands_e[band_name] = adu_per_s_to_electrons(arr, magzero, band)
        print(f"    shape={bands_e[band_name].shape}  MAGZERO={magzero:.3f}"
              f"  ADU/s→e⁻={adu_to_e:.1f}")

    shapes = {n: bands_e[n].shape for n in Config.LR_INPUT_BAND_NAMES}
    if len(set(shapes.values())) != 1:
        print(f"ERROR: per-band shapes disagree: {shapes}")
        return 1

    cube = np.stack([bands_e[n] for n in Config.LR_INPUT_BAND_NAMES], axis=-1)
    print(f"running model on cube {cube.shape} …")
    lr_vis, sr, _members = sr_from_model(model, cube)
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

    # Stacked 4-band original: one image plane per band (NAXIS3), in
    # LR_INPUT_BAND_NAMES order so band 0 is VIS. ``axis=0`` puts the band
    # axis as the slowest FITS axis, so viewers show four H×W planes under
    # the 2-D VIS WCS. All header values stay plain ASCII (FITS requires
    # it — non-ASCII silently crashes the write).
    stack = np.stack(
        [bands_e[n] for n in Config.LR_INPUT_BAND_NAMES], axis=0,
    ).astype(np.float32)
    stack_header = _clean(vis_header) or fits.Header()
    stack_header["OBJECT"] = "Euclid LR stack (electrons)"
    stack_header["BUNIT"]  = "electron"
    stack_header["BANDS"]  = (",".join(Config.LR_INPUT_BAND_NAMES),
                              "NAXIS3 plane order")
    stack_path = os.path.join(args.out_dir, "original_stack.fits")
    fits.PrimaryHDU(stack, header=stack_header).writeto(
        stack_path, overwrite=True, output_verify="silentfix")

    sr_header = (_clean(scaled_wcs_header(vis_header, scale))
                 if vis_header is not None else fits.Header())
    # 4-band SR cube → one plane per band (same convention as the
    # original_stack file).
    sr = np.ascontiguousarray(np.moveaxis(np.asarray(sr, dtype=np.float32), -1, 0))
    sr_header["OBJECT"] = "Euclid SR (WDSR, 4-band)"
    sr_header["BANDS"]  = (",".join(Config.LR_INPUT_BAND_NAMES),
                           "NAXIS3 plane order (band 0 = VIS)")
    sr_header["BUNIT"]  = "electron"
    # The model behind the SR: its combiner (member_mean = the plain mean),
    # the members it was fitted on and the ones that ran (its reads).
    sr_header["MODEL"] = _ascii(model.label)    # no comment: labels run long
    sr_header["COMBINER"] = (model.combiner_kind or "member_mean",
                             "production combiner")
    sr_header["MEMBERS"] = (",".join(map(_ascii, model.member_labels)),
                            "fitted member labels")
    sr_header["RUNMEMB"] = (",".join(map(_ascii, model.run_labels)),
                            "member labels that ran")
    sr_path = os.path.join(args.out_dir, "SR.fits")
    fits.PrimaryHDU(sr, header=sr_header).writeto(
        sr_path, overwrite=True, output_verify="silentfix")
    print(f"wrote {stack_path}  ({stack.shape}, 0.10\"/pix, "
          f"bands={Config.LR_INPUT_BAND_NAMES})")
    print(f"wrote {sr_path}  ({sr.shape}, 0.05\"/pix)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
