#!/usr/bin/env python
"""Write a PSF (or one cluster of a PSFSet) rotated by several angles into a
**multi-extension FITS** — the same stacked format ``PSFSet.save()`` produces
(``PrimaryHDU`` = mean, one ``ImageHDU`` per rotation, each sum=1) — so you can
flip through the rotations in a FITS viewer at full fidelity.

    python scripts/inspect_psf_rotation.py --psf data/euclid_psf/euclid_psf_VIS.fits
    python scripts/inspect_psf_rotation.py --psf euclid_psf_VIS.fits --hdu 3 \\
        --angles 0,15,30,45,90,135 --out psf_rotations.fits

``--round-trip`` turns it into a consistency benchmark: for each angle n it
rotates by +n then -n (back to the original orientation) and writes the
**difference from the original**. A perfect (lossless) rotation gives all
zeros; whole multiples of 90° take the exact ``np.rot90`` path so their residual
is exactly 0, while other angles reveal the spline-interpolation smearing
(worst on the sharp spikes). HDU0 is the original PSF; HDU1..N are the residuals
(signed, raw — NOT renormalised), each header carrying MAXADIFF / MAXAREL / RMS.

``--hdu 0`` = the PrimaryHDU (the mean, for a PSFSet file); ``--hdu N`` (N≥1)
picks the Nth cluster PSF. Read-only on the input — only writes the output FITS.
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

from euclid_polish.psf import PSF, PSFSet


def _load_psf(path: str, hdu: int) -> PSF:
    with fits.open(path) as hdul:
        image_hdus = [h for h in hdul if getattr(h, "data", None) is not None]
        if not image_hdus:
            raise SystemExit(f"no image data in {path}")
        if hdu < 0 or hdu >= len(image_hdus):
            raise SystemExit(
                f"--hdu {hdu} out of range (file has {len(image_hdus)} "
                f"image HDU(s); 0 = primary/mean)")
        h = image_hdus[hdu]
        data = np.asarray(h.data, dtype=np.float32)
        pix = h.header.get("PXSCALE", h.header.get("PIXSCALE", 0.0))
    return PSF(data=data, pixel_scale=float(pix)).with_unit_sum()


def _maybe_crop(psf: PSF, crop: int) -> PSF:
    """Centre-crop to an odd side (no renormalise, so residuals stay clean)."""
    if not crop:
        return psf
    return psf.centre_cropped_to(crop, renormalise=False)


def _write_rotation_stack(out: str, rotated, angles) -> None:
    """Rotated PSFs in the canonical PSFSet stacked format (each sum=1),
    with each rotation's angle stamped into its header."""
    pset = PSFSet.from_psfs(rotated)
    pset.save(os.path.dirname(out) or ".", os.path.basename(out))
    with fits.open(out, mode="update") as hdul:
        image_hdus = [h for h in hdul if getattr(h, "data", None) is not None]
        for ang, h in zip(angles, image_hdus[1:], strict=False):     # skip HDU0 (the mean)
            h.header["ROTANGLE"] = (float(ang), "Rotation applied (deg)")
            h.header["EXTNAME"] = f"ROT_{ang:g}"
        hdul.flush()


def _write_diff_stack(out: str, ref: PSF, angles, diffs) -> None:
    """Round-trip residuals: HDU0 = original PSF, HDU1..N = raw signed
    differences (NOT renormalised — they integrate to ~0)."""
    peak = float(np.abs(ref.data).max()) or 1.0
    primary = fits.PrimaryHDU(data=ref.data.astype(np.float32))
    primary.header["PXSCALE"] = (float(ref.pixel_scale), "Pixel scale (arcsec/pixel)")
    primary.header["NPSF"] = (len(diffs), "Number of difference extensions")
    primary.header["ROUNDTRP"] = (True, "Round-trip (+n then -n) difference stack")
    primary.header["COMMENT"] = (
        "HDU0=original; HDU1..N = rotate(+n).rotate(-n) - original.")
    hdus = [primary]
    for ang, diff in zip(angles, diffs, strict=False):
        h = fits.ImageHDU(data=diff.astype(np.float32), name=f"DIFF_{ang:g}")
        h.header["PXSCALE"]  = (float(ref.pixel_scale), "Pixel scale (arcsec/pixel)")
        h.header["ROTANGLE"] = (float(ang), "Round-trip angle: +n then -n (deg)")
        h.header["MAXADIFF"] = (float(np.abs(diff).max()), "max |roundtrip - original|")
        h.header["MAXAREL"]  = (float(np.abs(diff).max() / peak), "MAXADIFF / original peak")
        h.header["RMSDIFF"]  = (float(np.sqrt(np.mean(diff ** 2))), "RMS of the difference")
        hdus.append(h)
    fits.HDUList(hdus).writeto(out, overwrite=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--psf", required=True, help="PSF (or PSFSet) FITS path.")
    ap.add_argument("--hdu", type=int, default=0,
                    help="Image HDU index (0 = primary/mean; ≥1 = cluster PSF).")
    ap.add_argument("--angles", default="0,15,30,45,90,135,180,270",
                    help="Comma-separated rotation angles in degrees.")
    ap.add_argument("--order", type=int, default=3,
                    help="Spline order for non-90° angles (3=cubic, 1=linear).")
    ap.add_argument("--crop", type=int, default=0,
                    help="Centre-crop each PSF to this odd side (0 = full).")
    ap.add_argument("--round-trip", action="store_true",
                    help="Benchmark: rotate by +n then -n and write the "
                         "difference from the original (≈0 = lossless; exactly "
                         "0 at 90° multiples).")
    ap.add_argument("--out", default=None, help="Output FITS path.")
    args = ap.parse_args()

    angles = [float(a) for a in args.angles.split(",") if a.strip() != ""]
    if not angles:
        raise SystemExit("no angles given")
    psf = _load_psf(args.psf, args.hdu)
    print(f"loaded {args.psf} HDU#{args.hdu}: shape={psf.shape}, "
          f"pixel_scale={psf.pixel_scale:.4f}\"/pix")

    crop = args.crop
    if crop and crop % 2 == 0:
        crop += 1                                  # centre_cropped_to wants odd

    if args.round_trip:
        ref = _maybe_crop(psf, crop)
        peak = float(np.abs(ref.data).max()) or 1.0
        diffs = []
        print("\nround-trip consistency (rotate +n then -n, minus original):")
        print(f"  {'angle':>7}   {'max|diff|':>11}  {'max|diff|/peak':>14}   {'rms':>11}")
        for ang in angles:
            rt = psf.rotated(ang, order=args.order).rotated(-ang, order=args.order)
            diff = (_maybe_crop(rt, crop).data.astype(np.float64)
                    - ref.data.astype(np.float64))
            diffs.append(diff)
            mad = float(np.abs(diff).max())
            print(f"  {ang:7.2f}°  {mad:11.3e}  {mad / peak:13.3%}   "
                  f"{float(np.sqrt(np.mean(diff ** 2))):11.3e}")
        out = args.out or os.path.join(
            os.path.dirname(os.path.abspath(args.psf)),
            f"{os.path.splitext(os.path.basename(args.psf))[0]}_roundtrip_diff.fits")
        _write_diff_stack(out, ref, angles, diffs)
        print(f"\nwrote {out}")
        print("  HDU0 = original PSF")
        for i, ang in enumerate(angles, start=1):
            print(f"  HDU{i} = round-trip residual at {ang:g}°  (EXTNAME DIFF_{ang:g})")
        return 0

    # Default: write the rotated PSFs themselves.
    rotated = []
    for ang in angles:
        rp = _maybe_crop(psf.rotated(ang, order=args.order), crop)
        rotated.append(rp)
        print(f"  {ang:7.2f}°  sum={rp.data.sum():.4f}  min={rp.data.min():.2e}")
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.psf)),
        f"{os.path.splitext(os.path.basename(args.psf))[0]}_rotations.fits")
    _write_rotation_stack(out, rotated, angles)
    print(f"\nwrote {out}")
    print("  HDU0 = mean of the rotations (smeared — averaging rolls is wrong)")
    for i, ang in enumerate(angles, start=1):
        print(f"  HDU{i} = {ang:g}°  (EXTNAME ROT_{ang:g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
