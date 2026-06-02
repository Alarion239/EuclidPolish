#!/usr/bin/env python
"""Write a PSF (or one cluster of a PSFSet) rotated by several angles into a
**multi-extension FITS** — the same stacked format ``PSFSet.save()`` produces
(``PrimaryHDU`` = mean, one ``ImageHDU`` per rotation, each sum=1) — so you can
flip through the rotations in a FITS viewer at full fidelity.

    python scripts/inspect_psf_rotation.py --psf data/euclid_psf/euclid_psf_VIS.fits
    python scripts/inspect_psf_rotation.py --psf euclid_psf_VIS.fits --hdu 3 \\
        --angles 0,15,30,45,90,135 --out psf_rotations.fits

``--hdu 0`` = the PrimaryHDU (the mean, for a PSFSet file); ``--hdu N`` (N≥1)
picks the Nth cluster PSF. Each output extension is stamped with ``ROTANGLE``
and named ``ROT_<angle>`` so the angle is visible in the viewer; HDU0 is the
mean of the rotations (deliberately smeared — it shows why rolls must NOT be
averaged). Read-only on the input — only writes the output FITS.
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

    rotated = []
    for ang in angles:
        rp = psf.rotated(ang, order=args.order)
        if crop:
            rp = rp.centre_cropped_to(crop, renormalise=False)
        rotated.append(rp)
        print(f"  {ang:7.2f}°  sum={rp.data.sum():.4f}  min={rp.data.min():.2e}")

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.psf)),
        f"{os.path.splitext(os.path.basename(args.psf))[0]}_rotations.fits")

    # Save in the canonical PSFSet stacked format (PrimaryHDU = mean,
    # ImageHDU per rotation), then stamp each rotation's angle into its header.
    pset = PSFSet.from_psfs(rotated)
    pset.save(os.path.dirname(out) or ".", os.path.basename(out))
    with fits.open(out, mode="update") as hdul:
        image_hdus = [h for h in hdul if getattr(h, "data", None) is not None]
        for ang, h in zip(angles, image_hdus[1:]):     # skip HDU0 (the mean)
            h.header["ROTANGLE"] = (float(ang), "Rotation applied (deg)")
            h.header["EXTNAME"] = f"ROT_{ang:g}"
        hdul.flush()

    print(f"\nwrote {out}")
    print(f"  HDU0 = mean of the rotations (smeared — averaging rolls is wrong)")
    for i, ang in enumerate(angles, start=1):
        print(f"  HDU{i} = {ang:g}°  (EXTNAME ROT_{ang:g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
