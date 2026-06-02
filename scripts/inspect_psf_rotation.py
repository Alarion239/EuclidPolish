#!/usr/bin/env python
"""Render a PSF (or one cluster of a multi-extension PSFSet) at several
rotation angles as a montage PNG — to eyeball ``PSF.rotated()`` on a real,
spike-bearing PSF (e.g. confirm the diffraction spikes rotate cleanly and the
core stays put).

    python scripts/inspect_psf_rotation.py --psf data/euclid_psf/euclid_psf_VIS.fits
    python scripts/inspect_psf_rotation.py --psf euclid_psf_VIS.fits --hdu 3 \\
        --angles 0,15,30,45,90,135 --crop 201 --out psf_rot.png

``--hdu 0`` = the PrimaryHDU (the mean, for a PSFSet file); ``--hdu N`` (N≥1)
picks the Nth cluster PSF. Read-only — only writes the output PNG.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

from euclid_polish.psf import PSF


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


def _log_stretch(d: np.ndarray, floor_frac: float = 1e-4) -> np.ndarray:
    """log10 with a peak-relative floor so the faint spikes/wings show."""
    peak = float(np.nanmax(d)) or 1.0
    return np.log10(np.clip(d, peak * floor_frac, None))


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
                    help="Centre-crop each panel to this odd side for a zoom "
                         "(0 = full kernel).")
    ap.add_argument("--out", default=None, help="Output PNG path.")
    args = ap.parse_args()

    angles = [float(a) for a in args.angles.split(",") if a.strip() != ""]
    psf = _load_psf(args.psf, args.hdu)
    print(f"loaded {args.psf} HDU#{args.hdu}: shape={psf.shape}, "
          f"pixel_scale={psf.pixel_scale:.4f}\"/pix")

    n = len(angles)
    ncol = min(n, 4)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow),
                             squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        ang = angles[i]
        rp = psf.rotated(ang, order=args.order)
        d = rp.data
        if args.crop and args.crop > 0:
            d = rp.centre_cropped_to(args.crop, renormalise=False).data
        ax.imshow(_log_stretch(d), cmap="magma", origin="lower",
                  interpolation="nearest")
        # Crosshair at the centre so you can confirm the core stays put.
        cy, cx = (s // 2 for s in d.shape)
        ax.axhline(cy, color="cyan", lw=0.4, alpha=0.5)
        ax.axvline(cx, color="cyan", lw=0.4, alpha=0.5)
        neg = float(rp.data.min())
        ax.set_title(f"{ang:g}°  sum={rp.data.sum():.4f}  min={neg:.1e}",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.psf)),
        f"psf_rotation_hdu{args.hdu}.png")
    fig.suptitle(f"{os.path.basename(args.psf)} · HDU#{args.hdu} · "
                 f"log10 stretch", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
