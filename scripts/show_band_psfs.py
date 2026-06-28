#!/usr/bin/env python
"""Render the PSF for every band (empirical or Gaussian) for inspection."""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from euclid_polish.config import Config
from euclid_polish.psf.psf_library import (
    load_all_band_psfs,
    psf_inventory,
)

OUT = "data/vis/demo/band_psfs.png"


def main() -> int:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    inv = psf_inventory()
    psfs = load_all_band_psfs(target_pixel_scale=Config.DEFAULT_PIXEL_SCALE)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for col, (name, psf) in enumerate(psfs.items()):
        kind = "empirical" if inv[name] else "Gaussian fallback"

        # Top: log-scale full kernel (shows wings)
        ax = axes[0, col]
        d = np.clip(psf.data, 1e-8, None)
        ax.imshow(np.log10(d), origin="lower", cmap="viridis")
        ax.set_title(f"{name}  ({kind})\nshape={psf.shape}, "
                     f"FWHM≈{psf.fwhm_arcsec:.2f}\"",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        # Bottom: 1D radial slice through the centre (linear)
        ax = axes[1, col]
        cy = psf.data.shape[0] // 2
        slc = psf.data[cy]
        x_arcsec = (np.arange(len(slc)) - cy) * psf.pixel_scale
        ax.plot(x_arcsec, slc, lw=1.0, color="C0")
        ax.set_xlim(-2.0, 2.0)
        ax.set_yscale("log")
        ax.set_ylim(max(1e-10, slc[slc > 0].min() * 0.5), slc.max() * 2)
        ax.set_xlabel("arcsec")
        ax.grid(alpha=0.3)
        ax.set_title(f"radial slice (sum={psf.data.sum():.4f})", fontsize=9)

    fig.suptitle("Band-resolved PSFs @ HR pixel scale (0.05\"/pix). "
                 "Top: log10(intensity). Bottom: central-row slice.",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
