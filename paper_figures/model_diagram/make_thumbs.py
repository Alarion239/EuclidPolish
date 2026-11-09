#!/usr/bin/env python
"""Thumbnails for the model diagram (``model_diagram.tex``).

A real Euclid Q1 grade-A lens candidate (cached real tile) and its production
SR (Universal 257): per-band greyscale planes (VIS, Y_E, J_E, H_E) and the
console's "vivid" colour composite (R = H, G = mean(Y, J), B = VIS, each
asinh-stretched at the VIS sky sigma), LR and SR on the same surface-brightness
scale (SR electrons × 4 = per LR-pixel area).

    python paper_figures/model_diagram/make_thumbs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from euclid_polish.web.helpers import real_tiles  # noqa: E402

OUT = Path(__file__).resolve().parent / "thumbs"
TILE = "ra0057.00402_decm050.77530"      # Q1 grade-A lens 102018665_NEG570040238507752998
CROP_LR = 64                              # 6.4 arcsec
BANDS = ("VIS", "Y_E", "J_E", "H_E")
#: console vivid weights (viewer/color.ts): VIS-equivalent electrons
VIVID = {"VIS": 1.0, "Y_E": 0.2088, "J_E": 0.2929, "H_E": 0.3799}


def _stretch(x: np.ndarray, knee: float, white: float) -> np.ndarray:
    return np.clip(np.arcsinh(x / knee) / np.arcsinh(white / knee), 0.0, 1.0)


def _save(rgb_or_grey: np.ndarray, name: str, size: int = 512) -> None:
    img = Image.fromarray((rgb_or_grey * 255).astype(np.uint8))
    img.resize((size, size), Image.NEAREST).save(OUT / name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    entry = real_tiles.get_entry("tile", TILE)
    lr = real_tiles.get_tile("tile", TILE, entry=entry).lr_e
    sr = real_tiles.load_output(entry, "production")[0] * 4.0
    side = lr.shape[0]
    o = (side - CROP_LR) // 2
    lr = lr[o:o + CROP_LR, o:o + CROP_LR]
    sr = sr[2 * o:2 * (o + CROP_LR), 2 * o:2 * (o + CROP_LR)]

    def vis_equiv(cube, b):
        return cube[..., b] / VIVID[BANDS[b]]

    vis = lr[..., 0]
    sigma = 1.4826 * float(np.median(np.abs(vis - np.median(vis))))
    white = float(np.percentile(vis, 99.8))
    for b, name in enumerate(BANDS):
        _save(_stretch(vis_equiv(lr, b), sigma, white), f"lr_{name}.png")
        _save(_stretch(vis_equiv(sr, b), sigma, white), f"sr_{name}.png")
    for tag, cube in (("lr", lr), ("sr", sr)):
        r = vis_equiv(cube, 3)
        g = 0.5 * (vis_equiv(cube, 1) + vis_equiv(cube, 2))
        rgb = np.stack([_stretch(c, sigma, white) for c in (r, g, cube[..., 0])], -1)
        _save(rgb, f"{tag}_rgb.png")
    print(f"wrote thumbnails to {OUT} (sigma {sigma:.3g} e-, white {white:.3g} e-)")


if __name__ == "__main__":
    main()
