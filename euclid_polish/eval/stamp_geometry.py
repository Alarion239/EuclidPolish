"""Pixel-stamp cropping geometry — pure numpy, no ML / archive deps.

``crop_stamp`` used to live in :mod:`euclid_polish.eval.synthetic_runner`, but
that module imports the astroquery-backed Euclid stack at module scope. Keeping
the crop primitive here provides a lightweight, reusable geometry helper.
"""

from __future__ import annotations

import numpy as np


def crop_stamp(plane, *, cx: float, cy: float, m: int):
    """Crop an m×m stamp from a 2-D ``plane`` centered at (cx, cy) pixel coords."""
    x0 = int(round(cx)) - m // 2
    y0 = int(round(cy)) - m // 2
    return np.asarray(plane)[y0:y0 + m, x0:x0 + m]
