"""Realistic colour for every talk image: the in-browser viewer's AB balance (services/cutout/static/viewer.js).

Electrons of each band are converted to "VIS-equivalent electrons" (the same AB flux) with the bands' electron
zeropoints, so a flat-f_nu source is white. R = H, G = (Y + J) / 2, B = VIS, each channel through the same asinh
stretch (cores compress toward white instead of saturating into one hue) above a black point at the noise level
(no coloured speckle in the sky). The old talk colour divided by a "Euclid white balance" that weakened H ~6x
relative to VIS, so nearly every galaxy came out blue.

Inputs are (..., 4) cubes in electrons, bands (VIS, Y, J, H), in any pixel area as long as white / knee / black
use the same one. JWST cubes converted to Euclid-band electrons (triples_lib.mjysr_to_e) work as they are.
"""
import sys

import numpy as np

sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish/output/jwst_triples_257/code")
sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish")
from render import inpaint_small  # noqa: E402
from triples_lib import sky_subtract, tile_cubes  # noqa: E402

ZP_E = (34.29662776165843, 31.396441190651323, 31.616441190651326, 31.566441190651325)   # VIS, Y, J, H
TO_VIS = np.array([10 ** (0.4 * (ZP_E[0] - zp)) for zp in ZP_E])


def ab_channels(cube, mode="full"):
    """(..., 4) electrons -> (R, G, B) in VIS-equivalent electrons (NaN kept). ``full``: R = H, G = (Y + J) / 2,
    B = VIS; ``noJ``: G = Y; ``vy`` (only VIS and Y covered): R = Y, G = (VIS + Y) / 2, B = VIS."""
    c = np.asarray(cube, np.float64) * TO_VIS
    v, y, j, h = (c[..., k] for k in range(4))
    if mode == "full":
        return h, 0.5 * (y + j), v
    if mode == "noJ":
        return h, y, v
    return y, 0.5 * (v + y), v


def ab_intensity(cube, mode="full"):
    r, g, b = ab_channels(cube, mode)
    return (r + g + b) / 3.0


def noise_sigma(cube, mode="full"):
    """Robust per-pixel sigma of the AB intensity (MAD of the sky), for the black point."""
    i = ab_intensity(cube, mode)
    i = i[np.isfinite(i)]
    lo = i[i < np.percentile(i, 60)]                       # mostly sky
    return 1.4826 * np.median(np.abs(lo - np.median(lo))) if lo.size else 0.0


def ab_rgb(cube, *, white, knee, black=0.0, hole=0.12, mode="full"):
    """(..., 4) electrons -> (..., 3) RGB in [0, 1]. ``white`` maps to 1, ``knee`` is the asinh linear->log scale,
    ``black`` is subtracted first (all in VIS-equivalent electrons per pixel of ``cube``). Uncovered pixels
    (NaN in any band) become flat dark grey ``hole``."""
    r, g, b = ab_channels(cube, mode)
    holes = ~(np.isfinite(r) & np.isfinite(g) & np.isfinite(b))
    norm = np.arcsinh(max(white - black, 1e-30) / knee)

    def f(c):
        c = np.nan_to_num(c) - black
        return np.clip(np.arcsinh(np.maximum(c, 0) / knee) / norm, 0, 1)

    rgb = np.stack([f(r), f(g), f(b)], -1)
    rgb[holes] = hole
    return rgb


def auto_levels(ref_cube, *, pct=99.8, q=15.0, black_sigma=1.5, sigma=None, mode="full", white_scale=1.0):
    """white / knee / black from a reference panel: white at the ``pct`` percentile of its AB intensity,
    knee = white / q, black = black_sigma x the sky sigma (pass ``sigma`` to take it from another panel,
    e.g. the noisy Euclid one, so the POLISHed and JWST panels share it)."""
    i = ab_intensity(ref_cube, mode)
    white = white_scale * float(np.nanpercentile(i, pct))
    s = noise_sigma(ref_cube, mode) if sigma is None else sigma
    return dict(white=white, knee=white / q, black=black_sigma * s)


_raw = {}


def raw_triple(tid):
    """(lr, sr, jw) of a NEXUS tile, sky-subtracted e- per 0.05" pixel, JWST small holes inpainted
    (the triples' cubes without their solar calibration and white balance)."""
    if tid not in _raw:
        lr, sr, jw = tile_cubes(tid)
        jw = np.stack([inpaint_small(jw[..., k]) for k in range(4)], -1)
        _raw.clear()
        _raw[tid] = [sky_subtract(c) for c in (lr, sr, jw)]
    return _raw[tid]
