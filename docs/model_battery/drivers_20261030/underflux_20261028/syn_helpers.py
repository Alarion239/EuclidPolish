"""Real vs synthetic: where is flux lost, does the truth have envelopes, and does the net respond there?

python synth_analyze.py OUT_DIR  → prints tables, writes synth_report.npz
Both datasets use the photutils sources of detect.py (galaxies only), the same r/r_Kron and host-mag bins.
"""
import json
import sys

import numpy as np
from scipy import ndimage

from common import block_factors, kernels
from euclid_polish.eval import knee_attribution as ka


MODELS = ("member_238", "member_249")
BAND = ["VIS", "Y", "J", "H"]
TRIM = 10
R_EDGES = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 9.1])
MAG_EDGES = np.array([0, 18, 19.5, 21, 22.5, 24, 99])
SNR_EDGES = np.array([-np.inf, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 1e3, np.inf])
LO, HI = 40, 215
factors = block_factors()


def maps(s, i, shape=(235, 235)):
    sel = np.flatnonzero(s["tile"] == i)
    h, w = shape
    best = np.full(shape, np.inf)
    star = np.zeros(shape, bool)
    mag = np.full(shape, np.nan)
    yy, xx = np.mgrid[0:h, 0:w]
    for j in sel[np.argsort(s["mag"][sel])]:
        kron = np.clip(s["kron_px"][j], 1.0, 25.0)
        reach = int(np.ceil(4 * kron + 2))
        x, y = s["x"][j] - TRIM, s["y"][j] - TRIM
        x0, x1 = max(0, int(x) - reach), min(w, int(x) + reach + 2)
        y0, y1 = max(0, int(y) - reach), min(h, int(y) + reach + 2)
        if x0 >= x1 or y0 >= y1:
            continue
        dx, dy = xx[y0:y1, x0:x1] - x, yy[y0:y1, x0:x1] - y
        th = s["theta"][j]
        u = dx * np.cos(th) + dy * np.sin(th)
        v = -dx * np.sin(th) + dy * np.cos(th)
        rn = np.sqrt(u ** 2 + (v / max(1 - s["ellipticity"][j], 0.05)) ** 2) / kron
        better = rn < best[y0:y1, x0:x1]
        best[y0:y1, x0:x1][better] = rn[better]
        star[y0:y1, x0:x1][better] = s["star"][j]
        mag[y0:y1, x0:x1][better] = s["mag"][j]
    return np.minimum(best, 9.0), star, mag


def snr_of(img, lr_full):
    """3×3-mean S/N in the white-noise convention used for the real tables (snr_smooth)."""
    sky = ka.noise_sigma(lr_full) / factors[:, 0]
    level = np.sqrt(sky ** 2 + np.clip(img, 0, None))
    return ndimage.uniform_filter(img, (3, 3, 1)) / (factors[:, 0] * level / 3.0), level


def z4_of(diff, level2):
    hb = 232
    b = lambda v: v[:hb, :hb].reshape(58, 4, 58, 4, 4).mean((1, 3))
    z = np.full(diff.shape, np.nan)
    z[:hb, :hb] = np.repeat(np.repeat(b(diff) / (factors[:, 2] * np.sqrt(b(level2))), 4, 0), 4, 1)
    return z


def bins(rn, mag):
    return (np.clip(np.digitize(mag, MAG_EDGES) - 1, 0, len(MAG_EDGES) - 2),
            np.clip(np.digitize(rn, R_EDGES) - 1, 0, len(R_EDGES) - 2))


