"""Real MER: anisotropy + large-scale/pixel variance; simulation with rotated detector grids, mean vs median."""
import glob
import math

import numpy as np
from scipy import ndimage

A = "data/population_comparison/noise_assessment/arrays"
BANDS = ["VIS", "Y", "J", "H"]
LAGS = ((0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0))


def texture(x, sky):
    v = x[sky]
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))
    ok = sky & (np.abs(x - med) < 5 * mad)
    z = np.where(ok, x - x[ok].mean(), 0.0)
    m = ok.astype(float)
    var = (z ** 2).sum() / m.sum()
    h, w = x.shape
    out = []
    for dy, dx in LAGS:
        if dx >= 0:
            a, b = z[:h - dy, :w - dx] * z[dy:, dx:], m[:h - dy, :w - dx] * m[dy:, dx:]
        else:
            a, b = z[:h - dy, -dx:] * z[dy:, :w + dx], m[:h - dy, -dx:] * m[dy:, :w + dx]
        out.append(a.sum() / b.sum() / var)
    blk = []
    for k in (4, 8):
        hh = (h // k) * k
        bs = z[:hh, :hh].reshape(hh // k, k, hh // k, k).sum((1, 3))
        bm = m[:hh, :hh].reshape(hh // k, k, hh // k, k).sum((1, 3)) == k * k
        blk.append(np.var(bs[bm]) / (k * k * var) if bm.sum() > 15 else np.nan)   # aperture variance / white expectation
    return np.array(out + blk), math.sqrt(var)


print("columns: lag(0,1) (1,0) (1,1) (1,-1) (0,2) (2,0) | ap4 ap8 (k×k sum var / k²·pixel var) | σ_large/RMS (8×8)")
for b in BANDS:
    rows, lev = [], []
    for f in sorted(glob.glob(f"{A}/pointing_*_{b}_mer.npz")) + sorted(glob.glob(f"{A}/example_*_{b}_mer.npz")):
        d = np.load(f, allow_pickle=True)
        sci, rms, valid = d["science"], d["official_rms"], d["valid"]
        sky = valid & np.isfinite(sci) & np.isfinite(rms) & (rms < 1e5)
        lm = d["legacy_mask"]
        sky &= ~lm if lm.mean() < 0.5 else lm
        if sky.mean() < 0.3:
            continue
        t, s = texture(sci, sky)
        src = ndimage.binary_dilation(ndimage.uniform_filter(np.where(sky, sci, 0), 5) > 2 * s / 5 * 1.5, iterations=4)
        t, s = texture(sci, sky & ~src)
        rows.append(t)
        lev.append(s * math.sqrt(t[-1]) / np.median(rms[sky & ~src]))
    r = np.nanmedian(np.array(rows), 0)
    print(f"real {b:<3} " + " ".join(f"{v:6.3f}" for v in r) + f" | {np.nanmedian(lev):.3f}")


def sim(factor, theta_deg, n, combine, rng, shape=(192, 192)):
    """n exposures of unit white detector noise, detector rotated by θ and shifted, bilinear onto archive grid."""
    th = math.radians(theta_deg)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    cy, cx = shape[0] / 2, shape[1] / 2
    layers = []
    for _ in range(n):
        oy, ox = rng.random(2)
        # archive → detector coordinates (detector pixel = factor archive px)
        u = ((xx - cx) * math.cos(th) + (yy - cy) * math.sin(th)) / factor
        v = (-(xx - cx) * math.sin(th) + (yy - cy) * math.cos(th)) / factor
        det = rng.standard_normal((int(shape[0] * 1.5 / factor) + 8,) * 2)
        c = det.shape[0] / 2
        layers.append(ndimage.map_coordinates(det, [v + c + oy, u + c + ox], order=1, mode="wrap"))
    L = np.stack(layers)
    return L.mean(0) if combine == "mean" else np.median(L, 0)


rng = np.random.default_rng(3)
for name, factor in (("VIS", 1), ("NISP", 3)):
    for combine in ("mean", "median"):
        for theta in (0, 10, 20, 30, 45):
            res = np.nanmean([texture(sim(factor, theta, 4, combine, rng), np.ones((192, 192), bool))[0]
                              for _ in range(6)], 0)
            print(f"sim {name:<4} n=4 {combine:<6} θ={theta:>2}° " + " ".join(f"{v:6.3f}" for v in res))
