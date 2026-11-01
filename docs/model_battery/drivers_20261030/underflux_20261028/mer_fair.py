"""Fair comparison: identical sky masking on real Q1 MER cutouts and on simulated noise (× the cutout's RMS).

Stats: lag-1 (axis mean), diag, lag-2, block-sum σ/k for k=1,2,4,8 in RMS units.
Sims: shipped generator (mean of 4) and median-of-n stacks, all on the generator's large-scale normalisation.
"""
import glob
import math

import numpy as np
from scipy import ndimage

from euclid_polish.config import Config
from euclid_polish.sky.observation.noise import _detector_factor, _shifted_bilinear, dithered_unit_noise

A = "data/population_comparison/noise_assessment/arrays"
BANDS = {"VIS": "VIS", "Y": "Y_E", "J": "J_E", "H": "H_E"}
KS = (1, 2, 4, 8)


def sky_mask(x, base):
    """base & not near a source: 5×5 mean above 3σ of the 5×5-mean map's own (MAD) scatter."""
    sm = ndimage.uniform_filter(np.where(base, x, 0.0), 5)
    v = sm[base]
    s = 1.4826 * np.median(np.abs(v - np.median(v)))
    src = ndimage.binary_dilation(sm - np.median(v) > 3 * s, iterations=4)
    return base & ~src


def stats(x, sky):
    z = np.where(sky, x - x[sky].mean(), 0.0)
    m = sky.astype(float)
    var = (z ** 2).sum() / m.sum()
    h, w = x.shape
    def ac(dy, dx):
        return (z[:h - dy, :w - dx] * z[dy:, dx:]).sum() / (m[:h - dy, :w - dx] * m[dy:, dx:]).sum() / var
    out = [(ac(0, 1) + ac(1, 0)) / 2, ac(1, 1), (ac(0, 2) + ac(2, 0)) / 2]
    for k in KS:
        hh = (h // k) * k
        bs = z[:hh, :hh].reshape(hh // k, k, hh // k, k).sum((1, 3))
        full = m[:hh, :hh].reshape(hh // k, k, hh // k, k).sum((1, 3)) == k * k
        out.append(np.std(bs[full]) / k if full.sum() >= 12 else np.nan)
    return np.array(out)


def stack(band, rng, n, combine, shape=(256, 256)):
    factor = _detector_factor(band)
    pad = 2
    det = (math.ceil(shape[0] / factor) + 2 * pad, math.ceil(shape[1] / factor) + 2 * pad)
    oy = (rng.permutation(n) + rng.random(n)) / n
    ox = (rng.permutation(n) + rng.random(n)) / n
    L = np.stack([_shifted_bilinear(rng.standard_normal(det), factor, float(a), float(c), shape, pad)
                  for a, c in zip(oy, ox)])
    return (L.mean(0) if combine == "mean" else np.median(L, 0)) * math.sqrt(n) / factor


if __name__ == "__main__":
    hdr = "lag1  diag  lag2 | σ1/RMS σ2 σ4 σ8"
    print(hdr)
    for b, name in BANDS.items():
        band = Config.get_band(name)
        rows, sims = [], {key: [] for key in ("gen mean4", "median3", "median5", "median8")}
        rng = np.random.default_rng(0)
        for f in sorted(glob.glob(f"{A}/pointing_*_{b}_mer.npz")):
            d = np.load(f, allow_pickle=True)
            sci, rms, valid = d["science"], d["official_rms"], d["valid"]
            base = valid & np.isfinite(sci) & np.isfinite(rms) & (rms < 1e5)
            lm = d["legacy_mask"]
            base &= ~lm if lm.mean() < 0.5 else lm
            if base.mean() < 0.3:
                continue
            r = np.median(rms[base])
            rows.append(stats(sci, sky_mask(sci, base)) / np.r_[1, 1, 1, [r] * len(KS)])
            for key in sims:
                u = (dithered_unit_noise((256, 256), band, rng) if key == "gen mean4"
                     else stack(band, rng, int(key[-1]), "median"))
                sims[key].append(stats(u, sky_mask(u, base)))   # same footprint, same mask rule
        fmt = lambda a: " ".join(f"{v:5.3f}" for v in np.nanmedian(np.array(a), 0))
        print(f"{b:<3} real Q1 MER  {fmt(rows)}   (n={len(rows)})")
        for key, v in sims.items():
            print(f"{b:<3} {key:<11}  {fmt(v)}")
