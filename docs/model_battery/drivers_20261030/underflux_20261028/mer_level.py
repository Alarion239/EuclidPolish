"""Aperture noise vs official RMS at growing block size: real Q1 MER sky vs the shipped generator."""
import glob
import math

import numpy as np
from scipy import ndimage

from euclid_polish.config import Config
from euclid_polish.sky.observation.noise import dithered_unit_noise

A = "data/population_comparison/noise_assessment/arrays"
KS = (1, 2, 4, 8, 16, 32)
BANDS = {"VIS": "VIS", "Y": "Y_E", "J": "J_E", "H": "H_E"}


def block_sigma(z, m, k):
    h = (z.shape[0] // k) * k
    bs = z[:h, :h].reshape(h // k, k, h // k, k).sum((1, 3))
    full = m[:h, :h].reshape(h // k, k, h // k, k).sum((1, 3)) == k * k
    return np.std(bs[full]) / k if full.sum() >= 12 else np.nan, int(full.sum())


print("σ of k×k block SUM / k, in units of the official RMS   (white noise of level RMS would give 1.0 at every k)")
for b, name in BANDS.items():
    real = []
    for f in sorted(glob.glob(f"{A}/pointing_*_{b}_mer.npz")):
        d = np.load(f, allow_pickle=True)
        sci, rms, valid = d["science"], d["official_rms"], d["valid"]
        sky = valid & np.isfinite(sci) & np.isfinite(rms) & (rms < 1e5)
        lm = d["legacy_mask"]
        sky &= ~lm if lm.mean() < 0.5 else lm
        if sky.mean() < 0.3:
            continue
        s0 = 1.4826 * np.median(np.abs(sci[sky] - np.median(sci[sky])))
        src = ndimage.binary_dilation(ndimage.uniform_filter(np.where(sky, sci, 0), 5) > 2 * s0 / 5 * 1.5, iterations=4)
        sky &= ~src & (np.abs(sci - np.median(sci[sky])) < 5 * s0)
        z = np.where(sky, sci - sci[sky].mean(), 0.0)
        r = np.median(rms[sky])
        real.append([block_sigma(z, sky, k)[0] / r for k in KS])
    real = np.array(real)
    band = Config.get_band(name)
    u = dithered_unit_noise((1024, 1024), band, np.random.default_rng(0))
    gen = [block_sigma(u, np.ones(u.shape, bool), k)[0] for k in KS]
    print(f"{b:<3} real " + " ".join(f"k{k}:{v:5.3f}" for k, v in zip(KS, np.nanmedian(real, 0))) + f"   (n={len(real)})")
    print(f"{b:<3} gen  " + " ".join(f"k{k}:{v:5.3f}" for k, v in zip(KS, gen)))
