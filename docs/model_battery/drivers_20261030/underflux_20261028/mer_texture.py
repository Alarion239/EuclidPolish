"""Real MER cutouts (noise_assessment arrays): pixel scatter / official RMS, lag-1/2 correlation of empty sky,
vs the generator's dithered noise stacked by MEAN (as coded) and by MEDIAN (as the MER header says)."""
import glob
import math

import numpy as np
from scipy import ndimage

from euclid_polish.config import Config
from euclid_polish.sky.observation.noise import _detector_factor, _shifted_bilinear, dithered_unit_noise

A = "data/population_comparison/noise_assessment/arrays"
BANDS = ["VIS", "Y", "J", "H"]
CFG = {"VIS": Config.get_band("VIS"), "Y": Config.get_band("Y_E"), "J": Config.get_band("J_E"), "H": Config.get_band("H_E")}


def stats(x, sky):
    v = np.where(sky, x, np.nan)
    med = np.nanmedian(v)
    mad = 1.4826 * np.nanmedian(np.abs(v - med))
    clip = sky & (np.abs(x - med) < 5 * mad)
    z = np.where(clip, x - np.nanmean(np.where(clip, x, np.nan)), 0.0)
    m = clip.astype(float)
    var = (z ** 2).sum() / m.sum()
    def ac(dy, dx):
        h, w = x.shape
        return ((z[:h - dy, :w - dx] * z[dy:, dx:]).sum() / (m[:h - dy, :w - dx] * m[dy:, dx:]).sum()) / var
    return math.sqrt(var), mad, (ac(0, 1) + ac(1, 0)) / 2, (ac(0, 2) + ac(2, 0)) / 2, (ac(1, 1) + 0) / 1


print("== real MER cutouts (empty sky: valid, not legacy-masked, no >2σ smoothed sources)")
for b in BANDS:
    rows = []
    for f in sorted(glob.glob(f"{A}/pointing_*_{b}_mer.npz")) + sorted(glob.glob(f"{A}/example_*_{b}_mer.npz")):
        d = np.load(f, allow_pickle=True)
        sci, rms, valid = d["science"], d["official_rms"], d["valid"]
        sky0 = valid & np.isfinite(sci) & np.isfinite(rms) & (rms < 1e5)
        if "legacy_mask" in d.files:
            lm = d["legacy_mask"]
            sky0 &= ~lm if lm.mean() < 0.5 else lm
        if sky0.mean() < 0.3:
            continue
        s, mad, l1, l2, d11 = stats(sci, sky0)
        src = ndimage.binary_dilation(ndimage.uniform_filter(np.where(sky0, sci, 0), 5) > 2 * s / 5 * 1.5, iterations=4)
        sky = sky0 & ~src
        s, mad, l1, l2, d11 = stats(sci, sky)
        rows.append((s / np.median(rms[sky]), mad / np.median(rms[sky]), l1, l2, d11, np.median(rms[sky])))
    r = np.array(rows)
    print(f"  {b:<3} n={len(r):2d}  σ/RMS {np.median(r[:,0]):.3f}  MAD/RMS {np.median(r[:,1]):.3f}  lag1 {np.median(r[:,2]):.3f} "
          f"[{np.percentile(r[:,2],16):.3f},{np.percentile(r[:,2],84):.3f}]  lag2 {np.median(r[:,3]):.3f}  diag {np.median(r[:,4]):.3f}  RMS {np.median(r[:,5]):.2f}")


def stack(band, rng, n, combine, shape=(256, 256)):
    factor = _detector_factor(band)
    pad = 2
    det = (math.ceil(shape[0] / factor) + 2 * pad, math.ceil(shape[1] / factor) + 2 * pad)
    oy = (rng.permutation(n) + rng.random(n)) / n
    ox = (rng.permutation(n) + rng.random(n)) / n
    layers = np.stack([_shifted_bilinear(rng.standard_normal(det), factor, float(a), float(c), shape, pad)
                       for a, c in zip(oy, ox)])
    out = layers.mean(0) if combine == "mean" else np.median(layers, 0)
    return out * math.sqrt(n) / factor    # mean-stack normalisation of the generator


print("\n== generator texture: per-pixel σ (unit = large-scale σ of a MEAN stack), lag1, lag2, diag")
for b in BANDS:
    band = CFG[b]
    for n in (3, 4, 6):
        for combine in ("mean", "median"):
            rng = np.random.default_rng(0)
            res = [stats(stack(band, rng, n, combine), np.ones((256, 256), bool)) for _ in range(8)]
            r = np.array(res)
            print(f"  {b:<3} n={n} {combine:<6} σ {r[:,0].mean():.3f}  lag1 {r[:,2].mean():.3f}  lag2 {r[:,3].mean():.3f}  diag {r[:,4].mean():.3f}")
    print(f"  {b:<3} dithered_unit_noise (as shipped, n={band.n_exposures}): σ {np.std(dithered_unit_noise((256,256), band, np.random.default_rng(0))):.3f}")
