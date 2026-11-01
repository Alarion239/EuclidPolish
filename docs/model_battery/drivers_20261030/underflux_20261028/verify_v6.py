"""End-to-end: shipped apply_archive_noise at each real cutout's RMS vs the cutout, identical masking."""
import glob

import numpy as np

from euclid_polish.config import Config
from euclid_polish.sky.observation.noise import apply_archive_noise
from mer_fair import sky_mask, stats

A = "data/population_comparison/noise_assessment/arrays"
BANDS = {"VIS": "VIS", "Y": "Y_E", "J": "J_E", "H": "H_E"}
print("lag1  diag  lag2 | σ1/RMS σ2 σ4 σ8")
for b, name in BANDS.items():
    band = Config.get_band(name)
    real, sim = [], []
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
        norm = np.r_[1, 1, 1, [r] * 4]
        real.append(stats(sci, sky_mask(sci, base)) / norm)
        u = apply_archive_noise(np.zeros(sci.shape, np.float32), band, rng, sky_rms_e=float(r)).astype(np.float64)
        sim.append(stats(u, sky_mask(u, base)) / norm)
    fmt = lambda a: " ".join(f"{v:5.3f}" for v in np.nanmedian(np.array(a), 0))
    print(f"{b:<3} real {fmt(real)}")
    print(f"{b:<3} v6   {fmt(sim)}")
