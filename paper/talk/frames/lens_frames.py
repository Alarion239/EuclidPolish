"""Lens and showcase frames from fresh Q1 cutouts super-resolved by production model 257 (infer_q1.py):
the grade-A complete Einstein ring 102042919, NGC 6505's Einstein ring, the Hubble Ultra Deep Field area and
the Cat's Eye Nebula. Euclid / POLISHed squares in the triples' colour (sky-subtracted, solar-calibrated,
Euclid white balance, Lupton asinh), one stretch per pair."""
import os
import sys

sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish/paper_figures/generation_collages")
sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish")
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from astropy.io import fits
from build import BALANCE, BANDS, intensity, lupton

from euclid_polish.visualization import color

from style import save_panel

ROOT = "/Users/alarion239/Desktop/EuclidPolish/data/euclid_inference/talk"
# name: (folder, crop half-size in LR px or None for the whole cutout, q, white percentile, bar, label)
TARGETS = {
    "LENS_ring102042919": ("ring102042919", 50, 20, 99.9, 1.0, "Q1 lens, grade A"),
    "LENS_ring102042919_wide": ("ring102042919", None, 30, 99.8, 2.0, "Q1 lens, grade A"),
    "LENS_NGC6505": ("ngc6505", 60, 60, 99.95, 2.0, "NGC 6505"),
    "LENS_NGC6505_wide": ("ngc6505", 160, 80, 99.95, 5.0, "NGC 6505"),
    "HUDF_area": ("hudf", None, 40, 99.7, 10.0, "Ultra Deep Field area"),
    "HUDF_zoom": ("hudf", 128, 30, 99.7, 2.0, "Ultra Deep Field"),
}


def border_calibrated(cube, ref=None):
    """Sky from the cutout border (sigma-clipped), not the whole cutout: a galaxy that fills the cutout
    (NGC 6505) would otherwise be subtracted as sky, worst in NISP, and turn the ring magenta."""
    from astropy.stats import sigma_clipped_stats
    ref = cube if ref is None else ref
    out = cube.astype(np.float64).copy()
    n = ref.shape[0]
    b = max(4, n // 20)
    for k in range(cube.shape[-1]):
        plane = ref[..., k]
        edge = np.concatenate([plane[:b].ravel(), plane[-b:].ravel(), plane[:, :b].ravel(), plane[:, -b:].ravel()])
        _m, med, _s = sigma_clipped_stats(edge[np.isfinite(edge)], sigma=2.5, maxiters=8)
        out[..., k] -= med * (cube.shape[0] / ref.shape[0]) ** -2
    return color.calibrate(np.nan_to_num(out), band_names=BANDS, reference="solar") / BALANCE


def load(folder):
    lr = np.moveaxis(np.asarray(fits.getdata(os.path.join(ROOT, folder, "original_stack.fits")), np.float64), 0, -1)
    sr = np.moveaxis(np.asarray(fits.getdata(os.path.join(ROOT, folder, "SR.fits")), np.float64), 0, -1)
    return lr, sr


legend = ("VIS", "Y+J", "H")
for name, (folder, half, q, pct, bar, label) in TARGETS.items():
    if not os.path.exists(os.path.join(ROOT, folder, "SR.fits")):
        print(name, "not ready"); continue
    lr, sr = load(folder)
    n = lr.shape[0]
    c0, c1 = n // 2, lr.shape[1] // 2
    h = half or min(n, lr.shape[1]) // 2
    L = lr[c0 - h:c0 + h, c1 - h:c1 + h]
    S = sr[2 * (c0 - h):2 * (c0 + h), 2 * (c1 - h):2 * (c1 + h)]
    up = np.repeat(np.repeat(L, 2, 0), 2, 1) / 4.0
    cl = border_calibrated(up, ref=np.repeat(np.repeat(lr, 2, 0), 2, 1) / 4.0)
    cs = border_calibrated(S, ref=sr)
    leg = legend
    core = L[..., 2][L.shape[0] // 4:-L.shape[0] // 4, L.shape[1] // 4:-L.shape[1] // 4]
    if (core == 0).mean() > 0.01:          # a J-band hole (NGC 6505's J mosaic is blank on the ring): G = Y only
        cl[..., 2], cs[..., 2] = cl[..., 1], cs[..., 1]
        leg = ("VIS", "Y", "H")
    white = np.percentile(intensity(cs), pct)
    save_panel(lupton(cl, white=white, q=q), f"{name}_1_euclid.png", label=f"Euclid · {label}", legend=leg,
               interp="nearest", envelope=not name.endswith("_wide"))
    save_panel(lupton(cs, white=white, q=q), f"{name}_2_polished.png", label="POLISHed", legend=leg,
               bar_arcsec=bar)
    print(name, "ok", L.shape)
