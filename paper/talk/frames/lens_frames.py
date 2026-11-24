"""Lens and showcase frames from fresh Q1 cutouts super-resolved by production model 257 (infer_q1.py):
the grade-A complete Einstein ring 102042919, NGC 6505's Einstein ring, the Hubble Ultra Deep Field area and
the Cat's Eye Nebula. Euclid / POLISHed squares, one stretch per pair. Realistic colour since 11-23: the viewer's
AB balance (ab_colour.py), sky from the cutout border; clean labels (instrument name only). The panels used in the
talk are also written to prezi_upload/v13/ under their Prezi image keys."""
import os
import sys

sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish/paper_figures/generation_collages")
sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish")
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from PIL import Image

from ab_colour import ab_rgb, auto_levels, noise_sigma
from style import TALK, save_panel

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


UP = os.path.join(TALK, "prezi_upload", "v13")
KEYS = {"LENS_ring102042919": ("A11_lens_ring_Euclid", "K01_lens_ring_POLISHed"),
        "LENS_NGC6505": ("A12_NGC6505_ring_Euclid", "K02_NGC6505_ring_POLISHed_optional")}


def border_sky(cube, ref=None):
    """Sky from the cutout border (sigma-clipped), not the whole cutout: a galaxy that fills the cutout
    (NGC 6505) would otherwise be subtracted as sky, worst in NISP, and turn the ring magenta."""
    ref = cube if ref is None else ref
    out = cube.astype(np.float64).copy()
    n = ref.shape[0]
    b = max(4, n // 20)
    for k in range(cube.shape[-1]):
        plane = ref[..., k]
        edge = np.concatenate([plane[:b].ravel(), plane[-b:].ravel(), plane[:, :b].ravel(), plane[:, -b:].ravel()])
        _m, med, _s = sigma_clipped_stats(edge[np.isfinite(edge)], sigma=2.5, maxiters=8)
        out[..., k] -= med * (cube.shape[0] / ref.shape[0]) ** -2
    return out


def load(folder):
    lr = np.moveaxis(np.asarray(fits.getdata(os.path.join(ROOT, folder, "original_stack.fits")), np.float64), 0, -1)
    sr = np.moveaxis(np.asarray(fits.getdata(os.path.join(ROOT, folder, "SR.fits")), np.float64), 0, -1)
    return lr, sr


os.makedirs(UP, exist_ok=True)
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
    cl = border_sky(up, ref=np.repeat(np.repeat(lr, 2, 0), 2, 1) / 4.0)
    cs = border_sky(S, ref=sr)
    core = L[..., 2][L.shape[0] // 4:-L.shape[0] // 4, L.shape[1] // 4:-L.shape[1] // 4]
    mode = "noJ" if (core == 0).mean() > 0.01 else "full"   # NGC 6505's J mosaic is blank on the ring: G = Y
    lev = auto_levels(cs, pct=pct, q=max(10.0, q / 2), mode=mode, sigma=noise_sigma(cl, mode))
    paths = [save_panel(ab_rgb(cl, mode=mode, **lev), f"{name}_1_euclid.png", label="Euclid", interp="nearest"),
             save_panel(ab_rgb(cs, mode=mode, **lev), f"{name}_2_polished.png", label="POLISHed")]
    for path, key in zip(paths, KEYS.get(name, ())):
        Image.open(path).convert("RGB").resize((1600, 1600), Image.LANCZOS).save(os.path.join(UP, key + ".jpg"),
                                                                                quality=90)
    print(name, "ok", L.shape, mode)
