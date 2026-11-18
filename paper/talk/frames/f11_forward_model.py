"""F11: how a TNG truth becomes a Euclid-like image, stage by stage, on one synthetic test field.

The real forward model (ObservationSimulator with the empirical ePSFs) runs three times with the same
seed: no noise (PSF + pixels), noise without artifacts, and the full model. Writes identically framed
colour squares: 0 truth, 1 PSF + pixels, 2 + noise, 3 + artifacts (the training input)."""
import sys

sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish")
sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish/output/jwst_triples_257/code")
import os
import warnings

warnings.filterwarnings("ignore")
os.chdir("/Users/alarion239/Desktop/EuclidPolish")
import dataclasses

import numpy as np

from euclid_polish.image.tfio import read_images
from euclid_polish.psf.psf_library import load_all_band_psf_sets
from euclid_polish.sky.observation.observation_simulator import (ObservationSimulator,
                                                                 ObservationSimulatorConfig)
from scipy.ndimage import center_of_mass, label

from render import BAL
from triples_lib import calibrated
from style import save_panel

IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 29
CROP = int(sys.argv[2]) if len(sys.argv) > 2 else 254                                      # half-size in HR px (16" of the 25.5" field)
Q = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0

hr = read_images("data/records_v6_test/clean_test.tfrecord", num_images=IDX + 1)[IDX]
psf_sets = load_all_band_psf_sets(target_pixel_scale=0.05)
base = ObservationSimulatorConfig()
stages = {}
for name, kw in (("psf", dict(add_noise=False)), ("noise", dict(add_artifacts=False)), ("full", {})):
    sim = ObservationSimulator(psf_sets_by_band=psf_sets, config=dataclasses.replace(base, **kw))
    stages[name] = np.asarray(sim.apply(hr, rng=np.random.default_rng(7)).data, np.float64)
truth = np.asarray(hr.data, np.float64)
print("truth", truth.shape, {k: v.shape for k, v in stages.items()})

BANDS = ("VIS", "Y_E", "J_E", "H_E")


def rgb(cube, ref_white=None, q=20.0):
    """The triples' colour: solar-SED calibration, the Euclid white balance, Lupton asinh."""
    cal = calibrated(cube) / BAL["sr"]
    r, g, b = cal[..., 3], 0.5 * (cal[..., 1] + cal[..., 2]), cal[..., 0]
    i = (r + g + b) / 3
    white = ref_white if ref_white is not None else np.percentile(i, 99.8)
    soft = white / q
    f = np.arcsinh(np.maximum(i, 0) / soft) / np.arcsinh(white / soft)
    s = np.where(i > 0, f / np.maximum(i, 1e-30), 0)
    out = np.clip(np.stack([r, g, b], -1) * s[..., None], 0, None)
    out = np.clip(out / np.maximum(out.max(-1, keepdims=True), 1), 0, 1)
    lum = out.mean(-1, keepdims=True)
    return np.clip(lum + 1.25 * (out - lum), 0, 1), white


n = truth.shape[0]
v = truth[..., 0]
lab, _ = label(v > np.percentile(v, 50) + 5 * np.std(v[v < np.percentile(v, 90)]))
big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
cy, cx = (int(round(x)) for x in center_of_mass(lab == big))
cy = int(np.clip(cy, CROP, n - CROP)) // 2 * 2
cx = int(np.clip(cx, CROP, n - CROP)) // 2 * 2
t_crop = truth[cy - CROP:cy + CROP, cx - CROP:cx + CROP]
lr_up = {k: np.repeat(np.repeat(v, 2, 0), 2, 1) / 4.0 for k, v in stages.items()}   # native px, flux kept
crops = {k: v[cy - CROP:cy + CROP, cx - CROP:cx + CROP] for k, v in lr_up.items()}
img_t, white = rgb(t_crop, q=Q)
legend = ("VIS", "Y+J", "H")
save_panel(img_t, "F11_0_truth.png", label="TNG truth", legend=legend, bar_arcsec=2.0)
names = {"psf": "+ PSF + pixels", "noise": "+ noise", "full": "+ artifacts"}
for k, (key, lab) in enumerate(names.items(), 1):
    img, _ = rgb(crops[key], ref_white=white, q=Q)
    save_panel(img, f"F11_{k}_{key}.png", label=lab, legend=legend, interp="nearest")
print("ok")
