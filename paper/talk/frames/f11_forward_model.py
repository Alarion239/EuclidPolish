"""F11: how a TNG truth becomes a Euclid-like image, stage by stage, on one synthetic test field.

The real forward model (ObservationSimulator with the empirical ePSFs) runs three times with the same
seed: no noise (PSF + pixels), noise without artifacts, and the full model. Writes identically framed
colour squares: 0 truth, 1 PSF + pixels, 2 + noise, 3 + artifacts (the training input). Realistic colour since
11-23: the viewer's AB balance (ab_colour.py), one stretch set on the truth, black point at 2 sigma of the full
model's sky; clean labels. Also written to prezi_upload/v13/ under their Prezi keys (D01..D03)."""
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

from PIL import Image

from ab_colour import ab_rgb, auto_levels, noise_sigma
from style import TALK, save_panel

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
lev = auto_levels(t_crop, q=Q / 2, black_sigma=2.0, sigma=noise_sigma(crops["full"]))
UP = os.path.join(TALK, "prezi_upload", "v13")
os.makedirs(UP, exist_ok=True)
KEYS = {"truth": "D01_forward_truth", "psf": "D02_forward_PSF_pixels", "full": "D03_forward_noise_artifacts"}


def out(path, key):
    if key in KEYS:
        Image.open(path).convert("RGB").resize((1600, 1600), Image.LANCZOS).save(os.path.join(UP, KEYS[key] + ".jpg"),
                                                                                quality=90)


# 11-25: no in-picture labels; each panel has its own caption under it in Prezi
out(save_panel(ab_rgb(t_crop, **lev), "F11_0_truth.png", label=None), "truth")
names = {"psf": "+ PSF + pixels", "noise": "+ noise", "full": "+ artifacts"}
for k, (key, lab) in enumerate(names.items(), 1):
    out(save_panel(ab_rgb(crops[key], **lev), f"F11_{k}_{key}.png", label=None, interp="nearest"), key)
print("ok")
