"""F14: what did the network learn? Put a cat through it.

A tabby (Cat_November_2010-1a.jpg, Alvesgaspar, Wikimedia Commons, CC BY-SA 3.0) becomes a 51" truth on
the 0.05" grid: photo red/green/blue -> H / Y,J / VIS, set so the talk's colour mapping returns the photo's
colours, peak ~460 e- per pixel (a bright galaxy), soft-edged like a TNG stamp. The real forward model
(empirical ePSFs, Q1 noise and artifacts) makes the Euclid view; production model 257 POLISHes it.
Separate squares: photo, truth, Euclid, POLISHed."""
import dataclasses
import os
import sys

sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish")
sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish/paper_figures/generation_collages")
os.chdir("/Users/alarion239/Desktop/EuclidPolish")
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image as PILImage
from build import BALANCE, BANDS, calibrated, lupton

from euclid_polish.eval.ensemble_infer import load_eval_ensemble, sr_from_model
from euclid_polish.image.core import Image
from euclid_polish.psf.psf_library import load_all_band_psf_sets
from euclid_polish.sky.observation.observation_simulator import (ObservationSimulator,
                                                                 ObservationSimulatorConfig)
from euclid_polish.visualization import color
from style import TALK, save_panel

N = 1020                                              # 51" at 0.05"
photo = PILImage.open(os.path.join(TALK, "data", "cat", "cat_alvesgaspar.jpg")).convert("RGB")
w, h = photo.size
side = w
photo = photo.crop((0, 0, side, side)).resize((N, N), PILImage.LANCZOS)    # top square: face + body
rgb = np.asarray(photo, np.float64)[::-1] / 255.0                         # origin lower
yy, xx = np.mgrid[0:N, 0:N] / (N - 1) * 2 - 1
r = np.maximum(np.abs(xx), np.abs(yy))
window = np.clip((0.97 - r) / 0.12, 0, 1) ** 2                          # soft square edge
rgb = rgb ** 2.2 * window[..., None]                                   # to linear light

# electrons that the talk's colour mapping (calibrated()/BALANCE, R=H, G=(Y+J)/2, B=VIS) maps back
gain = color.calibrate(np.ones((1, 1, 4)), band_names=BANDS, reference="solar")[0, 0]
want = np.stack([rgb[..., 2], rgb[..., 1], rgb[..., 1], rgb[..., 0]], -1)  # VIS, Y, J, H
cube = want * BALANCE / gain
cube *= 460.0 / np.percentile(cube[..., 0], 99.9)

import os.path as _p
cache = _p.join(TALK, "data", "cat", "cat_cubes.npz")
if _p.exists(cache) and "--rerun" not in sys.argv:
    with np.load(cache) as d:
        cube, lr, sr = (np.asarray(d[k], np.float64) for k in ("truth", "lr", "sr"))
else:
    psf_sets = load_all_band_psf_sets(target_pixel_scale=0.05)
    sim = ObservationSimulator(psf_sets_by_band=psf_sets, config=ObservationSimulatorConfig())
    hr = Image(cube.astype(np.float32), 0.05, tuple(BANDS), True)
    lr = np.asarray(sim.apply(hr, rng=np.random.default_rng(3)).data, np.float64)
    model = load_eval_ensemble(log=print)
    sr = np.asarray(sr_from_model(model, lr.astype(np.float32))[1], np.float64)
    np.savez_compressed(cache, truth=cube.astype(np.float32), lr=lr.astype(np.float32),
                        sr=sr.astype(np.float32))

up = np.repeat(np.repeat(lr, 2, 0), 2, 1) / 4.0


def display(c, white):
    """Undo the photo preparation: the colour mapping's channels, linear, then gamma 1/2.2."""
    cal = color.calibrate(np.nan_to_num(c), band_names=BANDS, reference="solar") / BALANCE
    rgb = np.stack([cal[..., 3], 0.5 * (cal[..., 1] + cal[..., 2]), cal[..., 0]], -1)
    return np.clip(np.clip(rgb / white, 0, 1) ** (1 / 2.2), 0, 1)


cal_t = color.calibrate(cube, band_names=BANDS, reference="solar") / BALANCE
white = np.percentile(cal_t.max(-1), 99.9)
save_panel(np.asarray(photo, np.float64)[::-1] / 255.0, "F14_cat_0_photo.png", label="Photo")
save_panel(display(cube, white), "F14_cat_1_truth.png", label="Truth", bar_arcsec=5.0)
save_panel(display(up, white), "F14_cat_2_euclid.png", label="Euclid", interp="nearest")
save_panel(display(sr, white), "F14_cat_3_polished.png", label="POLISHed", bar_arcsec=5.0)
print("ok")
