"""F10: what we don't trust in the training truth, shown plainly (deep VIS stretch, full 160 kpc frames
rendered on FASRC by remote/tng_artifacts.py). Separate squares; the fix for each goes in the Prezi text."""
import os

import numpy as np
from PIL import Image

from style import TALK, save_panel

src = os.path.join(TALK, "data", "tng_artifacts")


def load(name, crop=None):
    im = np.asarray(Image.open(os.path.join(src, name)).convert("L"), np.float64) / 255.0
    if crop:
        x0, y0, s = crop
        im = im[y0:y0 + s, x0:x0 + s]
    im = im[::-1]                                     # PNGs are top-down; panels are origin="lower"
    return np.repeat(im[..., None], 3, -1)


save_panel(load("117266_O1.png"), "F10_a_dust_box_117266.png", label=None)  # 11-25: captions under the panels in Prezi say it
save_panel(load("229934_O1.png"), "F10_b_box_cut_229934.png", label=None)
save_panel(load("580250_O1.png"), "F10_c_slabs_580250.png", label=None)
save_panel(load("229934_O1.png", crop=(40, 40, 260)), "F10_d_particle_grain_229934.png",
           label=None, interp="nearest")
print("ok")
