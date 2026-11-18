"""Stars become points: a saturated Q1 star (archive field 066) in Euclid and POLISHed. The spikes, halo and
blanked core are the telescope; the network deconvolves them and returns the flux as a point (VIS: 21.7M e-
in the input aperture, 23.9M e- POLISHed). One log-like asinh stretch for both."""
import os
import sys

sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish/paper_figures/generation_collages")
sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish")
os.chdir("/Users/alarion239/Desktop/EuclidPolish")
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from build import calibrated, intensity, lupton, real_field

from euclid_polish.eval.ensemble_infer import load_eval_ensemble, sr_from_model
from style import save_panel

c = np.asarray(real_field("066"), np.float64)
o0, o1 = (c.shape[0] - 255) // 2, (c.shape[1] - 255) // 2
lr = c[o0:o0 + 255, o1:o1 + 255]
sr = np.asarray(sr_from_model(load_eval_ensemble(log=lambda *a: None), lr.astype(np.float32))[1], np.float64)
v = np.nan_to_num(lr[..., 0])
y, x = np.unravel_index(np.argmax(v), v.shape)
H = 60                                                         # 12" crop (LR px)
y, x = int(np.clip(y, H, 255 - H)), int(np.clip(x, H, 255 - H))
L = np.repeat(np.repeat(lr[y - H:y + H, x - H:x + H], 2, 0), 2, 1) / 4.0
S = sr[2 * (y - H):2 * (y + H), 2 * (x - H):2 * (x + H)]
cl, cs = calibrated(L), calibrated(S)
white = np.percentile(intensity(cl), 99.5)
legend = ("VIS", "Y+J", "H")
save_panel(lupton(cl, white=white, q=300), "STAR_1_euclid.png", label="Euclid", legend=legend, interp="nearest")
save_panel(lupton(cs, white=white, q=300), "STAR_2_polished.png", label="POLISHed", legend=legend, bar_arcsec=2.0)
print("ok", y, x)
