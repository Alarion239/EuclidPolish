"""DR1 double-source-plane lens candidates (arXiv:2611.20166, Gold) as talk panels: Euclid DR1 and
POLISHed (production 257, TTA x8) squares, 10" field. Realistic colour since 11-23: the viewer's AB balance
(ab_colour.py), one stretch per pair set on the POLISHed panel (SR x4 for equal surface brightness); clean labels.
Also written to prezi_upload/v13/ under their Prezi image keys (DR1_G<k>_E / _P)."""
import csv
import os
import numpy as np
from astropy.io import fits

from PIL import Image

from ab_colour import ab_rgb, auto_levels, noise_sigma
from style import TALK, save_panel

D = "/Users/alarion239/Desktop/EuclidPolish/output/lens_dr1_dspl"
UP = os.path.join(TALK, "prezi_upload", "v13")
os.makedirs(UP, exist_ok=True)

PICK = [2, 3, 4, 5, 6, 11]
FOV = 10.0
rows = [r for r in csv.DictReader(open(os.path.join(D, "lenses.csv"))) if r["tier"] == "Gold"]


def crop(a, side):
    h, w = a.shape[:2]
    y0, x0 = (h - side) // 2, (w - side) // 2
    return a[y0:y0 + side, x0:x0 + side]


for r in rows:
    k = int(r["no"])
    if k not in PICK:
        continue
    f = os.path.join(D, f"Gold_{k:02d}_{r['name']}.fits")
    with fits.open(f) as h:
        lr = np.moveaxis(np.asarray(h[0].data, np.float64), 0, -1)
        sr = np.moveaxis(np.asarray(h["SR"].data, np.float64), 0, -1)
    n = int(round(FOV / 0.1))
    L, S = crop(lr, n), crop(sr, 2 * n) * 4
    lev = auto_levels(S, pct=99.9, q=12, sigma=noise_sigma(L))
    for cube, part, label, key, interp in ((L, "1_euclid", "Euclid", "E", "nearest"),
                                           (S, "2_polished", "POLISHed", "P", "lanczos")):
        path = save_panel(ab_rgb(cube, **lev), f"DR1L_G{k:02d}_{part}.png", label=label, interp=interp)
        Image.open(path).convert("RGB").resize((1600, 1600), Image.LANCZOS).save(
            os.path.join(UP, f"DR1_G{k}_{key}.jpg"), quality=90)
    print(k, r["name"], {a: round(float(b), 1) for a, b in lev.items()})
