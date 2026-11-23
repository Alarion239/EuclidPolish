"""DR1 double-source-plane lens candidates (arXiv:2611.20166, Gold) as talk panels: Euclid DR1 and
POLISHed (production 257, TTA x8) squares, 10" field, colour R=H, G=(Y+J)/2, B=VIS in the in-browser
viewer's absolute scheme (output/lens_dr1_dspl/viewer_scheme.py), SR x4 for equal surface brightness."""
import csv
import os
import sys

import numpy as np
from astropy.io import fits

D = "/Users/alarion239/Desktop/EuclidPolish/output/lens_dr1_dspl"
sys.path.insert(0, D)
import viewer_scheme as vs  # noqa: E402

from style import save_panel  # noqa: E402

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
    a = vs.colour(crop(lr, n)); b = vs.colour(crop(sr, 2 * n) * 4)
    name = f"J{r['name'][1:5]}{r['name'][11:16]}"
    save_panel(a, f"DR1L_G{k:02d}_1_euclid.png", label=f"Euclid DR1 · lens G{k}", legend=("VIS", "Y+J", "H"),
               interp="nearest", coords=name)
    save_panel(b, f"DR1L_G{k:02d}_2_polished.png", label="POLISHed", legend=("VIS", "Y+J", "H"), bar_arcsec=2.0)
    print(k, name, a.shape, b.shape)
