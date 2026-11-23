"""One image spot that keeps zooming: DSS2 (10') -> Pan-STARRS1 (same 10') -> DECam (4') -> Euclid (102")
-> Hubble XDF (26") -> JWST JADES (12", the HUDF spiral). Each layer covers a region inside the previous
one, so in Prezi: swap to the sharper survey, zoom into the next region, swap again. Reuses hudf_ladder.py
(resampling + colour) with the larger DSS/PS1/DECam cutouts (data/hudf_multi/*big*)."""
import os
import sys

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hudf_ladder as L  # noqa: E402
from style import save_panel  # noqa: E402

D = L.D
GRA, GDEC = 53.17468, -27.799422
cosd = np.cos(np.radians(GDEC))
R = {  # name: (ra, dec, size arcsec)
    "R0": (L.RA0, L.DEC0, 600.0),
    "R1": (L.RA0, L.DEC0, 240.0),
    "R2": (L.RA0, L.DEC0, 102.4),
    "R3": (GRA - 6 / 3600 / cosd, GDEC + 4 / 3600, 26.0),     # nudged inside the Euclid cutout
    "R4": (GRA, GDEC, 12.0),
}


def big_bands(inst):
    if inst == "dss":
        r = L.load("dss2big_red.fits"); return [r, r, r]
    if inst == "ps1":
        return [L.load(f"ps1big_{b}.fits") for b in ("i", "r", "g")]
    if inst == "ls":
        a, w = L.load("lsbig_griz.fits"); return [(a[3], w), (a[1], w), (a[0], w)]
    return L.bands(inst)


LAYERS = [  # file, instrument, region, pixel scale, label, interp
    ("ZL_0_dss", "dss", "R0", 1.0, "DSS2 · 2″", "nearest"),
    ("ZL_1_ps1", "ps1", "R0", 0.25, "Pan-STARRS · 1.1″", "nearest"),
    ("ZL_2_ls", "ls", "R1", 0.262, "DECam · 0.9″", "nearest"),
    ("ZL_3_euclid", "euclid", "R2", 0.1, "Euclid · 0.16″", "nearest"),
    ("ZL_4_hst", "hst", "R3", 0.03, "Hubble · 0.09″", "lanczos"),
    ("ZL_5_jwst", "jwst", "R4", 0.03, "JWST · 0.06″", "lanczos"),
    ("ZL_6_polished", "polished", "R2", 0.05, "POLISHed Euclid", "lanczos"),
]
white_eu = None
for name, inst, reg, scale, label, interp in LAYERS:
    if len(sys.argv) > 1 and inst not in sys.argv[1:]:
        continue
    ra, dec, size = R[reg]
    planes = [L.resample(a, w, ra, dec, scale, size) for a, w in big_bands(inst)]
    if inst == "polished":
        if white_eu is None:
            pe = [L.resample(a, w, *R["R2"][:2], 0.1, R["R2"][2]) for a, w in big_bands("euclid")]
            white_eu = [np.percentile(p - L.sigma_clipped_stats(p, sigma=3, maxiters=5)[1], 99.7) for p in pe]
        white = [v / 4 for v in white_eu]
    else:
        white = []
        for p in planes:
            _, med, sd = L.sigma_clipped_stats(p, sigma=3, maxiters=5)
            white.append(max(np.percentile(p - med, 99.7), L.FLOOR.get(inst, 0) * sd))
        if inst == "euclid":
            white_eu = white
    img = L.rgb(planes, white, L.INST[inst][3])
    save_panel(img, f"{name}.png", label=label, interp=interp)
    print(name, reg, planes[0].shape, flush=True)
import json  # noqa: E402
json.dump({k: list(v) for k, v in R.items()}, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images", "ZL_regions.json"), "w"))
