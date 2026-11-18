"""C1: structure Euclid's near-infrared cannot see, recovered and confirmed by JWST.

NEXUS tile f200w-0063, a barred spiral: Euclid H (0.3" native pixels) shows a blob; POLISHed H shows
the bar, an arm and a knot; JWST F200W shows the same. Separate, identically framed squares, one shared
asinh stretch (white point from JWST), so the POLISHed and JWST panels drop onto the Euclid one.
Also the spiral in f200w-0253 as a second example.
"""
import sys

sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish/output/jwst_triples_257/code")
sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish")
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from render import TILES, coord_text

from preview_band import band_imgs
from style import save_panel

OBJECTS = {
    "barred": dict(tile="f200w-0063", cx=103, cy=230, hw=84),
    "spiral": dict(tile="f200w-0253", cx=123, cy=273, hw=102),
}
BANDS = {3: ("H", "F200W"), 2: ("J", "F150W")}

for name, o in OBJECTS.items():
    w = WCS(fits.getheader(f"{TILES}/starfull_combiner_{o['tile'].split('-')[1]}.fits")).celestial
    ra, dec = w.pixel_to_world_values(o["cx"], o["cy"])
    coords = coord_text(float(ra), float(dec))
    for k, (eb, jb) in BANDS.items():
        lr, sr, jw = band_imgs(o["tile"], o["cx"], o["cy"], o["hw"], k)
        rgb = [np.repeat(x[..., None], 3, -1) for x in (lr, sr, jw)]
        stem = f"C1_{name}_{eb}"
        save_panel(rgb[0], f"{stem}_1_euclid.png", label=f"Euclid · {eb}", interp="nearest",
                   envelope=True, coords=coords)
        save_panel(rgb[1], f"{stem}_2_polished.png", label=f"POLISHed · {eb}", bar_arcsec=1.0)
        save_panel(rgb[2], f"{stem}_3_jwst.png", label=f"JWST · {jb}", compass=True)
        print(stem, coords)
