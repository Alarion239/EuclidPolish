"""C1: the barred spiral in NEXUS f200w-0063 (and the 0106 tile), colour composites.

Writes separate, identically framed squares: Euclid (shown at the start, with the envelope mark),
POLISHed and JWST (dropped on the same spot at the end). Whole 25.5" tile + a zoom on the pair.
"""
import sys

sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish/output/jwst_triples_257/code")
sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish")
import warnings

warnings.filterwarnings("ignore")
from render import LEGEND, coord_text, triple_images, TILES
from astropy.io import fits
from astropy.wcs import WCS

from style import save_panel

TARGETS = {
    "barred": ("f200w-0063", dict(cx=103, cy=230, hw=70, q=15, bar=1.0)),
    "tile0106": ("f200w-0106", dict(cx=255, cy=255, hw=255, q=60, bar=5.0)),
    "pair0106": ("f200w-0106", dict(cx=175, cy=400, hw=100, q=30, bar=1.0)),
}
for view, (TILE, v) in TARGETS.items():
    w = WCS(fits.getheader(f"{TILES}/starfull_combiner_{TILE.split('-')[1]}.fits")).celestial
    rec = dict(tile=TILE, mode="full", **v)
    (lr, sr, jw), mode = triple_images(rec)
    ra, dec = w.pixel_to_world_values(v["cx"], v["cy"])
    coords = coord_text(float(ra), float(dec))
    leg_e, leg_j = LEGEND[mode]
    stem = f"C1_{view}"
    save_panel(lr, f"{stem}_1_euclid.png", label="Euclid", legend=leg_e, interp="nearest",
               envelope=True, coords=coords)
    save_panel(sr, f"{stem}_2_polished.png", label="POLISHed", legend=leg_e,
               bar_arcsec=v["bar"])
    save_panel(jw, f"{stem}_3_jwst.png", label="JWST / NIRCam", legend=leg_j, compass=True)
    print(stem, coords)
