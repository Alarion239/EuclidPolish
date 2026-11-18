"""C1: the cold-open tile (NEXUS f200w-0106) and its payoff.

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

TILE = "f200w-0106"
VIEWS = {
    "tile": dict(cx=255, cy=255, hw=255, q=60, bar=5.0),
    "pair": dict(cx=175, cy=400, hw=100, q=30, bar=1.0),
}
w = WCS(fits.getheader(f"{TILES}/starfull_combiner_{TILE.split('-')[1]}.fits")).celestial
for view, v in VIEWS.items():
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
