"""All ten NEXUS comparisons sent to Liam (output/jwst_triples_257/triples.json), as separate talk panels
(Euclid / POLISHed / JWST), same crops, calibration and stretch as the triples."""
import json
import sys

sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish/output/jwst_triples_257/code")
sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish")
import warnings

warnings.filterwarnings("ignore")
from render import LEGEND, triple_images

from style import save_panel

fin = {f["caption_id"]: f for f in json.load(open(
    "/Users/alarion239/Desktop/EuclidPolish/output/jwst_triples_257/code/finalists.json"))}
pick = [("31tidal", "tidal", dict(cx=395, cy=410, hw=100)), ("#3", "barred2", {}), ("#36", "grand_design", {}),
        ("#10", "edge_on", {}), ("#43", "clumpy", {}), ("#51", "warped", {}), ("#15", "interacting", {}),
        ("#48", "multi_arm", {}), ("#60", "inner_ring", {}), ("#52", "ringed_lenticular", {})]
meta = json.load(open("/Users/alarion239/Desktop/EuclidPolish/output/jwst_triples_257/triples.json"))
for n, ((cid, name, over), m) in enumerate(zip(pick, meta), 1):
    rec = dict(fin[cid], **over)
    (lr, sr, jw), mode = triple_images(rec)
    le, lj = LEGEND[mode]
    side = 2 * rec["hw"] * 0.05
    bar = 1.0 if side < 12 else 2.0
    stem = f"JWST{n:02d}_{name}"
    save_panel(lr, f"{stem}_1_euclid.png", label="Euclid", legend=le, interp="nearest")
    save_panel(sr, f"{stem}_2_polished.png", label="POLISHed", legend=le, bar_arcsec=bar)
    save_panel(jw, f"{stem}_3_jwst.png", label="JWST / NIRCam", legend=lj, compass=True)
    print(stem, m["what"])
