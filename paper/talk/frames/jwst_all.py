"""All ten NEXUS comparisons sent to Liam (output/jwst_triples_257/triples.json), as separate talk panels
(Euclid / POLISHed / JWST), same crops as the triples. Realistic colour since 11-23: the viewer's AB balance
(ab_colour.py), one stretch per galaxy set on the JWST panel, black point at the Euclid sky noise. Clean labels:
only the instrument name, top-left (no band legend, compass or bar)."""
import json
import os
import warnings

import numpy as np
from PIL import Image

from ab_colour import ab_rgb, auto_levels, noise_sigma, raw_triple
from style import TALK, save_panel

warnings.filterwarnings("ignore")
fin = {f["caption_id"]: f for f in json.load(open(
    "/Users/alarion239/Desktop/EuclidPolish/output/jwst_triples_257/code/finalists.json"))}
pick = [("31tidal", "tidal", dict(cx=395, cy=410, hw=100)), ("#3", "barred2", {}), ("#36", "grand_design", {}),
        ("#10", "edge_on", {}), ("#43", "clumpy", {}), ("#51", "warped", {}), ("#15", "interacting", {}),
        ("#48", "multi_arm", {}), ("#60", "inner_ring", {}), ("#52", "ringed_lenticular", {})]
meta = json.load(open("/Users/alarion239/Desktop/EuclidPolish/output/jwst_triples_257/triples.json"))
UP = os.path.join(TALK, "prezi_upload")
for n, ((cid, name, over), m) in enumerate(zip(pick, meta), 1):
    rec = dict(fin[cid], **over)
    mode, hw = rec.get("mode", "full"), rec["hw"]
    lr, sr, jw = raw_triple(rec["tile"])
    sl = (slice(rec["cy"] - hw, rec["cy"] + hw), slice(rec["cx"] - hw, rec["cx"] + hw))
    ref = jw[sl] if rec.get("ref", "jwst") == "jwst" else sr[sl]
    lev = auto_levels(ref, pct=rec.get("white_pct", 99.8), mode=mode, sigma=noise_sigma(lr[sl], mode))
    stem = f"JWST{n:02d}_{name}"
    for k, (cube, part, label) in enumerate(((lr, "1_euclid", "Euclid"), (sr, "2_polished", "POLISHed"),
                                             (jw, "3_jwst", "JWST"))):
        path = save_panel(ab_rgb(cube[sl], mode=mode, **lev), f"{stem}_{part}.png", label=label,
                          interp="nearest" if k == 0 else "lanczos")
        Image.open(path).convert("RGB").resize((1600, 1600), Image.LANCZOS).save(
            os.path.join(UP, f"L{n:02d}{k + 1}_{name}_{label}.jpg"), quality=90)
    print(stem, mode, {k: round(float(v), 2) for k, v in lev.items()}, m["what"])
