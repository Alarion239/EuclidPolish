"""Close-up crops for Prezi dives (like the cat face): the same fractional box cut from each panel of a
callback set (Euclid / POLISHed / JWST), so each crop can sit exactly over that region of the Euclid tile.
Box = (x0, y0, side) as fractions of the panel, measured from the top-left."""
import json
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
TALK = os.path.dirname(HERE)
IMG = os.path.join(TALK, "images")
OUT = os.path.join(TALK, "prezi_upload")
FONT = ImageFont.truetype(os.path.join(TALK, "fonts", "SpaceGrotesk-Bold.ttf"), 64)
SETS = {  # name: (stems, labels, box)
    "barred": (["C1_barred_1_euclid", "C1_barred_2_polished", "C1_barred_3_jwst"], ["Euclid", "POLISHed", "JWST"], (0.24, 0.24, 0.5)),
    "q1ring": (["LENS_ring102042919_1_euclid", "LENS_ring102042919_2_polished"], ["Euclid", "POLISHed"], (0.27, 0.27, 0.46)),
    "ngc6505": (["LENS_NGC6505_1_euclid", "LENS_NGC6505_2_polished"], ["Euclid", "POLISHed"], (0.2, 0.2, 0.6)),
    "dr1g4": (["DR1L_G04_1_euclid", "DR1L_G04_2_polished"], ["Euclid", "POLISHed"], (0.25, 0.25, 0.5)),
    "dr1g2": (["DR1L_G02_1_euclid", "DR1L_G02_2_polished"], ["Euclid", "POLISHed"], (0.27, 0.27, 0.46)),
    "grand": (["JWST03_grand_design_1_euclid", "JWST03_grand_design_2_polished", "JWST03_grand_design_3_jwst"],
              ["Euclid", "POLISHed", "JWST"], (0.22, 0.22, 0.56)),
}
meta = {}
for name, (stems, labels, (x0, y0, s)) in SETS.items():
    for i, (stem, lab) in enumerate(zip(stems, labels), 1):
        im = Image.open(os.path.join(IMG, stem + ".png")).convert("RGB")
        n = im.size[0]
        box = tuple(int(round(v * n)) for v in (x0, y0, x0 + s, y0 + s))
        c = im.crop(box).resize((1600, 1600), Image.NEAREST if lab == "Euclid" else Image.LANCZOS)
        ImageDraw.Draw(c).text((40, 30), lab, font=FONT, fill="white", stroke_width=5, stroke_fill="black")
        c.save(os.path.join(OUT, f"Z_{name}_{i}_{lab.lower()}.jpg"), quality=92)
    meta[name] = {"stems": stems, "box": [x0, y0, s]}
json.dump(meta, open(os.path.join(OUT, "zoom_crops.json"), "w"), indent=1)
print(json.dumps(meta))
