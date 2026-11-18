"""Close-ups of the cat's face (Euclid / POLISHed / truth) cut from the F14 panels, for a Prezi
dive into the Euclid cat tile: each crop sits exactly over the same region of the tile."""
import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
TALK = os.path.dirname(HERE)
X0, Y0, SIDE = 0.25, 0.14, 0.36          # fraction of the panel (from top-left)
FONT = ImageFont.truetype(os.path.join(TALK, "fonts", "SpaceGrotesk-Bold.ttf"), 64)
for src, dst, label in (("F14_cat_2_euclid.png", "G05_catface_Euclid.jpg", "Euclid"),
                        ("F14_cat_3_polished.png", "G06_catface_POLISHed.jpg", "POLISHed"),
                        ("F14_cat_1_truth.png", "G07_catface_truth.jpg", "Truth")):
    im = Image.open(os.path.join(TALK, "images", src)).convert("RGB")
    n = im.size[0]
    box = tuple(int(round(v * n)) for v in (X0, Y0, X0 + SIDE, Y0 + SIDE))
    c = im.crop(box).resize((1600, 1600), Image.NEAREST if label == "Euclid" else Image.LANCZOS)
    d = ImageDraw.Draw(c)
    d.text((40, 30), label, font=FONT, fill="white", stroke_width=5, stroke_fill="black")
    c.save(os.path.join(TALK, "prezi_upload", dst), quality=92)
    print(dst, box)
