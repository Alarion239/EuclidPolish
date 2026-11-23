"""POLISHed | JWST vertical-split squares for the Prezi JWST card: the left half of the POLISHed panel and
the right half of the matching JWST panel (same field, same framing, from jwst_all.py), a thin seam, labels
in the two top corners. Structures that continue across the seam = the super-resolution agrees with JWST."""
import glob
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
I, U = os.path.join(T, "images"), os.path.join(T, "prezi_upload")
FONT = ImageFont.truetype(os.path.join(T, "fonts", "SpaceGrotesk-Bold.ttf"), 64)
for p in sorted(glob.glob(os.path.join(I, "JWST??_*_2_polished.png"))):
    stem = os.path.basename(p)[:-len("_2_polished.png")]
    P = Image.open(p).convert("RGB")
    J = Image.open(os.path.join(I, stem + "_3_jwst.png")).convert("RGB").resize(P.size)
    n = P.size[0]
    # seam through the galaxy: column of the brightest smoothed blob in the central 60% of the panel
    lum = gaussian_filter(np.asarray(P.convert("L"), float), n / 60)
    c0, c1 = int(0.2 * n), int(0.8 * n)
    yx = np.unravel_index(np.argmax(lum[c0:c1, c0:c1]), (c1 - c0, c1 - c0))
    xs = int(np.clip(yx[1] + c0, 0.3 * n, 0.7 * n))
    out = P.copy()
    out.paste(J.crop((xs, 0, n, n)), (xs, 0))
    d = ImageDraw.Draw(out)
    d.line([(xs, 0), (xs, n)], fill="white", width=6)
    tw = d.textlength("JWST", font=FONT)
    d.text((n - tw - 60, 50), "JWST", font=FONT, fill="white", stroke_width=5, stroke_fill="black")
    out.resize((1600, 1600), Image.LANCZOS).save(os.path.join(U, f"S_{stem}.jpg"), quality=92)
    print(stem)
