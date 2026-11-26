"""Opening / closing hero: the NEXUS f200w-0253 grand-design spiral (triples #36; was the f200w-0063 barred
spiral until 11-23, which comes out smooth in POLISHed) as full-bleed 16:9 frames (Euclid, POLISHed, JWST),
identically framed so they fade into each other. The galaxy sits in the left third at the same scale as the
triples; the right side (darkened by a gentle gradient) carries the title at the start.

Opening (meeting 11-23, "from this to THIS"): HERO_1_euclid sits under the title text; HERO_1b_euclid_from
(the same frame + "From this…") fades in over the title, then HERO_2_polished ("…to THIS.") over it. The finale
fades in HERO_3_jwst ("JWST, for comparison."). Realistic colour: the viewer's AB balance (ab_colour.py), one
stretch for all three set on the JWST galaxy square. Small label top-left (the Prezi logo sits bottom-left).
The JWST frame's colour (not its brightness) is smoothed at the NIRCam PSF scale so its pixel noise is grey."""
import os
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

from ab_colour import ab_rgb, auto_levels, noise_sigma, raw_triple

warnings.filterwarnings("ignore")
T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TILE, CX, CY, HW = "f200w-0253", 123, 273, 80
H = 2 * HW                                   # rows
W = int(round(H * 16 / 9))
X0 = CX - int(0.32 * W)                      # galaxy centred at 32% of the width
lr, sr, jw = raw_triple(TILE)
ny, nx = jw.shape[:2]
assert X0 >= 0 and X0 + W <= nx and CY - HW >= 0 and CY + HW <= ny, (X0, W, nx, ny)
sq = (slice(CY - HW, CY + HW), slice(CX - HW, CX + HW))
rect = (slice(CY - HW, CY + HW), slice(X0, X0 + W))
lev = auto_levels(jw[sq], q=20, black_sigma=2.0, sigma=noise_sigma(lr[rect]))
BOLD = os.path.join(T, "fonts", "SpaceGrotesk-Bold.ttf")
FONT, BIG, HUGE = (ImageFont.truetype(BOLD, s) for s in (42, 132, 190))
TEAL = (47, 212, 191)
OUTW, OUTH = 2400, 1350
TX, TY = int(0.60 * OUTW), OUTH // 2         # phrase anchor: left edge of the dark right side, mid-height


def frame(cube, name, label):
    im = ab_rgb(cube[rect], **lev)
    im[~np.isfinite(cube[rect]).all(-1)] = 0.0
    if name == "jwst":      # colour-only smoothing (sigma 1.5 px ~ the NIRCam PSF): pixel noise as grey grain
        L = im.mean(-1, keepdims=True)
        sm = gaussian_filter(im, (1.5, 1.5, 0))
        im = (sm + 0.02) / (sm.mean(-1, keepdims=True) + 0.02) * L
    xs = np.linspace(0, 1, im.shape[1])[None, :, None]
    im = im * (1 - 0.68 * np.clip((xs - 0.48) / 0.22, 0, 1))  # darker right side for the text
    arr = (np.clip(im[::-1], 0, 1) * 255).astype(np.uint8)     # rows are y-up
    pil = Image.fromarray(arr).resize((OUTW, OUTH), Image.NEAREST if name == "euclid" else Image.LANCZOS)
    ImageDraw.Draw(pil).text((44, 44), label, font=FONT, fill="white", anchor="lt", stroke_width=4,
                             stroke_fill="black")
    return pil


def save(pil, stem):
    out = os.path.join(T, "prezi_upload", f"{stem}.jpg")
    pil.save(out, quality=93)
    print(out)


eu = frame(lr, "euclid", "Euclid")
save(eu, "HERO_1_euclid")
d = ImageDraw.Draw(eu)
d.text((TX, TY), "From this…", font=BIG, fill="white", anchor="lm")
save(eu, "HERO_1b_euclid_from")
po = frame(sr, "polished", "POLISHed")
save(po, "HERO_2_plain")      # 11-25 (user): the title and names replace "...to THIS." on this frame, as Prezi text
d = ImageDraw.Draw(po)
d.text((TX, TY), "…to", font=BIG, fill="white", anchor="ls")
x = TX + d.textlength("…to ", font=BIG)
d.text((x, TY), "THIS.", font=HUGE, fill=TEAL, anchor="ls")
save(po, "HERO_2_polished")
js = frame(jw, "jwst", "JWST")
d = ImageDraw.Draw(js)
d.text((TX, TY), "JWST,", font=BIG, fill="white", anchor="ls")
d.text((TX, TY + 40), "for comparison.", font=ImageFont.truetype(BOLD, 84), fill=(200, 205, 212), anchor="lt")
save(js, "HERO_3_jwst")

# 11-25 review: the finale's final state is a triptych of the title galaxy (Euclid | POLISHed | JWST, one stretch),
# so the closing zoom-out shows the comparison instead of "JWST, for comparison." on its own. Replaces HERO_3.
def square(cube, name):
    im = ab_rgb(cube[sq], **lev)
    im[~np.isfinite(cube[sq]).all(-1)] = 0.0
    if name == "jwst":
        L = im.mean(-1, keepdims=True)
        sm = gaussian_filter(im, (1.5, 1.5, 0))
        im = (sm + 0.02) / (sm.mean(-1, keepdims=True) + 0.02) * L
    arr = (np.clip(im[::-1], 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr)


SIDE, GAP3 = 720, 40
x0 = (OUTW - 3 * SIDE - 2 * GAP3) // 2
y0 = 300
tri = Image.new("RGB", (OUTW, OUTH), (0, 0, 0))
d = ImageDraw.Draw(tri)
LAB = ImageFont.truetype(BOLD, 96)
for k, (cube, name, label) in enumerate(((lr, "euclid", "Euclid"), (sr, "polished", "POLISHed"),
                                         (jw, "jwst", "JWST"))):
    x = x0 + k * (SIDE + GAP3)
    tri.paste(square(cube, name).resize((SIDE, SIDE), Image.NEAREST if name == "euclid" else Image.LANCZOS),
              (x, y0))
    d.text((x + SIDE // 2, y0 - 40), label, font=LAB, fill=TEAL if name == "polished" else "white", anchor="ms")
d.text((OUTW // 2, y0 + SIDE + 115), "The same galaxy, one color stretch.", font=ImageFont.truetype(BOLD, 64),
       fill=(200, 205, 212), anchor="ms")
save(tri, "HERO_3_triptych")
