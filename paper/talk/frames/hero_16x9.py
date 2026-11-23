"""Opening / closing hero: the NEXUS f200w-0063 barred spiral as full-bleed 16:9 frames (Euclid, POLISHed, JWST),
identically framed so they can fade into each other. The galaxy sits in the left third at the same scale as the
C1_barred squares; the right side (darkened by a gentle gradient) carries the title at the start and the takeaways at the end.
One stretch for all three (the JWST galaxy square's, as in c1_nexus_tile_colour.py). Small label bottom-left."""
import os
import sys

sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish/output/jwst_triples_257/code")
sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish")
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from render import cubes, lupton, rgb_channels, saturate

T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TILE, CX, CY, HW, Q = "f200w-0063", 103, 230, 70, 15
H = 2 * HW                                   # rows: same vertical scale as the C1_barred squares
W = int(round(H * 16 / 9))
X0 = CX - int(0.32 * W)                      # galaxy centred at 32% of the width
lr, sr, jw = cubes(TILE)
ny, nx = jw.shape[:2]
assert X0 >= 0 and X0 + W <= nx and CY - HW >= 0 and CY + HW <= ny, (X0, W, nx, ny)
sq = (slice(CY - HW, CY + HW), slice(CX - HW, CX + HW))
rect = (slice(CY - HW, CY + HW), slice(X0, X0 + W))
ref = rgb_channels(jw[sq], "full")
white = np.nanpercentile((ref[0] + ref[1] + ref[2]) / 3, 99.8)
soft = white / Q
FONT = ImageFont.truetype(os.path.join(T, "fonts", "SpaceGrotesk-Bold.ttf"), 42)
OUTW, OUTH = 2400, 1350
for k, (cube, name, label) in enumerate(((lr, "euclid", "Euclid"), (sr, "polished", "POLISHed"),
                                         (jw, "jwst", "JWST / NIRCam"))):
    r, g, b = rgb_channels(cube[rect], "full")
    holes = ~(np.isfinite(r) & np.isfinite(g) & np.isfinite(b))
    im = saturate(lupton(np.nan_to_num(r), np.nan_to_num(g), np.nan_to_num(b), soft=soft, Q=1.0, white=white), 1.25)
    im[holes] = 0.0
    xs = np.linspace(0, 1, im.shape[1])[None, :, None]
    im = im * (1 - 0.68 * np.clip((xs - 0.48) / 0.22, 0, 1))  # darker right side for the text
    arr = (np.clip(im[::-1], 0, 1) * 255).astype(np.uint8)     # rows are y-up
    pil = Image.fromarray(arr).resize((OUTW, OUTH), Image.NEAREST if k == 0 else Image.LANCZOS)
    d = ImageDraw.Draw(pil)
    d.text((44, OUTH - 44), label, font=FONT, fill="white", anchor="ls", stroke_width=4, stroke_fill="black")
    out = os.path.join(T, "prezi_upload", f"HERO_{k + 1}_{name}.jpg")
    pil.save(out, quality=93)
    print(out, arr.shape)
