"""Section panels for the 11-25 storyboard grid (user: follow the paper introduction, sections 1-10 in reading
order). Each section of the talk sits on one light rounded panel (common region, the strongest grouping cue) holding
its number + name and its 1 or 2 slide tiles. Plain JPEGs with the canvas grey (#EDEFF2) baked outside the rounded
corners. Geometry in canvas units (tile 20000 x 11250, column pitch 23400) is shared with the Prezi layout script."""
import os

from PIL import Image, ImageDraw

T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TILE_W, TILE_H, PITCH = 20000, 11250, 23400
PAD_X, PAD_TOP, TAG, PAD_BOT = 900, 500, 1900, 900
H = PAD_TOP + TAG + TILE_H + PAD_BOT
K = 0.06                                         # px per canvas unit
CANVAS, PANEL = (237, 239, 242), (222, 227, 234)
for n in (1, 2):
    w = TILE_W + (n - 1) * PITCH + 2 * PAD_X
    im = Image.new("RGB", (round(w * K), round(H * K)), CANVAS)
    ImageDraw.Draw(im).rounded_rectangle([0, 0, im.width - 1, im.height - 1], round(700 * K), fill=PANEL)
    out = os.path.join(T, "prezi_upload", f"REGION_{n}.jpg")
    im.save(out, quality=95)
    print(out, im.size, "canvas units", w, H)
