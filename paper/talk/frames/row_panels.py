"""Row panels for the 11-25 four-point layout (user: at most four overview points that summarize, not a recitation
of the slides). Each row of slide tiles sits on one light rounded panel under its summary sentence: rows of 5, 5, 4
and 2 tiles; the 2-tile row's sentence wraps to two lines, so its header band is twice as tall. JPEGs with the canvas
grey baked outside the rounded corners. Canvas units shared with the Prezi layout script."""
import os

from PIL import Image, ImageDraw

T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TILE_W, TILE_H, GAP = 20000, 11250, 1600
PAD_X, PAD_TOP, PAD_BOT, HDR = 900, 500, 700, 2300
K = 0.06
CANVAS, PANEL = (237, 239, 242), (222, 227, 234)
for n, hdr in ((5, HDR), (4, HDR), (2, 2 * HDR)):
    w = n * TILE_W + (n - 1) * GAP + 2 * PAD_X
    h = PAD_TOP + hdr + TILE_H + PAD_BOT
    im = Image.new("RGB", (round(w * K), round(h * K)), CANVAS)
    ImageDraw.Draw(im).rounded_rectangle([0, 0, im.width - 1, im.height - 1], round(700 * K), fill=PANEL)
    out = os.path.join(T, "prezi_upload", f"ROW_{n}.jpg")
    im.save(out, quality=95)
    print(out, im.size, "canvas units", w, h)
