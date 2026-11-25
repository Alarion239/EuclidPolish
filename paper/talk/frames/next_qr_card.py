"""Next + big-QR card for the top band of the canvas (user 11-25: a big QR code on the overview). A dark rounded
card, CARD_W x CARD_H canvas units (beside the hero, same height), with its soft shadow baked onto the canvas grey
(#EDEFF2) so it can go up as a JPEG. Left: future work; right: a QR code to the all-sky viewer (Euclid DR1, model 271)
filling the card's height. The image covers the card plus a SHADOW margin on every side (same proportions as
card_tile.py), so in Prezi it is scaled to (CARD_W + 2 * margin) wide and centred on the card.
Needs segno (pip install segno)."""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.environ.get("SEGNO_PATH", ""))
import segno  # noqa: E402

T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://cutout-production-538f5738.up.railway.app/#ra=228.827579&dec=45.009343&z=5.016&rel=dr1&m=271"
CARD_W, CARD_H = 50000, 16000                    # canvas units (printed for the Prezi placement)
H = 1200
W = round(H * CARD_W / CARD_H)
M, R = round(60 * H / 1125), round(22 * H / 1125)  # shadow margin and corner radius, as card_tile.py
CANVAS, BG, TXT, MUTED, TEAL = (237, 239, 242), (15, 18, 24), (230, 232, 235), (139, 147, 161), (47, 212, 191)
F = lambda name, s: ImageFont.truetype(os.path.join(T, "fonts", name), s)  # noqa: E731

full = Image.new("RGB", (W + 2 * M, H + 2 * M), CANVAS)
sh = Image.new("L", full.size, 0)
ImageDraw.Draw(sh).rounded_rectangle([M, M + 8, M + W, M + H + 8], R, fill=255)
sh = np.asarray(sh.filter(ImageFilter.GaussianBlur(18)), np.float64) / 255 * 0.10
full = Image.fromarray((np.asarray(full, np.float64) * (1 - sh[..., None])).astype(np.uint8))
d = ImageDraw.Draw(full)
d.rounded_rectangle([M, M, M + W, M + H], R, fill=BG)
x0, y0 = M + 110, M
d.text((x0, y0 + 120), "Next", font=F("SpaceGrotesk-Bold.ttf", 84), fill=TEAL, anchor="lt")
for k, line in enumerate(("POLISH all of Euclid DR1,", "then the full 14,000 deg² survey.")):
    d.text((x0, y0 + 300 + k * 150), line, font=F("SpaceGrotesk-Bold.ttf", 120), fill=TXT, anchor="lt")
d.text((x0, y0 + 760), "Try it now: the all-sky viewer,", font=F("SpaceGrotesk-Medium.ttf", 76), fill=MUTED, anchor="lt")
d.text((x0, y0 + 860), "Euclid DR1 POLISHed live in your browser  →", font=F("SpaceGrotesk-Medium.ttf", 76),
       fill=MUTED, anchor="lt")
qr = segno.make(URL, error="m")
mods = np.array([[1 if v else 0 for v in row] for row in qr.matrix], np.uint8)
n, pad = mods.shape[0], 50
side = H - 2 * 90
cell = (side - 2 * pad) // n
side = n * cell + 2 * pad
qx, qy = M + W - 90 - side, M + (H - side) // 2
d.rounded_rectangle([qx, qy, qx + side, qy + side], 30, fill="white")
full.paste(Image.fromarray((1 - mods) * 255).resize((n * cell, n * cell), Image.NEAREST), (qx + pad, qy + pad))
out = os.path.join(T, "prezi_upload", "NEXT_qr_wide.jpg")
full.save(out, quality=94)
print(out, full.size, "card px", W, H, "margin px", M, "-> image width in canvas units", round((W + 2 * M) * CARD_W / W))
