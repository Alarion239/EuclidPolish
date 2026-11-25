"""11-25 (user): the title slide and a BIG QR code go in the free space on the right of the left-aligned rows.
Two dark cards in the house style (shadow baked onto the canvas grey, like next_qr_card.py):
  NEXT_text.jpg  - 16:9 tile for row 4: future work, no QR (the QR has its own card);
  QR_portrait.jpg - a portrait card (20000 x 29300 canvas units) under the title slide: "Try it now" + a QR code to
                    the all-sky viewer (Euclid DR1, model 271) filling the card's width.
Needs segno (pip install segno)."""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.environ.get("SEGNO_PATH", ""))
import segno  # noqa: E402

T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://cutout-production-538f5738.up.railway.app/#ra=228.827579&dec=45.009343&z=5.016&rel=dr1&m=271"
CANVAS, BG, TXT, MUTED, TEAL = (237, 239, 242), (15, 18, 24), (230, 232, 235), (139, 147, 161), (47, 212, 191)
F = lambda name, s: ImageFont.truetype(os.path.join(T, "fonts", name), s)  # noqa: E731


def card(W, H):
    """Dark rounded card W x H px with a soft shadow on the canvas grey; returns (image, draw, margin)."""
    M, R = round(60 * min(W, H) / 1125), round(22 * min(W, H) / 1125)
    full = Image.new("RGB", (W + 2 * M, H + 2 * M), CANVAS)
    sh = Image.new("L", full.size, 0)
    ImageDraw.Draw(sh).rounded_rectangle([M, M + 8, M + W, M + H + 8], R, fill=255)
    sh = np.asarray(sh.filter(ImageFilter.GaussianBlur(18)), np.float64) / 255 * 0.10
    full = Image.fromarray((np.asarray(full, np.float64) * (1 - sh[..., None])).astype(np.uint8))
    d = ImageDraw.Draw(full)
    d.rounded_rectangle([M, M, M + W, M + H], R, fill=BG)
    return full, d, M


def save(im, name, card_w_px, card_w_units):
    out = os.path.join(T, "prezi_upload", name)
    im.save(out, quality=94)
    print(out, im.size, "-> image width in canvas units", round(im.width * card_w_units / card_w_px))


# Next (text only), 16:9 tile: 20000 x 11250 units
W, H = 2133, 1200
im, d, M = card(W, H)
x0 = M + 110
d.text((x0, M + 120), "Next", font=F("SpaceGrotesk-Bold.ttf", 80), fill=TEAL, anchor="lt")
for k, line in enumerate(("POLISH all of", "Euclid DR1, then", "the full survey.")):
    d.text((x0, M + 290 + k * 130), line, font=F("SpaceGrotesk-Bold.ttf", 112), fill=TXT, anchor="lt")
d.text((x0, M + 820), "14,000 deg², the same network.", font=F("SpaceGrotesk-Medium.ttf", 64), fill=MUTED, anchor="lt")
d.text((x0, M + 910), "Try the viewer: scan the code  →", font=F("SpaceGrotesk-Medium.ttf", 64), fill=MUTED, anchor="lt")
save(im, "NEXT_text.jpg", W, 20000)

# QR portrait card: 20000 x 29300 units
W, H = 1600, round(1600 * 29300 / 20000)
im, d, M = card(W, H)
cx = M + W // 2
d.text((cx, M + 110), "Try it now", font=F("SpaceGrotesk-Bold.ttf", 76), fill=TEAL, anchor="mt")
d.text((cx, M + 215), "the all-sky viewer", font=F("SpaceGrotesk-Bold.ttf", 104), fill=TXT, anchor="mt")
qr = segno.make(URL, error="m")
mods = np.array([[1 if v else 0 for v in row] for row in qr.matrix], np.uint8)
n, pad = mods.shape[0], 60
side0 = W - 2 * 100
cell = (side0 - 2 * pad) // n
side = n * cell + 2 * pad
qx, qy = M + (W - side) // 2, M + 420
d.rounded_rectangle([qx, qy, qx + side, qy + side], 36, fill="white")
im.paste(Image.fromarray((1 - mods) * 255).resize((n * cell, n * cell), Image.NEAREST), (qx + pad, qy + pad))
for k, line in enumerate(("Euclid DR1, POLISHed live", "in your browser")):
    d.text((cx, qy + side + 90 + k * 90), line, font=F("SpaceGrotesk-Medium.ttf", 68), fill=MUTED, anchor="mt")
save(im, "QR_portrait.jpg", W, 20000)
