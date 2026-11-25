"""Next card (meeting 11-23 + 11-25): future work and a QR code to the all-sky viewer, a dark 16:9 frame the size
of the hero, beside it on the canvas. Since 11-25 the three conclusions are the canvas's row headers (the final
slide is the zoom-out of everything), so this card carries only "Next" and the QR (in-browser POLISH viewer,
Euclid DR1, model 271). Needs segno (pip install segno)."""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.environ.get("SEGNO_PATH", ""))
import segno  # noqa: E402

T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://cutout-production-538f5738.up.railway.app/#ra=228.827579&dec=45.009343&z=5.016&rel=dr1&m=271"
W, H = 2400, 1350
BG, TXT, MUTED, TEAL = (15, 18, 24), (230, 232, 235), (139, 147, 161), (47, 212, 191)
F = lambda name, s: ImageFont.truetype(os.path.join(T, "fonts", name), s)  # noqa: E731
im = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(im)
d.text((110, 120), "Next", font=F("SpaceGrotesk-Bold.ttf", 92), fill=TXT, anchor="lt")
NEXT = [("POLISH all of Euclid DR1,", TXT), ("then the full", TXT), ("14,000 deg² survey.", TEAL)]
for k, (line, col) in enumerate(NEXT):
    d.text((110, 420 + k * 120), line, font=F("SpaceGrotesk-Bold.ttf", 88), fill=col, anchor="lt")
d.text((110, 860), "The viewer already runs on DR1:", font=F("SpaceGrotesk-Regular.ttf", 52), fill=MUTED, anchor="lt")
d.text((110, 930), "scan the code to try it.", font=F("SpaceGrotesk-Regular.ttf", 52), fill=MUTED, anchor="lt")
# QR on a white rounded tile
qr = segno.make(URL, error="m")
mods = np.array([[1 if v else 0 for v in row] for row in qr.matrix], np.uint8)
n = mods.shape[0]
Q, PAD = 560, 44
cell = Q // n
q = Image.fromarray((1 - mods) * 255).resize((n * cell, n * cell), Image.NEAREST)
x0, y0 = 1700, 250
side = n * cell + 2 * PAD
d.rounded_rectangle([x0, y0, x0 + side, y0 + side], 28, fill="white")
im.paste(q, (x0 + PAD, y0 + PAD))
cx = x0 + side / 2
d.text((cx, y0 + side + 50), "Try it: the all-sky viewer", font=F("SpaceGrotesk-Bold.ttf", 46), fill=TXT, anchor="mt")
d.text((cx, y0 + side + 118), "Euclid DR1, POLISHed live", font=F("SpaceGrotesk-Regular.ttf", 36), fill=MUTED,
       anchor="mt")
d.text((cx, y0 + side + 166), "in your browser", font=F("SpaceGrotesk-Regular.ttf", 36), fill=MUTED, anchor="mt")
d.line([(1560, 250), (1560, 1100)], fill=(44, 51, 64), width=3)
out = os.path.join(T, "prezi_upload", "NEXT_card.jpg")
im.save(out, quality=93)
print(out, n, cell, side)
