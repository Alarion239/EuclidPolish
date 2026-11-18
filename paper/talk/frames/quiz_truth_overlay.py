"""Transparent overlay for the Prezi quiz card: green frames + REAL on the real Q1 fields (A, D, F, G), red frames +
SIMULATED on the synthetic ones (B, C, E, H). Matches the card's 4 x 2 tile grid (tile 3.05 u, gap 0.2 u)."""
import json
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
TALK = os.path.dirname(HERE)
key = json.load(open(os.path.join(TALK, "images", "F12_quiz_answer_key.json")))
print(key)
PX = 200                                   # pixels per card unit
S, GAP = 3.05, 0.2
W, H = round((4 * S + 3 * GAP) * PX), round((2 * S + GAP) * PX)
im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
d = ImageDraw.Draw(im)
font = ImageFont.truetype(os.path.join(TALK, "fonts", "SpaceGrotesk-Bold.ttf"), 46)
GREEN, RED = (52, 199, 89, 255), (255, 69, 58, 255)
real = set("ADFG")
for i, letter in enumerate("ABCDEFGH"):
    x0, y0 = round((i % 4) * (S + GAP) * PX), round((i // 4) * (S + GAP) * PX)
    x1, y1 = x0 + round(S * PX) - 1, y0 + round(S * PX) - 1
    col = GREEN if letter in real else RED
    d.rectangle([x0, y0, x1, y1], outline=col, width=16)
    label = "REAL" if letter in real else "SIMULATED"
    tw = d.textlength(label, font=font)
    bx0, by1 = (x0 + x1) / 2 - tw / 2 - 18, y1 - 26
    d.rounded_rectangle([bx0, by1 - 64, bx0 + tw + 36, by1], radius=12, fill=col)
    d.text((bx0 + 18, by1 - 60), label, font=font, fill=(255, 255, 255, 255))
out = os.path.join(TALK, "prezi_upload", "E09_quiz_truth_overlay.png")
im.save(out)
print(out, im.size)
