"""One image for the quiz payoff: the eight POLISHed fields (F15_quiz_*_2_polished.png, f12_quiz.py) on the card's
4 x 2 grid, with the REAL / SIMULATED frames of quiz_truth_overlay.py drawn on top. Same canvas as the overlay, so
in Prezi it sits exactly on it and fades in with one click (instead of eight). Gaps stay transparent."""
import json
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
TALK = os.path.dirname(HERE)
key = json.load(open(os.path.join(TALK, "images", "F12_quiz_answer_key.json")))
real = {k for k, v in key.items() if v.startswith("real")}
PX = 300
S, GAP = 3.05, 0.2
W, H = round((4 * S + 3 * GAP) * PX), round((2 * S + GAP) * PX)
im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
d = ImageDraw.Draw(im)
font = ImageFont.truetype(os.path.join(TALK, "fonts", "SpaceGrotesk-Bold.ttf"), 69)
GREEN, RED = (52, 199, 89, 255), (255, 69, 58, 255)
for i, letter in enumerate("ABCDEFGH"):
    x0, y0 = round((i % 4) * (S + GAP) * PX), round((i // 4) * (S + GAP) * PX)
    side = round(S * PX)
    tile = Image.open(os.path.join(TALK, "images", f"F15_quiz_{letter}_2_polished.png")).convert("RGBA").resize((side, side), Image.LANCZOS)
    im.paste(tile, (x0, y0))
    x1, y1 = x0 + side - 1, y0 + side - 1
    col = GREEN if letter in real else RED
    d.rectangle([x0, y0, x1, y1], outline=col, width=24)
    label = "REAL" if letter in real else "SIMULATED"
    tw = d.textlength(label, font=font)
    bx0, by1 = (x0 + x1) / 2 - tw / 2 - 27, y1 - 39
    d.rounded_rectangle([bx0, by1 - 96, bx0 + tw + 54, by1], radius=18, fill=col)
    d.text((bx0 + 27, by1 - 90), label, font=font, fill=(255, 255, 255, 255))
out = os.path.join(TALK, "prezi_upload", "QUIZ_polished_composite.png")
im.save(out, optimize=True)
print(out, im.size, sorted(real))
