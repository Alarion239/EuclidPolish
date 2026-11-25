"""JWST card summary (11-25 review): the four galaxies as a 3 x 4 grid -- rows Euclid, JWST, POLISHed (the
user's order), one column per galaxy -- faded in on the card after the per-galaxy zooms, so the closing view
shows all three versions side by side and the card's empty lower third is used. Cells: jwst_all.py's unlabelled
panels. Size = the card's picture area under the result line (18,011 x 6,600 Prezi units), black like the
backing panel."""
import os

from PIL import Image, ImageDraw, ImageFont

from style import TALK

GAL = ["01_tidal", "02_barred2", "07_interacting", "04_edge_on"]
ROWS = [("Euclid", "1_euclid", "white"), ("JWST", "3_jwst", "white"), ("POLISHed", "2_polished", (47, 212, 191))]
W, H = 5458, 2000                                  # 18,011 x 6,600 units
GAP = 44
TOP = GAP + 12
CELL = (H - 2 * GAP - 2 * TOP) // 3
LABW = 1150
x0 = (W - LABW - 4 * CELL - 3 * GAP) // 2 + LABW
im = Image.new("RGB", (W, H), "black")
d0 = ImageDraw.Draw(im)
gx1 = x0 + 4 * CELL + 3 * GAP
d0.rectangle([x0 - GAP, TOP - GAP, gx1 + GAP - 1, TOP + 3 * CELL + 3 * GAP - 1], fill=(58, 64, 74))  # grid lines
d = ImageDraw.Draw(im)
font = ImageFont.truetype(os.path.join(TALK, "fonts", "SpaceGrotesk-Bold.ttf"), 120)
for r, (lab, part, col) in enumerate(ROWS):
    y = TOP + r * (CELL + GAP)
    d.text((x0 - 70, y + CELL // 2), lab, font=font, fill=col, anchor="rm")
    for c, g in enumerate(GAL):
        cell = Image.open(os.path.join(TALK, "images", f"JWST{g}_{part}_nolabel.png")).convert("RGB")
        im.paste(cell.resize((CELL, CELL), Image.LANCZOS), (x0 + c * (CELL + GAP), y))
out = os.path.join(TALK, "prezi_upload", "v18", "JWST_grid.jpg")
im.save(out, quality=92)
print(out, im.size, CELL)
