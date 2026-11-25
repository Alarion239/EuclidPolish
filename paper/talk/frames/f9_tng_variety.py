"""TNG card (11-25 review): 18 TNG50-SKIRT galaxies, 6 x 3, picked by eye from the 48-galaxy atlas wall
(data/tng_wall, AB colour) for variety -- grand-design, barred, ringed and flocculent spirals, edge-ons with dust
lanes, tidal features, a clumpy irregular, smooth early types -- instead of the 12 mostly smooth discs, so
"how does it make spiral arms?" is answered on the slide. One image (white gaps = the card) replaces B01..B12."""
import os

from PIL import Image

from style import TALK

src = os.path.join(TALK, "data", "tng_wall")
PICK = [425719, 427211, 610017, 96775, 592984, 537941,
        352426, 184938, 537236, 522530, 541218, 473330,
        619223, 525002, 596660, 515695, 63887, 583745]
COLS, ROWS, CELL = 6, 3, 600
GAP = round(CELL * 112 / 2533)                    # the old grid's gap/tile ratio
W, H = COLS * CELL + (COLS - 1) * GAP, ROWS * CELL + (ROWS - 1) * GAP
im = Image.new("RGB", (W, H), "white")
for i, sid in enumerate(PICK):
    t = Image.open(os.path.join(src, f"tng_{sid:06d}.png")).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
    im.paste(t, ((i % COLS) * (CELL + GAP), (i // COLS) * (CELL + GAP)))
out = os.path.join(TALK, "prezi_upload", "v18", "TNG_grid18.jpg")
os.makedirs(os.path.dirname(out), exist_ok=True)
im.save(out, quality=92)
print(out, im.size)
