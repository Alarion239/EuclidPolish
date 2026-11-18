"""F9: the TNG50-SKIRT atlas wall. 48 donors evenly in stellar mass (10^9.8 to 10^12 Msun, top-left
to bottom-right), orientation 1, R=H, G=(Y+J)/2, B=VIS, white-balanced on the wall's median galaxy.
Thumbnails rendered on FASRC by remote/tng_wall_thumbs.py into ../data/tng_wall/."""
import glob
import json
import os

from PIL import Image

from style import OUT, TALK

src = os.path.join(TALK, "data", "tng_wall")
files = sorted(glob.glob(os.path.join(src, "tng_*.png")))
meta = {m["id"]: m for m in json.load(open(os.path.join(src, "meta.json")))}
order = sorted(files, key=lambda f: meta[str(int(os.path.basename(f)[4:10]))]["logm"])
cols, rows, cell, gap = 8, 6, 360, 6
W, H = cols * cell + (cols - 1) * gap, rows * cell + (rows - 1) * gap
wall = Image.new("RGB", (W, H), "#0f1218")
for i, f in enumerate(order):
    wall.paste(Image.open(f).convert("RGB").resize((cell, cell), Image.LANCZOS),
               ((i % cols) * (cell + gap), (i // cols) * (cell + gap)))
os.makedirs(OUT, exist_ok=True)
wall.save(os.path.join(OUT, "F9_tng_atlas_wall.png"))
print(W, H, len(order), "galaxies; logM", meta[str(int(os.path.basename(order[0])[4:10]))]["logm"],
      "->", meta[str(int(os.path.basename(order[-1])[4:10]))]["logm"])
