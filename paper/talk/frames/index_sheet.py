"""Contact sheet of every talk image (images/00_index.png), file names under each thumbnail."""
import glob
import os

from PIL import Image, ImageDraw, ImageFont

from style import FONTS, OUT

files = sorted(f for f in glob.glob(os.path.join(OUT, "*.png")) if not os.path.basename(f).startswith("00_"))
cols, cell, cap = 10, 240, 34
rows = (len(files) + cols - 1) // cols
sheet = Image.new("RGB", (cols * cell, rows * (cell + cap)), "#0f1218")
draw = ImageDraw.Draw(sheet)
font = ImageFont.truetype(os.path.join(FONTS, "SpaceMono-Regular.ttf"), 11)
for i, f in enumerate(files):
    im = Image.open(f).convert("RGB")
    im.thumbnail((cell - 8, cell - 8))
    x, y = (i % cols) * cell, (i // cols) * (cell + cap)
    sheet.paste(im, (x + (cell - im.width) // 2, y + 4))
    name = os.path.basename(f)[:-4]
    draw.text((x + 4, y + cell + 2), name[:34], fill="#8b93a1", font=font)
    if len(name) > 34:
        draw.text((x + 4, y + cell + 16), name[34:68], fill="#8b93a1", font=font)
sheet.save(os.path.join(OUT, "00_index.png"))
print(len(files), "images")
