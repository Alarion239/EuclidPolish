"""Radio card image: the full-field simulated DSA-2000 dirty image from Wu, Connor, McCarty & Bouman 2026
(arXiv:2603.09162, Fig. 2 left panel, figures/patch_lr.jpg of the arXiv source), cropped inside the panel so
the paper's own label and scale bar are cut away. Every bright source carries the same starburst of
sidelobes: the interferometer's PSF, known exactly from the antenna layout."""
import os

from PIL import Image, ImageDraw, ImageFont

T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
im = Image.open(os.path.join(T, "data", "radio", "wu2026_patch_lr.jpg")).convert("RGB")
crop = im.crop((0, 1360, 1314, 2674))   # lower-left square of the left panel: below the zoom box, left of the scale bar
crop = crop.resize((1600, 1600), Image.LANCZOS)
FONT = ImageFont.truetype(os.path.join(T, "fonts", "SpaceGrotesk-Bold.ttf"), 64)
d = ImageDraw.Draw(crop)
d.text((44, 36), "Radio, simulated DSA-2000", font=FONT, fill="white", stroke_width=5, stroke_fill="black")
crop.save(os.path.join(T, "prezi_upload", "RADIO_dsa2000_dirty.jpg"), quality=92)
print("ok")
