"""White rounded card with a soft shadow, as a transparent PNG. In Prezi one copy sits behind every 16:9 card
frame (inner white area = the frame exactly) on a light-grey canvas, so the cards read as tiles in the overview
(frame borders are not drawn in Present mode) while a zoomed-in card still fills the screen with white."""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H, M, R = 2000, 1125, 60, 22             # card px, shadow margin, corner radius
shadow = Image.new("L", (W + 2 * M, H + 2 * M), 0)
ImageDraw.Draw(shadow).rounded_rectangle([M, M + 8, M + W, M + H + 8], R, fill=255)
shadow = shadow.filter(ImageFilter.GaussianBlur(18))
alpha = (np.asarray(shadow, np.float64) * 0.10).astype(np.uint8)            # 10% black, soft
img = np.zeros((H + 2 * M, W + 2 * M, 4), np.uint8)
img[..., 3] = alpha
card = Image.new("L", (W + 2 * M, H + 2 * M), 0)
ImageDraw.Draw(card).rounded_rectangle([M, M, M + W, M + H], R, fill=255)
c = np.asarray(card) > 127
img[c] = (255, 255, 255, 255)
out = os.path.join(T, "prezi_upload", "CARD_tile.png")
Image.fromarray(img, "RGBA").save(out, optimize=True)
print(out, img.shape, "inner fraction", W / (W + 2 * M))
