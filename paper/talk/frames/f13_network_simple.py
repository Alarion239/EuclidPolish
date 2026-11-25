"""Network card (11-25 review): the paper's architecture figure is too dense for a talk, so this is a 4-step
version -- Euclid image (VIS + Y + J + H, 0.1" grid) -> arcsinh at 8 knees -> 16 residual blocks (one shared trunk)
-> 2x upsample -> the same four bands on a 0.05" grid -- with three big numbers. Thumbnails are the title galaxy
(hero_16x9.py: NEXUS f200w-0253), Euclid and POLISHed, one stretch. The paper figure stays the backup.
Sized to the card's picture area under the title (16,885 x 7,000 Prezi units, aspect 2.41), white like the card;
the card's own text "15M steps = 70 A100-hours" sits under it."""
import os

import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np

from ab_colour import ab_rgb, auto_levels, noise_sigma, raw_triple
from style import MONO, SANS, TALK

INK, MUTE, TEAL, PALE = "#1d2127", "#6b7380", "#18a593", "#e7f7f4"
TILE, CX, CY, HW = "f200w-0253", 123, 273, 56
lr, sr, jw = raw_triple(TILE)
sq = (slice(CY - HW, CY + HW), slice(CX - HW, CX + HW))
lev = auto_levels(jw[sq], q=20, black_sigma=2.0, sigma=noise_sigma(lr[sq]))
thumbs = [np.clip(ab_rgb(c[sq], **lev), 0, 1) for c in (lr, sr)]

W, H = 24.1, 10.0
fig = plt.figure(figsize=(W, H), dpi=300, facecolor="white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

yc, s = 5.9, 3.6                                    # row centre, thumbnail side
ax.imshow(thumbs[0], origin="lower", extent=(0.35, 0.35 + s, yc - s / 2, yc + s / 2), interpolation="nearest")
ax.imshow(thumbs[1], origin="lower", extent=(W - 0.35 - s, W - 0.35, yc - s / 2, yc + s / 2), interpolation="lanczos")
for x, t1, t2 in ((0.35 + s / 2, "Euclid", "VIS·Y·J·H, 0.1″ grid"), (W - 0.35 - s / 2, "POLISHed", "VIS·Y·J·H, 0.05″ grid")):
    ax.text(x, yc + s / 2 + 0.6, t1, ha="center", va="bottom", fontsize=40, family=SANS, weight="bold",
            color=TEAL if t1 == "POLISHed" else INK)
    ax.text(x, yc + s / 2 + 0.15, t2, ha="center", va="bottom", fontsize=22, family=SANS, color=MUTE)

blocks = [("arcsinh\nat 8 knees", "8", "knees: faint and\nbright, one input"),
          ("16 residual\nblocks", "16", "blocks, one trunk\nshared by 4 bands"),
          ("upsample", "2×", "finer grid\nout")]
x0, x1 = 0.35 + s + 0.7, W - 0.35 - s - 0.7
bw = (x1 - x0 - 2 * 0.7) / 3
bh = 1.9
xs = [x0 + k * (bw + 0.7) for k in range(3)]
arrow = dict(arrowstyle="-|>", color=INK, lw=2.2, mutation_scale=26, shrinkA=0, shrinkB=0)
ax.annotate("", xy=(xs[0] - 0.05, yc), xytext=(0.35 + s + 0.05, yc), arrowprops=arrow)
for k, (x, (name, big, sub)) in enumerate(zip(xs, blocks)):
    main = k == 1
    ax.add_patch(mp.FancyBboxPatch((x, yc - bh / 2), bw, bh, boxstyle="round,pad=0,rounding_size=0.18",
                                   fc=TEAL if main else PALE, ec=TEAL, lw=2.4))
    ax.text(x + bw / 2, yc, name, ha="center", va="center", fontsize=28, family=SANS, weight="bold",
            color="white" if main else INK, linespacing=1.15)
    nxt = xs[k + 1] if k < 2 else W - 0.35 - s
    ax.annotate("", xy=(nxt - 0.05, yc), xytext=(x + bw + 0.05, yc), arrowprops=arrow)
    ax.text(x + bw * 0.42, 1.2, big, ha="right", va="baseline", fontsize=96, family=SANS, weight="bold", color=TEAL)
    ax.text(x + bw * 0.47, 1.2 + 0.62, sub, ha="left", va="center", fontsize=24, family=SANS, color=MUTE, linespacing=1.3)
# the per-band skip: the input, upsampled, is added to the output (the network learns the residual)
ys = yc + bh / 2 + 0.75
xa, xb = 0.35 + s + 0.35, xs[2] + bw + 0.35                 # from the input arrow to the output arrow
dash = dict(color=MUTE, lw=1.6, ls=(0, (4, 3)))
ax.plot([xa, xa, xb], [yc, ys, ys], **dash)
ax.annotate("", xy=(xb, yc + 0.08), xytext=(xb, ys),
            arrowprops=dict(arrowstyle="-|>", mutation_scale=20, shrinkA=0, shrinkB=0, **dash))
ax.text((xa + xb) / 2, ys + 0.12, "+ the input itself, upsampled: the network learns only the correction",
        ha="center", va="bottom", fontsize=21, family=SANS, color=MUTE)
# (no footer: the card carries "15M steps = 70 A100-hours")
out_png = os.path.join(TALK, "images", "F13_network_simple.png")
fig.savefig(out_png, dpi=300, facecolor="white")
plt.close(fig)
from PIL import Image  # noqa: E402
os.makedirs(os.path.join(TALK, "prezi_upload", "v18"), exist_ok=True)
Image.open(out_png).convert("RGB").save(os.path.join(TALK, "prezi_upload", "v18", "NET_simple.jpg"), quality=92)
print(out_png)
