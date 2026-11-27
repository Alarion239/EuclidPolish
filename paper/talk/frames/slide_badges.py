"""Slide-number badges for the Prezi (one per zoom stop).

Each view of the path gets a pill in its top-right corner with its first
visit number. A view the path returns to also gets an overlay pill per
return, listing every visit so far: earlier numbers grey, the current one
bold. The overlay is right-aligned with the badge it covers and fades in
with an existing click while the camera is elsewhere.

Output: prezi_upload/v25_badges.pdf (one vector page per badge, page size
= the pill) + prezi_upload/v25_badges.json (page -> visits, aspect).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

from style import SANS, TALK

INK = "#16202a"
GREY = "#9aa3ad"
FILL = "#e9edf1"

# view -> visit numbers (slide = zoom stop, as in the Prezi sidebar)
VIEWS = {   # 11-27 numbering: the chart zoom is gone, the chart return is the whole card
    "hero": [1, 30], "c0": [2, 9], "c10": [3, 8], "win": [4, 7, 28], "c1": [5], "c2": [6], "c11": [10],
    "c3": [11], "c4": [12], "c5": [13], "c6": [14, 17], "c7": [15], "c12": [16], "c13": [18, 21],
    "lens1": [19], "lens2": [20], "c8": [22, 27], "g1": [23], "g2": [24], "g3": [25], "g4": [26], "c9": [29],
    "next": [31], "qr": [32], "close": [33],
}

H = 1.0          # pill height (inches, arbitrary: Prezi rescales)
FS = 40          # number size (pt) at that height
PAD = 0.32       # horizontal padding (inches)
GAP = 0.20       # gap between numbers


def badges():
    out = []
    for view, visits in VIEWS.items():
        for k in range(1, len(visits) + 1):
            out.append((view, visits[:k]))
    return out


def text_width(fig, s, weight):
    t = fig.text(0, 0, s, fontsize=FS, family=SANS, weight=weight)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def draw(pdf, visits, preview=None):
    probe = plt.figure(figsize=(4, 1), dpi=200)
    parts = [(str(v), "medium", GREY) for v in visits[:-1]] + [(str(visits[-1]), "bold", INK)]
    widths = [text_width(probe, s, w) for s, w, _ in parts]
    plt.close(probe)
    W = 2 * PAD + sum(widths) + GAP * (len(parts) - 1)
    W = max(W, 1.25 * H)
    fig = plt.figure(figsize=(W, H), dpi=200)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), W, H, boxstyle=f"round,pad=0,rounding_size={0.5 * H}",
                                fc=FILL, ec="none"))
    x = 0.5 * (W - sum(widths) - GAP * (len(parts) - 1))
    for (s, wgt, col), w in zip(parts, widths):
        ax.text(x, 0.5 * H, s, fontsize=FS, family=SANS, weight=wgt, color=col,
                ha="left", va="center_baseline")
        x += w + GAP
    pdf.savefig(fig, transparent=True)
    if preview:
        fig.savefig(preview, transparent=False, facecolor="#444")
    plt.close(fig)
    return W / H


def main():
    up = os.path.join(TALK, "prezi_upload")
    manifest = []
    with PdfPages(os.path.join(up, "v25_badges.pdf")) as pdf:
        for i, (view, visits) in enumerate(badges()):
            prev = os.path.join(os.environ["BADGE_PREVIEW"], f"badge_{i}.png") if os.environ.get("BADGE_PREVIEW") else None
            asp = draw(pdf, visits, prev)
            manifest.append({"page": i + 1, "view": view, "visits": visits, "aspect": round(asp, 4)})
    with open(os.path.join(up, "v25_badges.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(len(manifest), "badges")


if __name__ == "__main__":
    main()
