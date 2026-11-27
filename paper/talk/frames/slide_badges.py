"""Slide-number badges for the Prezi (one per zoom stop).

Each view of the path gets a pill in its top-right corner with its first
visit number. A view the path returns to also gets an overlay pill per
return, listing every visit so far: earlier numbers grey, the current one
bold. The overlay is right-aligned with the badge it covers and fades in
with an existing click while the camera is elsewhere.

Output: prezi_upload/v23_badges.pdf (one vector page per badge, page size
= the pill) + prezi_upload/v23_badges.json (page -> visits, aspect).
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
VIEWS = {
    "hero": [1, 31], "c0": [2], "ladder": [3, 10], "c10": [4, 9],
    "win": [5, 8, 29], "c1": [6], "c2": [7], "c11": [11], "c3": [12],
    "c4": [13], "c5": [14], "c6": [15, 18], "c7": [16], "c12": [17],
    "c13": [19, 22], "lens1": [20], "lens2": [21], "c8": [23, 28],
    "g1": [24], "g2": [25], "g3": [26], "g4": [27], "c9": [30],
    "next": [32], "qr": [33], "close": [34],
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
    with PdfPages(os.path.join(up, "v23_badges.pdf")) as pdf:
        for i, (view, visits) in enumerate(badges()):
            prev = os.path.join(os.environ["BADGE_PREVIEW"], f"badge_{i}.png") if os.environ.get("BADGE_PREVIEW") else None
            asp = draw(pdf, visits, prev)
            manifest.append({"page": i + 1, "view": view, "visits": visits, "aspect": round(asp, 4)})
    with open(os.path.join(up, "v23_badges.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(len(manifest), "badges")


if __name__ == "__main__":
    main()
