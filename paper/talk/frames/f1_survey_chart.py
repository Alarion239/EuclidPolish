"""F1 (v4, talk style): imaged area vs median delivered PSF FWHM, sharper UP and wider RIGHT, so the
top-right corner is best and only Euclid is there. Helvetica Neue, direct labels, no title/legend/footnote; Euclid's
filled dot is DR1 (released), the open one the full Wide survey.
v6 (meeting 11-23, "better show how Euclid lies on the plot"): the top-right corner (wider than 100 deg2 AND sharper
than 0.3") is a tinted box, "Wide and sharp", empty in the base version; in the Euclid version Euclid sits inside
it, the old frontier stays as a grey dashed line, the new one is drawn in the accent colour, and a dotted guide
reads Euclid's 0.16" off the axis.

Sources (speaker notes, not on the plot):
  SDSS DR7 imaging 11,663 deg2, r 1.43" (Abazajian+09); Pan-STARRS1 3pi 30,000 deg2, i 1.11" (Chambers+16);
  DES DR2 5,000 deg2, i 0.88" (Abbott+21); HSC-SSP PDR3 Wide 670 deg2 full depth, i 0.61" (Aihara+22);
  Euclid DR1 1,900 deg2 / Wide 14,000 deg2, VIS 0.16" (Euclid Q1 VIS paper); HST COSMOS ACS 1.64 deg2,
  ~0.09" (Scoville+07); JWST COSMOS-Web 0.54 deg2, F150W 0.061" (Casey+23; Zhuang+24).
"""
import os

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from style import OUT, TALK

FONT = "Helvetica Neue"
INK, GREY, SPACE, ACCENT, TINT = "#1d2127", "#8a94a3", "#3d7ea6", "#e0553b", "#fdf3f0"
XL, YL = (0.2, 1.5e5), (1.9, 0.045)
plt.rcParams.update({"font.family": FONT, "axes.linewidth": 1.6, "xtick.major.width": 1.6,
                     "ytick.major.width": 1.6, "xtick.major.size": 8, "ytick.major.size": 8,
                     "xtick.direction": "out", "ytick.direction": "out"})
pts = [  # name, area, fwhm, colour, label offset (points), ha
    ("SDSS", 11663, 1.43, GREY, (-22, 0), "right"),
    ("Pan-STARRS", 30000, 1.11, GREY, (0, 34), "center"),
    ("DES", 5000, 0.88, GREY, (-20, 0), "right"),
    ("HSC", 670, 0.61, GREY, (-20, 0), "right"),
    ("Hubble", 1.64, 0.09, SPACE, (-20, 0), "right"),
    ("JWST", 0.54, 0.061, SPACE, (20, 0), "left"),
]

def front_of(points):
    return sorted(p for p in points if not any(q[0] >= p[0] and q[1] <= p[1] and q != p for q in points))


def corner_box(ax, with_euclid):
    """The wide-and-sharp corner: > 100 deg2 and < 0.3"."""
    ax.add_patch(Rectangle((100, YL[1]), XL[1] - 100, 0.3 - YL[1], facecolor=TINT, edgecolor=ACCENT, lw=2,
                           ls=(0, (6, 4)), alpha=1.0, zorder=0))
    ax.text(1.2e5, 0.052, "Wide and sharp", ha="right", va="top", fontsize=30, color=ACCENT, fontweight="bold")
    if not with_euclid:
        ax.text(1.2e5, 0.075, "no survey yet", ha="right", va="top", fontsize=26, color=ACCENT,
                fontstyle="italic")


def chart(with_euclid):
    """One chart; the talk shows it first without Euclid, then fades in the version with Euclid."""
    fig = plt.figure(figsize=(16, 9), dpi=200, facecolor="white")
    ax = fig.add_axes([0.12, 0.16, 0.84, 0.78])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*XL); ax.set_ylim(*YL)                          # sharper is up
    base = [(a, f) for _, a, f, *_ in pts]
    corner_box(ax, with_euclid)
    for name, a, f, c, (dx, dy), ha in pts:
        ax.scatter(a, f, s=340, color=c, edgecolor="white", linewidth=2, zorder=3)
        ax.annotate(name, (a, f), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=27, color=c, zorder=4, path_effects=[pe.withStroke(linewidth=8, foreground="white")])
    fx, fy = zip(*front_of(base))
    ax.plot(fx, fy, ls=(0, (4, 4)), lw=2.2, color=GREY, alpha=0.8, zorder=1)
    if with_euclid:
        nx, ny = zip(*front_of(base + [(14000, 0.16)]))
        ax.plot(nx, ny, lw=3, color=ACCENT, alpha=0.9, zorder=1)
        ax.plot([1900, XL[0]], [0.16, 0.16], ls=(0, (1, 3)), lw=2, color=ACCENT, alpha=0.8, zorder=1)
        ax.plot([1900, 14000], [0.16, 0.16], color=ACCENT, lw=4, zorder=2, solid_capstyle="butt")
        ax.scatter(1900, 0.16, s=620, color=ACCENT, edgecolor="white", linewidth=2, zorder=3)
        ax.scatter(14000, 0.16, s=620, facecolor="white", edgecolor=ACCENT, linewidth=4, zorder=3)
        ax.annotate("Euclid", (5150, 0.16), xytext=(0, 40), textcoords="offset points", ha="center",
                    va="center", fontsize=40, color=ACCENT, fontweight="bold")
        for a, txt in ((1900, "DR1 today"), (14000, "full survey")):   # filled = released, open = planned
            ax.annotate(txt, (a, 0.16), xytext=(0, -36), textcoords="offset points", ha="center", va="center",
                        fontsize=23, color=ACCENT)
        ax.annotate("0.16″", (XL[0], 0.16), xytext=(8, 14), textcoords="offset points", ha="left",
                    va="center", fontsize=22, color=ACCENT)
    ax.set_xlabel("Area  [deg²]", fontsize=28, color=INK, labelpad=12)
    ax.set_ylabel("PSF FWHM  [arcsec]", fontsize=28, color=INK, labelpad=12)
    ax.set_xticks([1, 100, 1e4]); ax.set_xticklabels(["1", "100", "10 000"])
    ax.set_yticks([1.0, 0.3, 0.1]); ax.set_yticklabels(["1", "0.3", "0.1"])
    ax.minorticks_off()
    ax.tick_params(labelsize=24, colors=INK, pad=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK)
    name = f"F1v6_wide_or_sharp_{'euclid' if with_euclid else 'base'}"
    out = os.path.join(OUT, name + ".png")
    fig.savefig(out, dpi=200, facecolor="white")
    Image.open(out).convert("RGB").resize((2400, 1350), Image.LANCZOS).save(
        os.path.join(TALK, "prezi_upload", name + ".jpg"), quality=92)
    print(out)


chart(False)
chart(True)
