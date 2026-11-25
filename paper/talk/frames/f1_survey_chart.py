"""F1 (v4, talk style): imaged area vs median delivered PSF FWHM, sharper UP and wider RIGHT, so the
top-right corner is best and only Euclid is there. Helvetica Neue, direct labels, no title/legend/footnote; Euclid's
filled dot is DR1 (released), the open one the full Wide survey.
v6 (meeting 11-23, "better show how Euclid lies on the plot"): the top-right corner (wider than 100 deg2 AND sharper
than 0.3") is a tinted box, "Wide and sharp", empty in the base version; in the Euclid version Euclid sits inside
it, the old frontier stays as a grey dashed line, the new one is drawn in the accent colour, and a dotted guide
reads Euclid's 0.16" off the axis.
v7 (user 11-25): dots with names only — no box, no frontier lines, no guides. The second version adds Euclid
(one dot, the 14,000 deg2 Wide survey, in the accent colour).

Sources (speaker notes, not on the plot):
  SDSS DR7 imaging 11,663 deg2, r 1.43" (Abazajian+09); Pan-STARRS1 3pi 30,000 deg2, i 1.11" (Chambers+16);
  DES DR2 5,000 deg2, i 0.88" (Abbott+21); HSC-SSP PDR3 Wide 670 deg2 full depth, i 0.61" (Aihara+22);
  Euclid DR1 1,900 deg2 / Wide 14,000 deg2, VIS 0.16" (Euclid Q1 VIS paper); HST COSMOS ACS 1.64 deg2,
  ~0.09" (Scoville+07); JWST COSMOS-Web 0.54 deg2, F150W 0.061" (Casey+23; Zhuang+24).
"""
import os

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from PIL import Image

from style import OUT, TALK

FONT = "Helvetica Neue"
INK, GREY, SPACE, ACCENT = "#1d2127", "#8a94a3", "#3d7ea6", "#e0553b"
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

def chart(with_euclid):
    """One chart; the talk shows it first without Euclid, then fades in the version with Euclid."""
    fig = plt.figure(figsize=(16, 9), dpi=200, facecolor="white")
    ax = fig.add_axes([0.12, 0.16, 0.84, 0.78])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*XL); ax.set_ylim(*YL)                          # sharper is up
    allp = pts + ([("Euclid", 14000, 0.16, ACCENT, (0, 40), "center")] if with_euclid else [])
    for name, a, f, c, (dx, dy), ha in allp:
        big = name == "Euclid"
        ax.scatter(a, f, s=620 if big else 340, color=c, edgecolor="white", linewidth=2, zorder=3)
        ax.annotate(name, (a, f), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=40 if big else 27, color=c, fontweight="bold" if big else "normal", zorder=4,
                    path_effects=[pe.withStroke(linewidth=8, foreground="white")])
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
    name = f"F1v7_wide_or_sharp_{'euclid' if with_euclid else 'base'}"
    out = os.path.join(OUT, name + ".png")
    fig.savefig(out, dpi=200, facecolor="white")
    Image.open(out).convert("RGB").resize((2400, 1350), Image.LANCZOS).save(
        os.path.join(TALK, "prezi_upload", name + ".jpg"), quality=92)
    print(out)


chart(False)
chart(True)
