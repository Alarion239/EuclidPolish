"""F1 (v4, talk style): imaged area vs median delivered PSF FWHM, sharper UP and wider RIGHT, so the
top-right corner is best and only Euclid is there. Helvetica Neue, direct labels, no title/legend/footnote; Euclid's
filled dot is DR1 (released), the open one the full Wide survey.

Sources (speaker notes, not on the plot):
  SDSS DR7 imaging 11,663 deg2, r 1.43" (Abazajian+09); Pan-STARRS1 3pi 30,000 deg2, i 1.11" (Chambers+16);
  DES DR2 5,000 deg2, i 0.88" (Abbott+21); HSC-SSP PDR3 Wide 670 deg2 full depth, i 0.61" (Aihara+22);
  Euclid DR1 1,900 deg2 / Wide 14,000 deg2, VIS 0.16" (Euclid Q1 VIS paper); HST COSMOS ACS 1.64 deg2,
  ~0.09" (Scoville+07); JWST COSMOS-Web 0.54 deg2, F150W 0.061" (Casey+23; Zhuang+24).
"""
import os

import matplotlib.pyplot as plt

from style import OUT

FONT = "Helvetica Neue"
INK, GREY, SPACE, ACCENT = "#1d2127", "#8a94a3", "#3d7ea6", "#e0553b"
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
    """One chart; the talk shows it first without Euclid, then fades in the version with Euclid. The faint dashed
    line is the Pareto front (no survey both wider and sharper); with Euclid it runs through Euclid."""
    fig = plt.figure(figsize=(16, 9), dpi=200, facecolor="white")
    ax = fig.add_axes([0.12, 0.16, 0.84, 0.78])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.2, 1.5e5); ax.set_ylim(1.9, 0.045)            # sharper is up
    for name, a, f, c, (dx, dy), ha in pts:
        ax.scatter(a, f, s=340, color=c, edgecolor="white", linewidth=2, zorder=3)
        ax.annotate(name, (a, f), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=27, color=c, zorder=4)
    allp = [(a, f) for _, a, f, *_ in pts] + ([(14000, 0.16)] if with_euclid else [])
    front = sorted(p for p in allp if not any(q[0] >= p[0] and q[1] <= p[1] and q != p for q in allp))
    fx, fy = zip(*front)
    ax.plot(fx, fy, ls=(0, (4, 4)), lw=2, color=ACCENT if with_euclid else GREY, alpha=0.55, zorder=1)
    if with_euclid:
        ax.plot([1900, 14000], [0.16, 0.16], color=ACCENT, lw=3, zorder=2, solid_capstyle="butt")
        ax.scatter(1900, 0.16, s=560, color=ACCENT, edgecolor="white", linewidth=2, zorder=3)
        ax.scatter(14000, 0.16, s=560, facecolor="white", edgecolor=ACCENT, linewidth=3.5, zorder=3)
        ax.annotate("Euclid", (5150, 0.16), xytext=(0, 34), textcoords="offset points", ha="center",
                    va="center", fontsize=34, color=ACCENT, fontweight="bold")
        for a, txt in ((1900, "DR1 today"), (14000, "full survey")):   # filled = released, open = planned
            ax.annotate(txt, (a, 0.16), xytext=(0, -32), textcoords="offset points", ha="center", va="center",
                        fontsize=22, color=ACCENT)
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
    out = os.path.join(OUT, f"F1v5_wide_or_sharp_{'euclid' if with_euclid else 'base'}.png")
    fig.savefig(out, dpi=200, facecolor="white")
    print(out)


chart(False)
chart(True)
