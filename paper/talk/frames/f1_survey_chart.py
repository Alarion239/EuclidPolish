"""F1 (v2): survey area vs image sharpness, official numbers only. White figure for the Prezi card.

Area / delivered PSF FWHM (median), with sources:
  SDSS DR7 imaging       11,663 deg2   r 1.43"   Abazajian+09; classic.sdss.org/dr7 seeing
  Pan-STARRS1 3pi        30,000 deg2   i 1.11"   Chambers+16; Magnier+20
  DES DR2                 5,000 deg2   i 0.88"   Abbott+21 (DES DR2)
  HSC-SSP Wide (PDR3)       670 deg2   i 0.61"   Aihara+22 (full-depth, full-colour area)
  Euclid Wide            14,000 deg2   VIS 0.16" Euclid Q1 II (VIS processing), Scaramella+22
  Euclid DR1 (now)        1,900 deg2   VIS 0.16" DR1 (12 Nov 2026)
  HST COSMOS (ACS)         1.64 deg2   F814W ~0.09"  Scoville+07, Koekemoer+07
  HST UDF / XDF          0.0013 deg2   F775W ~0.09"  Illingworth+13
  JWST COSMOS-Web          0.54 deg2   F150W 0.061"  Casey+23; Zhuang+24
"""
import os

import matplotlib.pyplot as plt
from matplotlib import patheffects as pe

from style import OUT, SANS, MONO

GROUND, SPACE, EUCLID = "#5b6b82", "#0E8C7E", "#e2583e"
pts = [  # name, area deg2, fwhm arcsec, kind, label offset (dx, dy in points), ha
    ("SDSS", 11663, 1.43, "g", (10, -4), "left"),
    ("Pan-STARRS1", 30000, 1.11, "g", (-10, 8), "right"),
    ("DES", 5000, 0.88, "g", (-10, -2), "right"),
    ("HSC-SSP", 670, 0.61, "g", (-10, -2), "right"),
    ("Euclid Wide", 14000, 0.16, "e", (16, 0), "left"),
    ("Euclid DR1", 1900, 0.16, "e", (-18, 0), "right"),
    ("HST COSMOS", 1.64, 0.09, "s", (10, 6), "left"),
    ("HST UDF", 0.0013, 0.09, "s", (10, 6), "left"),
    ("JWST COSMOS-Web", 0.54, 0.061, "s", (10, -14), "left"),
]
fig = plt.figure(figsize=(16, 9), dpi=200, facecolor="white")
ax = fig.add_axes([0.085, 0.2, 0.86, 0.67])
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(5e-4, 1e5); ax.set_ylim(2.0, 0.04)        # sharper is up
ax.axhspan(2.0, 0.5, color="#eef1f5", zorder=0)
ax.text(7e-4, 1.75, "seeing-limited (atmosphere)", color="#7b8798", fontsize=17, family=SANS, va="bottom")
ax.text(7e-4, 0.045, "diffraction-limited (space)", color="#7b8798", fontsize=17, family=SANS, va="top")
for name, a, f, k, (dx, dy), ha in pts:
    col = {"g": GROUND, "s": SPACE, "e": EUCLID}[k]
    hollow = name == "Euclid Wide"
    ax.scatter(a, f, s=420 if k == "e" else 260, marker="o", facecolor="white" if hollow else col,
               edgecolor=col, linewidth=3, zorder=3)
    ax.annotate(name, (a, f), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                fontsize=22 if k == "e" else 19, family=SANS, weight="bold" if k == "e" else "medium", color=col,
                path_effects=[pe.withStroke(linewidth=4, foreground="white")], zorder=4)
ax.annotate("", xy=(14000, 0.165), xytext=(1900, 0.165), zorder=2,
            arrowprops=dict(arrowstyle="-|>", color=EUCLID, lw=2.5, mutation_scale=22, linestyle="--"))
ax.set_xlabel("Survey area  [deg²]", fontsize=22, family=SANS, color="#222")
ax.set_ylabel("Image sharpness: PSF FWHM  [arcsec]", fontsize=22, family=SANS, color="#222")
ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5])
ax.set_xticklabels(["0.001", "0.01", "0.1", "1", "10", "100", "1,000", "10,000", "100,000"])
ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1.0, 2.0]); ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2"])
ax.tick_params(labelsize=17, colors="#333", length=6)
ax.minorticks_off()
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#555")
ax.grid(True, which="major", color="#dde2e8", lw=1, zorder=0)
fig.text(0.085, 0.945, "Wide or sharp, never both — until Euclid", fontsize=34, family=SANS, weight="bold", color="#111")
fig.text(0.085, 0.905, "Median delivered PSF FWHM vs imaged area", fontsize=19, family=SANS, color="#555")
fig.text(0.085, 0.022, "Sources: SDSS DR7 (Abazajian+09); Pan-STARRS1 (Chambers+16); DES DR2 (Abbott+21); HSC-SSP PDR3 (Aihara+22); "
         "Euclid Q1 VIS paper (2025) and DR1;\nHST COSMOS (Scoville+07); HST XDF (Illingworth+13); JWST COSMOS-Web (Casey+23; Zhuang+24). "
         "Ground: median i/r-band seeing; space: VIS / F814W / F150W.", fontsize=13, family=SANS, color="#777", linespacing=1.5)
out = os.path.join(OUT, "F1v2_wide_or_sharp.png")
fig.savefig(out, dpi=200, facecolor="white")
print(out)
