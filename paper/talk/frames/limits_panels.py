"""Limits card: three standalone panels from production model 257's test battery (output/battery/257_255_249,
output/light_profiles_257): faint flux deficit vs magnitude (VIS curve of growth at 3″, sky-matched, against the Euclid input),
the faint pixel grid on a real NEXUS galaxy (display knee near the noise), and a hole in the poster galaxy's
bright nucleus (Euclid vs POLISHed, same stretch)."""
import os
import sys

sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish/paper_figures/generation_collages")
sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish")
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from build import BALANCE, BANDS, intensity, lupton

from euclid_polish.visualization import color
from style import BG, CARD, CORAL, MONO, MUTED, OUT, SANS, TEAL, TXT, save_panel

GRID_Y, GRID_X, GRID_WHITE = int(os.environ.get("GY", 96)), int(os.environ.get("GX", 288)), 8
ROOT = "/Users/alarion239/Desktop/EuclidPolish"
CACHE = os.path.join(ROOT, "output/battery/257_255_249/sr_cache/member_257@15000000-ee718c30")

# 1 · faint flux: median F_SR(<3″)/F_Euclid(<3″) in VIS per magnitude bin
d = np.load(os.path.join(ROOT, "output/light_profiles_257/cog.npz"))
r = d["radii_arcsec"]
i3 = int(np.argmin(np.abs(r - 3.0)))
off = np.nan_to_num(d["sky_offset"][:, 0])                       # e- per SR px, as in plot_cog.py
sr_m = d["cog_sr"][:, i3, 0] - off * d["area"][:, i3]
ratio = sr_m / d["cog_lr"][:, i3, 0]
mag = d["mag"]
edges = np.arange(20.0, 25.51, 0.5)
mids, med, lo, hi, ns = [], [], [], [], []
for a, b in zip(edges[:-1], edges[1:]):
    m = (mag >= a) & (mag < b) & np.isfinite(ratio)
    if m.sum() < 12:
        continue
    q = np.percentile(ratio[m], [25, 50, 75])
    mids.append((a + b) / 2); lo.append(q[0]); med.append(q[1]); hi.append(q[2]); ns.append(m.sum())
print("flux bins", [(round(x, 2), round(y, 3), n) for x, y, n in zip(mids, med, ns)])
px = 2160
fig = plt.figure(figsize=(px / 200, px / 200), dpi=200, facecolor="white")
ax = fig.add_axes([0.16, 0.14, 0.78, 0.70])
ax.fill_between(mids, lo, hi, color=TEAL, alpha=0.18, lw=0)
ax.plot(mids, med, "-o", color="#0E8C7E", lw=3, ms=9)
ax.axhline(1.0, color="0.35", lw=1.5, ls="--")
ax.set_ylim(0.7, 1.15)
ax.set_xlim(edges[0], edges[-1])
ax.set_xlabel("VIS magnitude", fontsize=22, family=SANS, color="#222")
ax.set_ylabel("POLISHed / Euclid flux (r < 3″)", fontsize=22, family=SANS, color="#222")
ax.tick_params(labelsize=18, colors="#333")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0.16, 0.92, "Faint galaxies lose flux", fontsize=34, family=SANS, weight="bold", color="#111")
fig.text(0.16, 0.875, f"{len(mag)} real galaxies (NEXUS) · sky-matched · median and quartiles", fontsize=18, family=SANS,
         color="#555")
fig.savefig(os.path.join(OUT, "LIM1_faint_flux.png"), dpi=200, facecolor="white")
plt.close(fig)

# 2 · faint pixel grid: NEXUS f200w-0038, VIS, the galaxy's outskirts (4.8″), display knee at 0.2 sigma
sr = np.load(os.path.join(CACHE, "nexus-f200w-0038__5f9146a62c59.npy"))[..., 0].astype(np.float64)
sky = np.median(sr)
k = 0.2 * 0.895
crop = sr[GRID_Y:GRID_Y + 96, GRID_X:GRID_X + 96]
x = np.arcsinh((crop - sky) / k)
img = np.clip((x + 1) / (np.arcsinh(GRID_WHITE * 0.895 / k) + 1), 0, 1)
save_panel(np.stack([img] * 3, -1), "LIM2_faint_grid.png", label="POLISHed · faint outskirts", interp="nearest",
           bar_arcsec=1.0)

# 3 · bright core: poster nucleus B in VIS, Euclid vs POLISHed, one asinh stretch (NISP saturates here)
lr = np.asarray(fits.getdata(os.path.join(ROOT, "poster/target_181255_lr.fits"), 1), np.float64)
srp = np.load(os.path.join(CACHE, "poster-target_181255__a5754d75cdff.npy"))[..., 0].astype(np.float64)
cy, cx, h = 512, 528, 64
up = np.repeat(np.repeat(lr[cy - h:cy + h, cx - h:cx + h], 2, 0), 2, 1) / 4.0
S = srp[2 * (cy - h):2 * (cy + h), 2 * (cx - h):2 * (cx + h)]
sk = np.median(lr) / 4.0
kk = np.percentile(up - sk, 50)
top = np.arcsinh(np.percentile(up - sk, 99.9) / kk)
# holes as in the battery (eval/battery holes): within the tile's bright-1 % LR pixels, SR < 1/2 LR/4
bright = np.repeat(np.repeat(lr[cy - h:cy + h, cx - h:cx + h] > np.percentile(lr, 99), 2, 0), 2, 1)
hole = bright & (S < 0.5 * up)
from scipy import ndimage
edge = hole & ~ndimage.binary_erosion(hole, iterations=1)
edge = ndimage.binary_dilation(edge, iterations=1)
print("hole px", int(hole.sum()), "of bright", int(bright.sum()))
coral = np.array([0xF0, 0x8A, 0x6C]) / 255.0
for nm, c, lab, mark in (("LIM3_core_1_euclid.png", up, "Euclid · bright nucleus (VIS)", False),
                         ("LIM3_core_2_polished.png", S, "POLISHed · holes outlined", True)):
    v = np.clip(np.arcsinh((c - sk) / kk) / top, 0, 1)
    rgb = np.stack([v] * 3, -1)
    if mark:
        rgb[edge] = coral
    save_panel(rgb, nm, label=lab, interp="nearest", bar_arcsec=2.0)
print("ok")
