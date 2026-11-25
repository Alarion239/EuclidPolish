"""Survey area vs. sharpness, on one spot of sky: concentric boxes around the HUDF spiral whose AREA is
proportional to each survey's sky coverage (side ∝ sqrt(area), JWST = 12"), each filled with that survey's
own image of this sky. Two PDF pages per survey: the wide box at a coarse pixel (from fetch_ladder_wide.py /
fetch_euclid_wide.py), and an inset page of native-resolution data around the zoom window (Prezi renders a
PDF page at a capped resolution, so one page cannot stay sharp through a ~100x zoom). Labels are vector text: survey · area on
the box, survey · PSF at the zoom window's top-left.

Areas: DSS2 the whole sky (41,253 deg²), Pan-STARRS 3π 30,000, Euclid Wide 14,000, DES 5,000 (shown with
Legacy Surveys DR10 DECam images, mostly DES data here), Hubble 1.64 (COSMOS, the largest HST mosaic) and
JWST 0.54 (COSMOS-Web), the same numbers as the wide-or-sharp chart.

Writes images/AL_<survey>.pdf (one page each), the walk-down layers of the zoom window (see space():
AL_hst_over_ls.pdf, AL_{hst,jwst}_{area,res}.pdf) and images/AL_geometry.json (sky offsets for the Prezi layout).

Zoom window, clicks 10-16 (each a fade-IN on top, so area labels and resolution labels never mix):
10 zoom onto the window: Hubble box "Hubble · 1.6 deg²" (AL_hst_over_ls, visible from the start; it also hides
   the DES inset's label)  11 + "JWST · 0.5 deg²"  12 + "JWST · 0.06″"  13 + "Hubble · 0.09″" (covers the JWST box)
14 + a copy of the DES inset ("DES · 0.9″")  15 + Pan-STARRS inset  16 + DSS2 inset.
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("pdf")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.stats import sigma_clipped_stats  # noqa: E402
from astropy.wcs import WCS  # noqa: E402
from scipy.ndimage import gaussian_filter, map_coordinates  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hudf_ladder as L  # noqa: E402
from style import OUT, SANS  # noqa: E402
sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish/services/cutout")
from cutout_server import SIM_ZEROPOINT_E  # noqa: E402

D = L.D
GRA, GDEC = 53.17468, -27.799422                 # the HUDF spiral: centre of every box
COSD = np.cos(np.radians(GDEC))
AREA = {"dss": 41253, "ps1": 30000, "euclid": 14000, "ls": 5000, "hst": 1.64, "jwst": 0.54}
SIDE = {k: 12.0 * np.sqrt(a / AREA["jwst"]) for k, a in AREA.items()}
# zoom window: 16:9, inside the 102.4" Euclid / SR / XDF / JADES footprint around L.RA0, L.DEC0
WIN_E, WIN_N, WIN_W, WIN_H = 25.8, -28.9, 50.4, 28.35   # centre offset from (RA0, DEC0), size, arcsec
WIN_RA = L.RA0 + WIN_E / 3600 / COSD
WIN_DEC = L.DEC0 + WIN_N / 3600
STROKE = [pe.withStroke(linewidth=3, foreground="black")]


def grid(ra, dec, scale, nx, ny):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [(nx + 1) / 2, (ny + 1) / 2]
    w.wcs.cdelt = [-scale / 3600, scale / 3600]
    return w


def resample(img, w_in, ra, dec, scale, wx, wy=None, nan_out=False):
    """img onto a north-up grid of wx × wy arcsec centred on (ra, dec); NaN where img does not reach."""
    wy = wx if wy is None else wy
    nx, ny = int(round(wx / scale)), int(round(wy / scale))
    wo = grid(ra, dec, scale, nx, ny)
    yy, xx = np.mgrid[0:ny, 0:nx]
    r, d = wo.pixel_to_world_values(xx, yy)
    x, y = w_in.world_to_pixel_values(r, d)
    good = np.isfinite(img)
    v = map_coordinates(np.where(good, img, 0.0), [y, x], order=1, mode="constant", cval=np.nan)
    m = map_coordinates(good.astype(float), [y, x], order=1, mode="constant", cval=0.0)
    out = np.where(m > 0.5, v / np.maximum(m, 1e-6), np.nan)
    return out if nan_out else np.nan_to_num(out)


def mosaic(files, ext, ra, dec, scale, size, zero_is_blank=True):
    acc = []
    for f in files:
        with fits.open(f) as h:
            a = np.asarray(h[ext].data, np.float64)
            w = WCS(h[ext].header).celestial
        if zero_is_blank:
            a[a == 0] = np.nan
        acc.append(resample(a, w, ra, dec, scale, size, nan_out=True))
    with np.errstate(all="ignore"):
        return np.nanmean(np.stack(acc), 0)


def match(ov, ov_scale, patch, p_ra, p_dec, p_scale, p_w, p_h):
    """Factor k with ov ≈ k · patch (units / pixel-area differences), from the bright pixels they share:
    the overview resampled onto the patch grid vs. the patch smoothed to the overview's pixel."""
    w_ov = grid(GRA, GDEC, ov_scale, ov.shape[1], ov.shape[0])
    o = resample(ov, w_ov, p_ra, p_dec, p_scale, p_w, p_h, nan_out=True)
    p = gaussian_filter(patch, ov_scale / p_scale / 2)
    _, mp, sp = sigma_clipped_stats(p, sigma=3, maxiters=5)
    ok = np.isfinite(o) & (p - mp > 15 * sp)
    _, mo, _ = sigma_clipped_stats(o[np.isfinite(o)], sigma=3, maxiters=5)
    return np.sum(o[ok] - mo) / np.sum(p[ok] - mp)


def colour(planes, med, white, Q, own_sky=True, chroma_px=0.0, cap=False):
    """asinh colour; each plane minus its OWN sky (sigma-clipped median) when own_sky, so a wide
    overview and its native inset agree even if their backgrounds were subtracted differently.
    chroma_px > 0 smooths only the colour (channel ratios) with a Gaussian of that sigma in pixels and
    keeps the brightness at full resolution, so per-pixel noise shows as grey grain, not coloured
    speckle; the sharpness of every source is unchanged.
    cap (11-25 review, Euclid only): a channel may exceed CAP_RATIO x the brighter of the other two by at most
    3 sigma of its own noise. Single-band detector residuals (green Y+J crescents, red H-only and blue VIS-only
    specks with nothing in the other bands) drop to the noise; real red or blue galaxies, which have light in
    every band, are untouched."""
    if own_sky:
        med = [sigma_clipped_stats(p[np.isfinite(p) & (p != 0)], sigma=3, maxiters=5)[1] for p in planes]
    out = [np.clip(np.nan_to_num(p - m), 0, None) / w for p, m, w in zip(planes, med, white)]
    if cap:
        sig = [sigma_clipped_stats(o[o > 0], sigma=3, maxiters=5)[2] for o in out]
        out = [np.minimum(o, CAP_RATIO * np.maximum(out[(i + 1) % 3], out[(i + 2) % 3]) + 3 * sig[i])
               for i, o in enumerate(out)]
    I = (out[0] + out[1] + out[2]) / 3 + 1e-12
    if chroma_px > 0:
        sm = [gaussian_filter(o, chroma_px) for o in out]
        Is = (sm[0] + sm[1] + sm[2]) / 3
        eps = 3 * max(sigma_clipped_stats(x, sigma=3, maxiters=5)[2] for x in sm)   # sky (noise only) -> grey
        out = [(s + eps) / (Is + eps) * I for s in sm]
    f = np.arcsinh(Q * I) / np.arcsinh(Q) / I
    return np.clip(np.stack([o * f for o in out], -1), 0, 1)


def stretch(planes, inst):
    med, white = [], []
    for p in planes:
        _, m, sd = sigma_clipped_stats(p, sigma=3, maxiters=5)
        med.append(m)
        white.append(max(np.percentile(p - m, 99.7), L.FLOOR.get(inst, 0) * sd))
    return med, white


def off(ra, dec):
    """(west, north) offset in arcsec of (ra, dec) from the spiral: x right = west on a north-up image."""
    return -(ra - GRA) * COSD * 3600, (dec - GDEC) * 3600


def page(name, side, layers, labels, centre=(GRA, GDEC), outline=None):
    """One PDF page covering a side × side box around `centre`. layers: (rgb, ra, dec, w, h);
    labels: (text, west, north, size_arcsec) at the top-left anchor."""
    cx, cy = off(*centre)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(cx - side / 2, cx + side / 2)
    ax.set_ylim(cy - side / 2, cy + side / 2)
    ax.axis("off")
    fig.patch.set_facecolor("black")
    for img, ra, dec, w, h in layers:
        x, y = off(ra, dec)
        ax.imshow(img, origin="lower", extent=(x - w / 2, x + w / 2, y - h / 2, y + h / 2),
                  interpolation="none", zorder=1)
    for text, x, y, size, *ha in labels:
        pts = size / side * 720                              # arcsec -> points on a 10-inch page
        ax.text(x, y, text, color="white", fontsize=pts, family=SANS, weight="bold", va="top",
                ha=ha[0] if ha else "left",
                path_effects=[pe.withStroke(linewidth=pts / 6, foreground="black")], zorder=3)
    if outline:                                              # line width in points on the 10-inch page
        d = outline / 720 * side / 2
        ax.add_patch(plt.Rectangle((cx - side / 2 + d, cy - side / 2 + d), side - 2 * d, side - 2 * d, fill=False,
                                   edgecolor="white", alpha=0.85, linewidth=outline, zorder=2))
    path = os.path.join(OUT, f"AL_{name}.pdf")
    fig.savefig(path, facecolor="black")
    plt.close(fig)
    print("wrote", path, flush=True)


# labels: box label = survey · area, top-left of its box; window label = survey · PSF, top-left of the window
WX, WY = off(WIN_RA, WIN_DEC)
WIN_TL = (WX - WIN_W / 2 + 0.03 * WIN_H, WY + WIN_H / 2 - 0.03 * WIN_H)
# Euclid and POLISHed window labels sit top-RIGHT (11-25 review): top-left, "Euclid · 0.16″" crowded the Hubble
# box label right under it when the box fades in on top
WIN_TR = (WX + WIN_W / 2 - 0.03 * WIN_H, WY + WIN_H / 2 - 0.03 * WIN_H)
BOX_LABEL = 0.035 * SIDE["dss"]              # one physical size for every box label on the card
WIN_LABEL = 0.075 * WIN_H
SPACE_LABEL = 0.95                           # Hubble / JWST box labels, arcsec: readable at the zoom window
INSET = 52.0                                 # inset page side, arcsec: just covers the 50.4" x 28.35" window
CAP_RATIO = 2.5                              # single-band residual cap (colour(cap=True)), Euclid and POLISHed
CHROMA = {"ground": 1.0, "euclid": 0.25}     # colour-only smoothing of the insets, arcsec (at or below each PSF)
OUTLINE = 2.5                                # box outline on the DSS2 page, points; same width on the card for all


def box_tl(inst):
    s = SIDE[inst]
    return -s / 2 + 0.012 * SIDE["dss"], s / 2 - 0.012 * SIDE["dss"]


NAMES = {"dss": ("DSS2", "all sky", "2″"), "ps1": ("Pan-STARRS", "30 000 deg²", "1.1″"),
         "ls": ("DES", "5 000 deg²", "0.9″"), "euclid": ("Euclid", "14 000 deg²", "0.16″"),
         "hst": ("Hubble", "1.6 deg²", "0.09″"), "jwst": ("JWST", "0.5 deg²", "0.06″")}


def ground(inst):
    name, area, psf = NAMES[inst]
    s = SIDE[inst]
    if inst == "dss":
        ov_scale, pw = 1.0, None
        ov = [mosaic([os.path.join(D, "dsswide_red.fits")], 0, GRA, GDEC, ov_scale, s)] * 3
        patch = None
    else:
        if inst == "ps1":
            ov_scale, p_scale, pw = 1.0, 0.25, 160.0
            ov = []
            for b in ("i", "r", "g"):        # 1" cutouts, gaps at projection-cell seams filled by whole-cell 2" ones
                fine = mosaic(sorted(glob.glob(os.path.join(D, f"ps1wide_{b}_*.fits"))), 0, GRA, GDEC, ov_scale, s)
                fill = mosaic(sorted(glob.glob(os.path.join(D, f"ps1fill_{b}_*.fits"))), 0, GRA, GDEC, ov_scale, s)
                ov.append(np.where(np.isfinite(fine), fine, fill))
            src = [L.load(f"ps1big_{b}.fits") for b in ("i", "r", "g")]
        else:
            ov_scale, p_scale, pw = 1.0, 0.262, 160.0
            with fits.open(os.path.join(D, "lswide_griz.fits")) as h:
                a = np.asarray(h[0].data, np.float64); w = WCS(h[0].header).celestial
            ov = [resample(a[i], w, GRA, GDEC, ov_scale, s) for i in (3, 1, 0)]
            big, wb = L.load("lsbig_griz.fits")
            src = [(big[i], wb) for i in (3, 1, 0)]
        # a light smoothing (well below the seeing) so the inset's pixel noise does not show as a brighter square
        patch = [gaussian_filter(resample(a, w, WIN_RA, WIN_DEC, p_scale, pw), 1.0 if inst == "ps1" else 0.7)
                 for a, w in src]
        ks = [match(o, ov_scale, p, WIN_RA, WIN_DEC, p_scale, pw, pw) for o, p in zip(ov, patch)]
        print(inst, "overview/patch factors", np.round(ks, 4), flush=True)
        ov = [o / k for o, k in zip(ov, ks)]
    if patch is None:                                       # DSS2: its own 1" data is the inset
        patch = [resample(ov[0], grid(GRA, GDEC, 1.0, ov[0].shape[1], ov[0].shape[0]), WIN_RA, WIN_DEC, 0.25, 160.0)] * 3
        pw = 160.0
    med, white = stretch(patch, inst)
    Q = L.INST[inst][3]
    page(inst, s, [(colour(ov, med, white, Q), GRA, GDEC, s, s)], [(f"{name} · {area}", *box_tl(inst), BOX_LABEL)],
         outline=OUTLINE * SIDE["dss"] / s)
    # the inset page only covers the zoom window (a ~10 px square on the card), cropped from the 160" patch
    inset = colour(patch, med, white, Q, chroma_px=CHROMA["ground"] / p_scale if inst != "dss" else 0)
    page(inst + "_inset", INSET, [(inset, WIN_RA, WIN_DEC, pw, pw)],
         [(f"{name} · {psf}", *WIN_TL, WIN_LABEL)], centre=(WIN_RA, WIN_DEC))
    if inst == "ls":
        COVER["ls"] = (inset, pw)


def euclid():
    name, area, psf = NAMES["euclid"]
    s, ov_scale = SIDE["euclid"], 0.8
    ov = []
    for b in ("H_E", "Y_E", "J_E", "VIS"):              # archive ADU/s -> electrons, as original_stack
        files = sorted(glob.glob(os.path.join(D, f"euwide_*_{b}.fits")))
        to_e = 10 ** ((SIM_ZEROPOINT_E[b] - fits.getheader(files[0])["MAGZERO"]) / 2.5)
        ov.append(to_e * mosaic(files, 0, GRA, GDEC, ov_scale, s))
    ov = [ov[0], 0.5 * (ov[1] + ov[2]), ov[3]]
    src = L.bands("euclid")                                   # original_stack (H, Y+J, VIS), 102.4" at RA0
    patch = [resample(a, w, L.RA0, L.DEC0, 0.1, L.FIELD) for a, w in src]
    ks = [match(o, ov_scale, p, L.RA0, L.DEC0, 0.1, L.FIELD, L.FIELD) for o, p in zip(ov, patch)]
    print("euclid overview/patch check (should be ~1)", np.round(ks, 3), flush=True)
    med, white = stretch(patch, "euclid")
    Q = L.INST["euclid"][3]
    page("euclid", s, [(colour(ov, med, white, Q, cap=True), GRA, GDEC, s, s)], [(f"{name} · {area}", *box_tl("euclid"), BOX_LABEL)],
         outline=OUTLINE * SIDE["dss"] / s)
    page("euclid_inset", L.FIELD, [(colour(patch, med, white, Q, chroma_px=CHROMA["euclid"] / 0.1, cap=True), L.RA0, L.DEC0, L.FIELD, L.FIELD)],
         [(f"{name} · {psf}", *WIN_TR, WIN_LABEL, "right")], centre=(L.RA0, L.DEC0))
    # POLISHed: the 102.4" super-resolved field, same colour (white / 4 for the 2x finer pixel)
    sr = [resample(a, w, L.RA0, L.DEC0, 0.05, L.FIELD) for a, w in L.bands("polished")]
    pol = colour(sr, [m / 4 for m in med], [v / 4 for v in white], Q, cap=True)
    x0, y0 = off(L.RA0, L.DEC0)
    page("polished", L.FIELD, [(pol, L.RA0, L.DEC0, L.FIELD, L.FIELD)], [("POLISHed Euclid", *WIN_TR, WIN_LABEL, "right")],
         centre=(L.RA0, L.DEC0))


COVER = {}                                   # the DES inset image, kept for hst_over_ls()


def hst_over_ls(hst_img, label):
    """The Hubble box drawn on a page with the DES inset's geometry, plus a patch of the DES inset itself
    (same pixels, no text) over the "DES · 0.9″" window label. In Prezi this page sits exactly on the DES inset
    and is visible from the start, so the zoom onto the window lands on the Hubble box (area label) with no DES
    label; the DES label returns with the DES-inset copy faded in on top at click 14."""
    inset, pw = COVER["ls"]
    s, pad = SIDE["hst"], 0.45
    name, area, psf = NAMES["hst"]
    cx, cy = off(WIN_RA, WIN_DEC)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(cx - INSET / 2, cx + INSET / 2)
    ax.set_ylim(cy - INSET / 2, cy + INSET / 2)
    ax.axis("off")
    fig.patch.set_alpha(0)
    # the cover: from the page's left/top edges to just past the label, stopping above the Hubble box
    lab = ax.text(*WIN_TL, f"{NAMES['ls'][0]} · {NAMES['ls'][2]}", fontsize=WIN_LABEL / INSET * 720, family=SANS,
                  weight="bold", va="top", ha="left")
    fig.canvas.draw()
    bb = lab.get_window_extent().transformed(ax.transData.inverted())
    lab.remove()
    x1, y0 = bb.x1 + 0.8, max(bb.y0 - 0.5, s / 2 + 0.12)     # never over the Hubble box (centred on the spiral)
    assert bb.y0 > s / 2, (bb.y0, s / 2)                      # the label's text box (descent included) clears it
    x, y = off(WIN_RA, WIN_DEC)
    n = inset.shape[0]
    px = (np.arange(n) + 0.5) / n * pw - pw / 2              # pixel centres, offset from the window centre
    alpha = ((x + px)[None, :] <= x1) & ((y + px)[:, None] >= y0)
    rgba = np.dstack([inset, alpha.astype(float)])
    ax.imshow(rgba, origin="lower", extent=(x - pw / 2, x + pw / 2, y - pw / 2, y + pw / 2), interpolation="none", zorder=1)
    ax.imshow(hst_img, origin="lower", extent=(-s / 2, s / 2, -s / 2, s / 2), interpolation="none", zorder=1)
    lw = 0.08 / INSET * 720                                   # 0.08" wide, as on the Hubble page
    d = lw / 720 * INSET / 2
    ax.add_patch(plt.Rectangle((-s / 2 + d, -s / 2 + d), s - 2 * d, s - 2 * d, fill=False, edgecolor="white",
                               alpha=0.85, linewidth=lw, zorder=2))
    pts = SPACE_LABEL / INSET * 720
    ax.text(-s / 2 + pad, s / 2 - pad, label, color="white", fontsize=pts, family=SANS,
            weight="bold", va="top", ha="left", path_effects=[pe.withStroke(linewidth=pts / 6, foreground="black")],
            zorder=3)
    path = os.path.join(OUT, "AL_hst_over_ls.pdf")
    fig.savefig(path, transparent=True)
    plt.close(fig)
    print("wrote", path, "cover x <=", round(x1, 2), "y >=", round(y0, 2), flush=True)


def space(inst):
    name, area, psf = NAMES[inst]
    s = SIDE[inst]
    planes = [resample(a, w, GRA, GDEC, 0.03, s) for a, w in L.bands(inst)]
    med, white = stretch(planes, inst)
    img = colour(planes, med, white, L.INST[inst][3])
    pad = 0.45                                                # arcsec; both space boxes share one label size
    page(inst, s, [(img, GRA, GDEC, s, s)], [(f"{name} · {psf}\n{area}", -s / 2 + pad, s / 2 - pad, SPACE_LABEL)],
         outline=0.08 / s * 720)                             # 0.08" wide: visible at the zoom window
    # 11-25 review: two-line labels from the start (sharpness and area together), so the JWST box no longer
    # switches meaning from area to resolution; the "res" pages are kept identical for the old layout
    for kind, label in (("area", f"{name} · {psf}\n{area}"), ("res", f"{name} · {psf}\n{area}")):
        page(f"{inst}_{kind}", s, [(img, GRA, GDEC, s, s)], [(label, -s / 2 + pad, s / 2 - pad, SPACE_LABEL)],
             outline=0.08 / s * 720)
    if inst == "hst" and "ls" in COVER:
        hst_over_ls(img, f"{name} · {psf}\n{area}")


which = sys.argv[1:] or ["dss", "ps1", "ls", "hst", "jwst", "euclid"]
for inst in which:
    {"dss": ground, "ps1": ground, "ls": ground, "hst": space, "jwst": space}.get(inst, lambda _i: euclid())(inst)
geo = {"side": SIDE, "win": {"ra": WIN_RA, "dec": WIN_DEC, "w": WIN_W, "h": WIN_H},
       "spiral": [GRA, GDEC], "polished": {"ra": L.RA0, "dec": L.DEC0, "side": L.FIELD},
       "win_off": off(WIN_RA, WIN_DEC), "pol_off": off(L.RA0, L.DEC0)}
json.dump(geo, open(os.path.join(OUT, "AL_geometry.json"), "w"), indent=1)
