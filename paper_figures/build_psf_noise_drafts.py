"""Draft paper figures for the PSF and the Noise & Artifacts subsections.

Run with the project environment::

    PYTHONPATH=. python paper_figures/build_psf_noise_drafts.py [psf_gallery ...]

Inputs (gitignored, under data/euclid_psf/):
  * psf_fig_stack.npz     — per band (VIS, Y, J, H) a (45, 601, 601) stack:
                            HDU0 mean + 44 cluster ePSFs, central 30"
  * psf_fig_full_mean.npz — the full 51" (1023 px) mean ePSF per band
  * euclid_psf_clusters.json — cluster centres, star counts, FWHM by band

Outputs go to paper_figures/drafts/.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from scipy.ndimage import gaussian_filter, label, rotate, zoom
from scipy.signal import fftconvolve

from euclid_polish.config import Config
from euclid_polish.sky.observation.artifacts import (
    ArtifactConfig,
    inject_cosmic_rays,
    inject_dead_pixels,
    inject_hot_pixels,
    inject_streaks,
)
from euclid_polish.sky.observation.field_variations import (
    DistantStarWing,
    add_distant_star_wings,
    draw_noise_scale_map,
)
from euclid_polish.sky.observation.noise import (
    apply_archive_noise,
    dithered_unit_noise,
    pixel_scatter_factor,
)
from euclid_polish.sky.observation.saturation import (
    StarSaturationModel,
    apply_saturation_masking,
)

ROOT = Path(__file__).resolve().parents[1]
PSF_DIR = ROOT / "data" / "euclid_psf"
OUT = ROOT / "paper_figures" / "drafts"

BANDS = ["VIS", "Y", "J", "H"]
BAND_KEY = {"VIS": "VIS", "Y": "Y_E", "J": "J_E", "H": "H_E"}
LABEL = {"VIS": r"$I_E$ (VIS)", "Y": r"$Y_E$", "J": r"$J_E$", "H": r"$H_E$"}
SHORT = {"VIS": "VIS", "Y": r"$Y_E$", "J": r"$J_E$", "H": r"$H_E$"}
# Band identity hues, blue → red with wavelength.
HUE = {"VIS": "#3d8bff", "Y": "#19c3a6", "J": "#ffb000", "H": "#ff3d5a"}
NATIVE_PIX = {"VIS": 0.1, "Y": 0.3, "J": 0.3, "H": 0.3}
OVERSAMPLED_PIX = 0.05  # arcsec

INK = "#1b1d24"
MUTED = "#6b7080"
NIGHT = "#07080d"

mpl.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "savefig.dpi": 300,
    "figure.dpi": 150,
})

FULL_WIDTH = 7.1   # in, A&A two-column figure*
COL_WIDTH = 3.5    # in, one column


# ---------------------------------------------------------------- helpers

def glow_cmap(hue: str, name: str) -> LinearSegmentedColormap:
    """Black → saturated band hue → white: a vivid single-hue ramp whose
    lightness rises monotonically, so log stretches read as magnitude."""
    rgb = np.array(mpl.colors.to_rgb(hue))
    deep = rgb * 0.35
    stops = [
        (0.00, NIGHT),
        (0.30, tuple(deep)),
        (0.62, tuple(rgb)),
        (0.85, tuple(0.45 * rgb + 0.55)),
        (1.00, "#ffffff"),
    ]
    return LinearSegmentedColormap.from_list(name, stops)


CMAP = {b: glow_cmap(HUE[b], f"glow_{b}") for b in BANDS}
DIVERGE = LinearSegmentedColormap.from_list(
    "psf_div", ["#0b3c8c", "#2f7fff", "#bcd3ff", "#f7f7f7", "#ffc2a6", "#ff5a2b", "#8c1c06"])


def load_stack() -> dict[str, np.ndarray]:
    d = np.load(PSF_DIR / "psf_fig_stack.npz")
    return {b: d[b].astype(np.float64) for b in BANDS} | {
        b + "_nstars": d[b + "_nstars"] for b in BANDS}


def load_full_mean() -> dict[str, np.ndarray]:
    d = np.load(PSF_DIR / "psf_fig_full_mean.npz")
    return {b: d[b].astype(np.float64) for b in BANDS}


def load_clusters() -> list[dict]:
    return json.loads((PSF_DIR / "euclid_psf_clusters.json").read_text())["clusters"]


def median_fwhm(clusters: list[dict]) -> dict[str, float]:
    return {b: float(np.median([c["fwhm_by_band"][BAND_KEY[b]] for c in clusters]))
            for b in BANDS}


def crop(a: np.ndarray, half_arcsec: float, pix: float = OVERSAMPLED_PIX) -> np.ndarray:
    c = a.shape[-1] // 2
    r = int(round(half_arcsec / pix))
    return a[..., c - r:c + r + 1, c - r:c + r + 1]


def extent(a: np.ndarray, pix: float = OVERSAMPLED_PIX) -> list[float]:
    h = a.shape[-1] * pix / 2
    return [-h, h, -h, h]


def radial_profile(img: np.ndarray, pix: float = OVERSAMPLED_PIX,
                   nbins: int = 70, rmax: float | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthal mean in log-spaced annuli about the stamp centre."""
    n = img.shape[0]
    yy, xx = np.indices(img.shape)
    c = (n - 1) / 2
    r = np.hypot(yy - c, xx - c) * pix
    rmax = rmax or r.max() * 0.999
    edges = np.concatenate([[0], np.geomspace(pix * 1.2, rmax, nbins)])
    idx = np.digitize(r.ravel(), edges) - 1
    s = np.bincount(idx, weights=img.ravel(), minlength=nbins + 1)[:nbins]
    k = np.bincount(idx, minlength=nbins + 1)[:nbins]
    mid = np.sqrt(edges[:-1] * edges[1:])
    mid[0] = pix * 0.3
    ok = k > 0
    return mid[ok], s[ok] / k[ok]


def encircled_energy(img: np.ndarray, radii: np.ndarray,
                     pix: float = OVERSAMPLED_PIX, core_arcsec: float = 1.5,
                     up: int = 8) -> np.ndarray:
    """Fraction of the stamp flux within each radius; inside ``core_arcsec``
    the stamp is spline-upsampled so the curve has no pixel staircase."""
    out = _ee_sorted(img, radii, pix)
    c = img.shape[0] // 2
    h = int(core_arcsec / pix) + 3
    core = zoom(img[c - h:c + h + 1, c - h:c + h + 1], up, order=3) / up ** 2
    inner = radii < core_arcsec
    # anchor the upsampled core to the exact enclosed flux at core_arcsec
    ref = _ee_sorted(img, np.array([core_arcsec]), pix)[0]
    fine = _ee_sorted(core, np.append(radii[inner], core_arcsec), pix / up, total=1.0)
    out[inner] = fine[:-1] / fine[-1] * ref
    return out


def _ee_sorted(img: np.ndarray, radii: np.ndarray, pix: float,
               total: float | None = None) -> np.ndarray:
    n = img.shape[0]
    yy, xx = np.indices(img.shape)
    c = (n - 1) / 2
    r = np.hypot(yy - c, xx - c) * pix
    order = np.argsort(r.ravel())
    cum = np.cumsum(img.ravel()[order])
    return np.interp(radii, r.ravel()[order], cum) / (total or img.sum())


def taper_window(n: int, inner: float = 0.82, outer: float = 0.98) -> np.ndarray:
    """The radial cosine roll-off of PSF.background_cleaned."""
    half = n / 2
    yy, xx = np.indices((n, n), dtype=np.float64)
    rr = np.hypot(yy - (n - 1) / 2, xx - (n - 1) / 2)
    t = np.clip((rr - inner * half) / ((outer - inner) * half), 0, 1)
    return 0.5 * (1 + np.cos(np.pi * t))


def scale_bar(ax, length: float, label: str, *, color: str = "white",
              loc: tuple[float, float] = (0.07, 0.08), lw: float = 1.6) -> None:
    """Horizontal bar of ``length`` data units starting at axes fraction ``loc``."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xs = x0 + loc[0] * (x1 - x0)
    ys = y0 + loc[1] * (y1 - y0)
    ax.plot([xs, xs + length], [ys, ys], color=color, lw=lw,
            solid_capstyle="butt")
    ax.text(xs + length / 2, ys + 0.035 * (y1 - y0), label, color=color,
            ha="center", va="bottom", fontsize=7)


def band_colorbar(fig, rect: list[float], norm, ticks, ticklabels, label: str,
                  bands: list[str] = BANDS) -> None:
    """One colourbar drawn as stacked stripes, one per band hue."""
    ax = fig.add_axes(rect)
    if isinstance(norm, LogNorm):
        vals = np.geomspace(norm.vmin, norm.vmax, 256)
    else:
        vals = np.linspace(norm.vmin, norm.vmax, 256)
    for i, b in enumerate(bands[::-1]):
        ax.imshow(norm(vals)[None, :], cmap=CMAP[b], vmin=0, vmax=1, aspect="auto",
                  extent=[0, 1, i, i + 1], interpolation="bilinear")
    ax.set_ylim(0, len(bands))
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks(norm(np.asarray(ticks, dtype=float)))
    ax.set_xticklabels(ticklabels)
    ax.tick_params(axis="x", labelsize=6.5, length=2, direction="out")
    ax.set_xlabel(label, fontsize=7, labelpad=1.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)


def representative_cluster() -> int:
    """A well-populated region whose four-band FWHMs sit near the survey
    medians (index into the stack; 0 is the mean)."""
    return 9


def bare(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03,
                facecolor=fig.get_facecolor())
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))
    return path


# ---------------------------------------------------------------- figures

def fig_psf_gallery() -> None:
    """One region's four-band ePSFs: 25" log stamps over 2" linear cores with
    the native detector pixel grid, showing how NISP undersamples its PSF."""
    stack = load_stack()
    fwhm = median_fwhm(load_clusters())
    k = representative_cluster()
    fig = plt.figure(figsize=(FULL_WIDTH, 3.95))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.06, wspace=0.04,
                          left=0.01, right=0.99, top=0.92, bottom=0.10)
    lognorm = LogNorm(3e-6, 1)
    for j, b in enumerate(BANDS):
        mean = stack[b][k]
        img = crop(mean, 12.5) / mean.max()
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(np.clip(img, 1e-7, None), origin="lower", cmap=CMAP[b],
                  norm=lognorm, extent=extent(img), interpolation="lanczos")
        bare(ax)
        ax.set_title(LABEL[b], color=INK, pad=3)
        ax.text(0.96, 0.95, f"FWHM {fwhm[b]:.2f}\u2033", transform=ax.transAxes,
                ha="right", va="top", color="white", fontsize=7.5)
        if j == 0:
            scale_bar(ax, 5, "5\u2033")

        core = crop(mean, 1.0) / mean.max()
        ax2 = fig.add_subplot(gs[1, j])
        ax2.imshow(core, origin="lower", cmap=CMAP[b], vmin=0, vmax=1,
                   extent=extent(core), interpolation="nearest")
        # native detector pixel grid (centred on the PSF peak)
        p = NATIVE_PIX[b]
        lim = extent(core)[1]
        grid = np.arange(-lim - p / 2, lim + p, p)
        for v in grid[(grid > -lim) & (grid < lim)]:
            ax2.axvline(v, color="white", lw=0.35, alpha=0.45)
            ax2.axhline(v, color="white", lw=0.35, alpha=0.45)
        ax2.add_patch(Circle((0, 0), fwhm[b] / 2, fill=False, ec="white",
                             lw=0.9, ls=(0, (3, 2))))
        ax2.set_xlim(-lim, lim)
        ax2.set_ylim(-lim, lim)
        bare(ax2)
        ax2.text(0.04, 0.95, f"native {p:.1f}\u2033 pixels", transform=ax2.transAxes,
                 ha="left", va="top", color="white", fontsize=6.8)
        if j == 0:
            scale_bar(ax2, 0.5, "0.5\u2033")
        for a in (ax, ax2):
            for s in a.spines.values():
                s.set_visible(True)
                s.set_color(HUE[b])
                s.set_linewidth(1.2)

    fig.text(0.0, 0.73, "log stretch, 25\u2033", rotation=90, va="center",
             ha="right", color=MUTED, fontsize=7.5)
    fig.text(0.0, 0.32, "linear, 2\u2033 core", rotation=90, va="center",
             ha="right", color=MUTED, fontsize=7.5)

    band_colorbar(fig, [0.17, 0.045, 0.28, 0.03], lognorm,
                  [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
                  ["$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "1"],
                  r"$I/I_{\rm peak}$ (top row)")
    band_colorbar(fig, [0.56, 0.045, 0.28, 0.03], mpl.colors.Normalize(0, 1),
                  [0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"],
                  r"$I/I_{\rm peak}$ (bottom row; dashed circle: median FWHM)")
    save(fig, "psf_gallery")


def field_name(c: dict) -> str:
    if c["dec"] > 40:
        return "EDF-N"
    if c["dec"] > -35:
        return "EDF-F"
    return "EDF-S"


def frame(ax, color: str, lw: float = 1.0) -> None:
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(color)
        s.set_linewidth(lw)


STAR_DIR = ROOT / "data" / "euclid_stars" / "cutouts"
STAR_BAND_DIR = {"VIS": "VIS", "Y": "Y_E", "J": "J_E", "H": "H_E"}


def fig_psf_measurement() -> None:
    """Individual Q1 star cutouts in every band → the stacked ePSF."""
    stars = ["star_0003", "star_0004", "star_0008", "star_0015", "star_0932"]
    stack = load_stack()
    k = representative_cluster()
    n_stars = int(stack["VIS_nstars"][k])
    half = 4.0   # arcsec
    norm = LogNorm(2e-4, 1)
    fig = plt.figure(figsize=(FULL_WIDTH, 4.62))
    ncol = len(stars) + 2
    gs = fig.add_gridspec(4, ncol, width_ratios=[1] * len(stars) + [0.62, 1.0],
                          wspace=0.05, hspace=0.05,
                          left=0.045, right=0.995, top=0.95, bottom=0.01)
    for i, b in enumerate(BANDS):
        for j, name in enumerate(stars):
            a = fits.getdata(STAR_DIR / STAR_BAND_DIR[b] / f"{name}_255.fits").astype(float)
            a = a - np.nanmedian(a)
            cy, cx = np.unravel_index(np.nanargmax(a[100:154, 100:154]), (54, 54))
            cy, cx = cy + 100, cx + 100
            r = int(half / 0.1)
            t = a[cy - r:cy + r + 1, cx - r:cx + r + 1] / a[cy, cx]
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(np.clip(t, 1e-6, None), origin="lower", cmap=CMAP[b], norm=norm,
                      interpolation="nearest")
            bare(ax)
            frame(ax, "#c9ccd6", 0.5)
            if i == 0:
                ax.set_title(f"star {j + 1}", fontsize=8, color=MUTED, pad=3)
            if j == 0:
                ax.set_ylabel(LABEL[b], fontsize=9, color=INK, labelpad=4)
                ax.yaxis.set_visible(True)
                ax.set_yticks([])
        axa = fig.add_subplot(gs[i, len(stars)])
        axa.axis("off")
        axa.add_patch(FancyArrowPatch((0.08, 0.5), (0.92, 0.5), transform=axa.transAxes,
                                      arrowstyle="-|>,head_length=6,head_width=3.5",
                                      lw=1.6, color=HUE[b], mutation_scale=1))
        if i == 0:
            axa.text(0.5, 0.66, "align\n" + r"oversample $\times 2$" + "\nstack",
                     transform=axa.transAxes, ha="center", va="bottom", fontsize=6.8,
                     color=INK, linespacing=1.15)
        e = crop(stack[b][k], half)
        ax = fig.add_subplot(gs[i, -1])
        ax.imshow(np.clip(e / e.max(), 1e-7, None), origin="lower", cmap=CMAP[b],
                  norm=norm, interpolation="lanczos", extent=extent(e))
        bare(ax)
        frame(ax, HUE[b], 1.5)
        if i == 0:
            ax.set_title(f"ePSF ({n_stars} stars)", fontsize=8, color=INK, pad=3)
        if i == 3:
            scale_bar(ax, 2, "2″")
    save(fig, "psf_measurement")


def fig_psf_rgb() -> None:
    """Chromatic PSF: H, J+Y and VIS of one region as R, G, B with a
    colour-preserving (Lupton et al. 2004) asinh stretch on flux-normalised
    kernels, so the sharp VIS core shows blue inside the broader NISP halo.
    A 0.05" Gaussian smooth hides the ePSF's 2x2 oversampling plaid
    (display only)."""
    stack = load_stack()
    k = representative_cluster()

    def chan(b, half):
        return gaussian_filter(crop(stack[b][k], half), 1.0) / stack[b][k].sum()

    fig, axes = plt.subplots(1, 2, figsize=(COL_WIDTH, 1.78),
                             gridspec_kw={"wspace": 0.03, "left": 0.0, "right": 1.0,
                                          "top": 1.0, "bottom": 0.0})
    for ax, half, (stretch, q) in zip(axes, [12.5, 1.0], [(2e-6, 20), (3e-3, 2)], strict=True):
        rgb = make_lupton_rgb(chan("H", half), 0.5 * (chan("Y", half) + chan("J", half)),
                              chan("VIS", half), stretch=stretch, Q=q, minimum=0)
        ax.imshow(rgb, origin="lower", extent=[-half, half, -half, half],
                  interpolation="lanczos")
        bare(ax)
    scale_bar(axes[0], 5, "5\u2033")
    scale_bar(axes[1], 0.5, "0.5\u2033")
    axes[0].add_patch(Rectangle((-1, -1), 2, 2, fill=False, ec="white", lw=0.6, alpha=0.8))
    for i, (txt, col) in enumerate([(r"R  $H_E$", "#ff5a6e"), (r"G  $Y_E{+}J_E$", "#3df07a"),
                                    ("B  VIS", "#6aa8ff")]):
        axes[0].text(0.035, 0.965 - 0.085 * i, txt, color=col, transform=axes[0].transAxes,
                     ha="left", va="top", fontsize=7)
    axes[1].text(0.965, 0.965, "core", color="white", transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=7)
    save(fig, "psf_rgb")


def fig_psf_variation() -> None:
    """Spatial variation: the 44 star regions on the sky (coloured by VIS
    FWHM), FWHM spread per band, and every region's VIS ePSF."""
    clusters = load_clusters()
    stack = load_stack()
    fields = ["EDF-N", "EDF-F", "EDF-S"]
    vis_fwhm = np.array([c["fwhm_by_band"]["VIS"] for c in clusters])
    fnorm = mpl.colors.Normalize(np.percentile(vis_fwhm, 2), np.percentile(vis_fwhm, 98))
    fcmap = LinearSegmentedColormap.from_list(
        "fwhm", ["#0a1f5c", "#1f5cff", "#19c3a6", "#ffd23d", "#ff3d5a"])

    fig = plt.figure(figsize=(FULL_WIDTH, 5.35))
    top = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.15], wspace=0.3,
                           left=0.06, right=0.99, top=0.965, bottom=0.565)
    for fi, fld in enumerate(fields):
        cs = [c for c in clusters if field_name(c) == fld]
        ra0 = np.median([c["ra"] for c in cs])
        de0 = np.median([c["dec"] for c in cs])
        x = np.array([(c["ra"] - ra0) * np.cos(np.radians(de0)) for c in cs])
        y = np.array([c["dec"] - de0 for c in cs])
        x, y = x - (x.max() + x.min()) / 2, y - (y.max() + y.min()) / 2
        f = np.array([c["fwhm_by_band"]["VIS"] for c in cs])
        n = np.array([c["n_stars"] for c in cs])
        ax = fig.add_subplot(top[0, fi])
        span = max(np.abs(x).max(), np.abs(y).max()) + 1.0
        npx = 600
        g = np.linspace(-span, span, npx)
        gx, gy = np.meshgrid(g, g)
        d2 = (gx[..., None] - x) ** 2 + (gy[..., None] - y) ** 2
        near = d2.argmin(-1)
        dmin = np.sqrt(d2.min(-1))
        img = fcmap(fnorm(f[near]))
        reach = np.clip(1.25 - dmin / 0.8, 0, 1)
        img[..., 3] = reach * 0.6
        edge = np.zeros_like(near, dtype=bool)
        edge[:, 1:] |= near[:, 1:] != near[:, :-1]
        edge[1:, :] |= near[1:, :] != near[:-1, :]
        img[edge, :3] = 1.0
        img[edge, 3] = np.maximum(img[edge, 3], reach[edge])
        ax.imshow(img, origin="lower", extent=[-span, span, -span, span],
                  interpolation="bilinear")
        ax.scatter(x, y, s=n / 22, c=f, cmap=fcmap, norm=fnorm, ec=INK, lw=0.5, zorder=3)
        ax.set_xlim(span, -span)   # RA increases to the left
        ax.set_ylim(-span, span)
        ax.set_aspect("equal")
        ax.set_title(f"{fld}  ({len(cs)} regions)", fontsize=8.5, pad=3)
        ax.set_xlabel(r"$\Delta\alpha\cos\delta$ [deg]", fontsize=7)
        if fi == 0:
            ax.set_ylabel(r"$\Delta\delta$ [deg]", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.set_facecolor("#f4f5f8")
    cax = fig.add_axes([0.10, 0.515, 0.42, 0.013])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=fnorm, cmap=fcmap), cax=cax,
                      orientation="horizontal")
    cb.set_label("VIS FWHM [arcsec]   (marker area ∝ stars in the region)",
                 fontsize=7, labelpad=1)
    cb.ax.tick_params(labelsize=6.5, length=2)

    ax = fig.add_subplot(top[0, 3])
    rng = np.random.default_rng(3)
    for j, b in enumerate(BANDS):
        v = np.array([c["fwhm_by_band"][BAND_KEY[b]] for c in clusters])
        xs = j + rng.uniform(-0.18, 0.18, len(v))
        bad = (v < 0.2) & (b != "VIS")      # below the NISP diffraction limit
        ax.scatter(xs[~bad], v[~bad], s=9, color=HUE[b], alpha=0.85, ec="white", lw=0.3,
                   zorder=3)
        ax.scatter(xs[bad], v[bad], s=9, facecolor="none", ec=HUE[b], lw=0.8, zorder=3)
        med = np.median(v)
        ax.plot([j - 0.3, j + 0.3], [med, med], color=INK, lw=1.4, zorder=4)
        ax.text(j + 0.32, med, f"{med:.2f}", va="center", ha="left", fontsize=6.5,
                color=INK, bbox={"fc": "white", "ec": "none", "pad": 0.4, "alpha": 0.8}, zorder=5)
    ax.set_xticks(range(4))
    ax.set_xticklabels([SHORT[b] for b in BANDS])
    ax.set_ylabel("FWHM [arcsec]")
    ax.set_xlim(-0.5, 3.8)
    ax.grid(axis="y", color="#e3e5ec", lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title("FWHM per region", fontsize=8.5, pad=3)
    ax.text(0.03, 0.97, "open: FWHM fit below\nthe NISP diffraction limit",
            transform=ax.transAxes, ha="left", va="top", fontsize=6, color=MUTED)

    # contact sheet: all 44 VIS ePSFs grouped by field, one mosaic image
    order = sorted(range(len(clusters)),
                   key=lambda i: (fields.index(field_name(clusters[i])), clusters[i]["ra"]))
    nrow, ncolumns = 4, 11
    half = 6.0
    norm = LogNorm(3e-6, 1)
    tiles = [crop(stack["VIS"][clusters[ci]["index"]], half) for ci in order]
    t = tiles[0].shape[0]
    gap = 14
    H = nrow * t + (nrow - 1) * gap
    W = ncolumns * t + (ncolumns - 1) * gap
    mosaic = np.ones((H, W, 4))
    boxes = []
    for i, (ci, e) in enumerate(zip(order, tiles, strict=True)):
        r, c = divmod(i, ncolumns)
        y0 = (nrow - 1 - r) * (t + gap)
        x0 = c * (t + gap)
        mosaic[y0:y0 + t, x0:x0 + t] = CMAP["VIS"](norm(np.clip(e / e.max(), 1e-7, None)))
        boxes.append((x0, y0, clusters[ci]))
    axm = fig.add_axes([0.005, 0.005, 0.99, 0.43])
    axm.imshow(mosaic, origin="lower", interpolation="lanczos", extent=[0, W, 0, H])
    for x0, y0, c in boxes:
        axm.add_patch(Rectangle((x0, y0), t, t, fill=False, lw=1.3,
                                ec=fcmap(fnorm(c["fwhm_by_band"]["VIS"]))))
        axm.text(x0 + 0.05 * t, y0 + 0.95 * t, field_name(c).split("-")[1] + f"{c['index']}",
                 color="white", fontsize=5.5, ha="left", va="top", alpha=0.85)
    axm.set_xlim(-2, W + 2)
    axm.set_ylim(-2, H + 2)
    axm.set_aspect("equal")
    axm.axis("off")
    fig.text(0.5, 0.445, "VIS ePSF of every region (12″ stamps, log stretch; "
             "frame colour = VIS FWHM)", ha="center", va="bottom", fontsize=8, color=INK)
    save(fig, "psf_variation")


def fig_psf_profiles() -> None:
    """Azimuthal profiles with the region-to-region spread, encircled energy,
    and the radial cosine taper that removes the square stamp edge."""
    stack = load_stack()
    full = load_full_mean()
    fig = plt.figure(figsize=(FULL_WIDTH, 2.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.15], wspace=0.32,
                          left=0.065, right=0.995, top=0.9, bottom=0.17)
    ax = fig.add_subplot(gs[0, 0])
    axe = fig.add_subplot(gs[0, 1])
    n_full = full["VIS"].shape[0]
    r_in = 0.82 * n_full / 2 * OVERSAMPLED_PIX
    r_out = 0.98 * n_full / 2 * OVERSAMPLED_PIX
    radii = np.geomspace(0.02, 25, 160)
    for b in BANDS:
        profs = []
        for i in range(1, 45):
            r, pr = radial_profile(stack[b][i], rmax=14.5)
            profs.append(pr / stack[b][i].max())
        profs = np.array(profs)
        lo, med, hi = np.nanpercentile(profs, [16, 50, 84], axis=0)
        ax.fill_between(r, lo, hi, color=HUE[b], alpha=0.22, lw=0)
        fm = full[b]
        tap = fm * taper_window(n_full)
        rf, pf = radial_profile(fm, rmax=25.5)
        rt, pt = radial_profile(tap, rmax=25.5)
        sel = rf > r_in * 0.95
        ax.plot(rf[sel], pf[sel] / fm.max(), color=HUE[b], lw=0.8, ls=(0, (2, 1.5)))
        ax.plot(rt, pt / fm.max(), color=HUE[b], lw=1.3, label=SHORT[b])
        ee = encircled_energy(tap, radii)
        axe.plot(radii, ee, color=HUE[b], lw=1.4, label=SHORT[b])
    ax.axvspan(r_in, r_out, color="#9aa0b4", alpha=0.18, lw=0)
    ax.text(np.sqrt(r_in * r_out), 3e-1, "taper", rotation=90, ha="center", va="top",
            fontsize=6.5, color=MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.03, 26)
    ax.set_ylim(1e-8, 1.5)
    ax.set_xlabel("radius [arcsec]")
    ax.set_ylabel(r"azimuthal mean $I/I_{\rm peak}$")
    ax.legend(frameon=False, loc="lower left", handlelength=1.4, ncol=2,
              columnspacing=0.8)
    ax.set_title("(a) radial profile", fontsize=8.5, pad=3, loc="left")
    ax.grid(color="#eceef3", lw=0.5, which="major")

    for lvl in (0.5, 0.8, 0.95):
        axe.axhline(lvl, color="#c5c9d6", lw=0.6, ls=":")
        axe.text(0.032, lvl + 0.012, f"{int(lvl * 100)}%", fontsize=6.3, color=MUTED)
    axe.set_xscale("log")
    axe.set_xlim(0.03, 26)
    axe.set_ylim(0, 1.02)
    axe.set_xlabel("radius [arcsec]")
    axe.set_ylabel("encircled energy")
    axe.set_title("(b) encircled energy", fontsize=8.5, pad=3, loc="left")

    # (c) the taper: raw stamp edge (left half) vs tapered (right half), VIS
    axc = fig.add_subplot(gs[0, 2])
    fm = full["VIS"] / full["VIS"].max()
    tap = fm * taper_window(n_full)
    pad = 70
    raw_p = np.pad(fm, pad)
    tap_p = np.pad(tap, pad)
    m = raw_p.shape[0] // 2
    combo = np.where(np.arange(raw_p.shape[1])[None, :] < m, raw_p, tap_p)
    norm = LogNorm(2e-7, 3e-4)
    ext = extent(combo)
    axc.imshow(np.clip(np.abs(combo), 1e-9, None), origin="lower", cmap=CMAP["VIS"],
               norm=norm, extent=ext, interpolation="bilinear")
    axc.axvline(0, color="white", lw=0.8, ls=(0, (3, 2)))
    for rr in (r_in, r_out):
        axc.add_patch(Circle((0, 0), rr, fill=False, ec="#ffd23d", lw=0.6, alpha=0.8))
    axc.text(0.03, 0.97, "raw stamp", transform=axc.transAxes, color="white",
             fontsize=7, ha="left", va="top")
    axc.text(0.97, 0.97, "tapered", transform=axc.transAxes, color="white",
             fontsize=7, ha="right", va="top")
    bare(axc)
    frame(axc, HUE["VIS"], 1.0)
    scale_bar(axc, 10, "10″", loc=(0.06, 0.06))
    axc.set_title("(c) edge taper, VIS (deep stretch)", fontsize=8.5, pad=3, loc="left")
    save(fig, "psf_profiles")


# ------------------------------------------------------------ noise model

LR_NAMES = {"VIS": "VIS", "Y": "Y_E", "J": "J_E", "H": "H_E"}
TNG_DIR = ROOT / "data" / "tng_skirt"
NOISE_TABLE = ROOT / "euclid_polish" / "sky" / "observation" / "mer_noise_levels.json"
BRIGHT_GALAXY = ROOT / "poster" / "fig" / "poster" / "euclid_bright_star_cutout_4band.fits"
# Artifact rates multiplied for visibility in the demo scene (labelled in
# the figure); the training forward model uses the Config rates.
ARTIFACT_BOOST = {"cr": 6.0, "hot": 60.0, "dead": 1.0, "streak": 25.0}


@dataclass
class DemoScene:
    hr: dict[str, np.ndarray]           # e- per 0.05" pixel, truth
    optical: dict[str, np.ndarray]      # PSF-convolved, 0.1" grid, pre-noise
    noisy: dict[str, np.ndarray]        # + correlated sky/shot noise
    field: dict[str, np.ndarray]        # + depth seam + off-field wing
    final: dict[str, np.ndarray]        # + artifacts + saturation blackout
    sigma: dict[str, float]             # per-pixel sky sigma, e-
    scale_map: np.ndarray
    artifact_masks: dict[str, np.ndarray]
    bright_star_lr: tuple[int, int]


def _tng_stamp(gid: str, orient: str, band: str, size_hr: int, angle: float) -> np.ndarray:
    a = fits.getdata(TNG_DIR / gid / f"TNG{gid}_{orient}_Euclid_{band}.fits").astype(float)
    a = np.nan_to_num(a)
    c = a.shape[0] // 2
    a = a[c - 260:c + 260, c - 260:c + 260]
    a = zoom(a, size_hr / a.shape[0], order=1)
    a = rotate(a, angle, reshape=False, order=1)
    return np.clip(a, 0, None)


def build_demo_scene(seed: int = 7) -> DemoScene:
    """A 25.6" illustrative scene (TNG50 galaxies + stars) pushed through the
    project's MER-level noise, seam, off-field-wing, artifact and saturation
    functions, band by band."""
    stack = load_stack()
    k = representative_cluster()
    n_hr, n_lr = 512, 256
    bands = {b: Config.get_band(LR_NAMES[b]) for b in BANDS}
    sigma = {b: bands[b].mer_rms_e * pixel_scatter_factor(bands[b]) for b in BANDS}
    tng_band = {"VIS": "VIS", "Y": "Y", "J": "J", "H": "H"}

    galaxies = [  # gid, orientation, centre (HR px), size (HR px), angle, S/N scale
        ("167396", "O4", (300, 285), 230, 20, 1.0),
        ("275546", "O1", (120, 380), 150, -35, 0.6),
        ("504560", "O2", (410, 110), 120, 60, 0.9),
        ("167396", "O1", (150, 120), 110, 100, 0.5),
        ("275546", "O3", (430, 420), 80, 10, 0.5),
    ]
    stars = [(70, 230, 3e3), (250, 60, 1.2e3), (470, 260, 600.0), (330, 470, 2.5e3),
             (200, 470, 900.0), (40, 40, 700.0), (360, 190, 400.0)]
    bright = (205, 255)                                     # saturating star
    star_colour = {"VIS": 1.0, "Y": 1.1, "J": 1.0, "H": 0.9}

    hr, optical = {}, {}
    for b in BANDS:
        img = np.zeros((n_hr, n_hr))
        for gid, orient, (cx, cy), size, ang, sn in galaxies:
            st = _tng_stamp(gid, orient, tng_band[b], size, ang)
            # MJy/sr -> e-: one factor per band referenced to the sky level, so
            # TNG colours set the band ratios and S/N is comparable per band
            st = st * sn * 9.0 * sigma[b]
            h = size // 2
            img[cy - h:cy - h + size, cx - h:cx - h + size] += st
        for x, y, f in stars:
            img[y, x] += f * sigma[b] * star_colour[b]
        psf = stack[b][k] / stack[b][k].sum()
        peak_frac = fftconvolve(np.pad(np.ones((1, 1)), 1), psf[290:311, 290:311],
                                mode="full").max()
        well = Config.STAR_SATURATION_WELL_E[LR_NAMES[b]]
        img[bright[1], bright[0]] += 2.5 * well / max(peak_frac * 4, 1e-6)
        hr[b] = img
        conv = fftconvolve(img, psf, mode="same")
        optical[b] = conv.reshape(n_lr, 2, n_lr, 2).sum((1, 3))

    seam_rng = np.random.default_rng(seed + 100)
    scale_map = draw_noise_scale_map(
        (n_lr, n_lr), seam_rng, global_scale_min=1.0, global_scale_max=1.0,
        region_probability=1.0, region_fraction_min=0.32, region_fraction_max=0.42,
        region_step_min=1.35, region_step_max=1.42)
    wing = DistantStarWing(angle_rad=np.radians(62), offset_lr_pix=-70.0, source_side=1,
                           amplitude_sigma=5.0, width_lr_pix=1.6,
                           fade_length_lr_pix=180.0)

    noisy, field, final = {}, {}, {}
    masks = {"cr": np.zeros((n_lr, n_lr), bool), "hot": np.zeros((n_lr, n_lr), bool),
             "dead": np.zeros((n_lr, n_lr), bool), "streak": np.zeros((n_lr, n_lr), bool)}
    for i, b in enumerate(BANDS):
        band = bands[b]
        sig = optical[b]
        noisy[b] = apply_archive_noise(sig, band, np.random.default_rng(seed + i),
                                       sky_rms_e=band.mer_rms_e)
        f = apply_archive_noise(sig, band, np.random.default_rng(seed + i),
                                sky_rms_e=band.mer_rms_e, noise_scale_map=scale_map)
        field[b] = add_distant_star_wings(f, (wing,), local_sigma_e=sigma[b])
        area = 1.0 if b == "VIS" else 9.0
        base = ArtifactConfig()
        cfg = replace(
            base,
            cr_rate_per_s_per_cm2=base.cr_rate_per_s_per_cm2 * ARTIFACT_BOOST["cr"],
            cr_charge_median_e=base.cr_charge_median_e / area,
            hot_pixel_fraction=base.hot_pixel_fraction * ARTIFACT_BOOST["hot"],
            hot_pixel_charge_mean_e=base.hot_pixel_charge_mean_e / area,
            dead_pixel_fraction=base.dead_pixel_fraction * ARTIFACT_BOOST["dead"],
            streak_rate_per_kpix2=base.streak_rate_per_kpix2 * ARTIFACT_BOOST["streak"],
        )
        arng = np.random.default_rng(seed + 50 + i)
        # the inject_artifacts order, one stage at a time to record VIS masks
        out = field[b].copy()
        for name in ("cr", "hot", "streak", "dead"):
            if name == "cr":
                new = inject_cosmic_rays(out, band, arng, cfg)
            elif name == "hot":
                new = inject_hot_pixels(out, arng, cfg)
            elif name == "streak":
                new = inject_streaks(out, arng, cfg, sigma[b])
            else:
                new = inject_dead_pixels(out, arng, cfg, sigma[b])
            if b == "VIS":
                masks[name] |= np.abs(new - out) > 1e-3
            out = new
        final[b] = out
    lr = np.stack([final[b] for b in BANDS], axis=-1)
    trig = np.stack([optical[b] for b in BANDS], axis=-1)
    apply_saturation_masking(lr, StarSaturationModel(), np.random.default_rng(seed + 9),
                             band_names=Config.LR_INPUT_BAND_NAMES, trigger_4ch=trig,
                             mask_probability=1.0)
    for i, b in enumerate(BANDS):
        final[b] = lr[..., i]
    return DemoScene(hr, optical, noisy, field, final, sigma, scale_map, masks,
                     (bright[0] // 2, bright[1] // 2))


def lupton_sigma(imgs: dict[str, np.ndarray], sigma: dict[str, float], *, per_pix: float = 1.0,
                 stretch: float = 10.0, q: float = 7.0) -> np.ndarray:
    """H, Y+J, VIS composite in units of each band's sky sigma."""
    u = {b: imgs[b] / (sigma[b] * per_pix) for b in BANDS}
    return make_lupton_rgb(u["H"], 0.5 * (u["Y"] + u["J"]), u["VIS"],
                           stretch=stretch, Q=q, minimum=-0.6)


def asinh_sigma(img: np.ndarray, sig: float, soft: float = 2.0, top: float = 300.0) -> np.ndarray:
    v = np.arcsinh(img / sig / soft)
    return np.clip((v + np.arcsinh(1.5 / soft)) / (np.arcsinh(top / soft) + np.arcsinh(1.5 / soft)), 0, 1)


def panel_label(ax, text: str, *, color: str = "white", loc: str = "ul", size: float = 7.5) -> None:
    x, ha = (0.04, "left") if loc[1] == "l" else (0.96, "right")
    y, va = (0.96, "top") if loc[0] == "u" else (0.04, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, color=color, fontsize=size)


def fig_noise_artifacts() -> None:
    """The forward model from truth to a MER-like observation (top), and the
    measured ingredients that drive it (bottom)."""
    sc = build_demo_scene()
    fig = plt.figure(figsize=(FULL_WIDTH, 3.4))
    top = fig.add_gridspec(1, 5, wspace=0.12, left=0.005, right=0.995, top=0.945, bottom=0.565)
    titles = ["(a) truth, 0.05″", "(b) ⊛ ePSF, 0.1″ grid",
              "(c) + correlated noise", "(d) + seam + off-field wing",
              "(e) + artifacts, saturation"]
    imgs = [None, sc.optical, sc.noisy, sc.field, sc.final]
    axes = []
    for j in range(5):
        ax = fig.add_subplot(top[0, j])
        if j == 0:
            rgb = lupton_sigma({b: gaussian_filter(sc.hr[b], 0.6) * 4 for b in BANDS}, sc.sigma)
        else:
            rgb = lupton_sigma(imgs[j], sc.sigma)
        ax.imshow(rgb, origin="lower", interpolation="lanczos" if j == 0 else "nearest")
        bare(ax)
        ax.set_title(titles[j], fontsize=7.6, pad=2.5)
        axes.append(ax)
    for j in range(4):
        b0 = axes[j].get_position()
        b1 = axes[j + 1].get_position()
        xm = (b0.x1 + b1.x0) / 2
        ym = (b0.y0 + b0.y1) / 2
        fig.patches.append(FancyArrowPatch((xm - 0.008, ym), (xm + 0.009, ym),
                                           transform=fig.transFigure,
                                           arrowstyle="-|>,head_length=3.5,head_width=2.2",
                                           color=INK, lw=0.9, mutation_scale=1))
    scale_bar(axes[0], 200, "10″")   # HR px at 0.05"
    scale_bar(axes[1], 100, "10″")
    for t, (c, txt) in enumerate([("#ff5a6e", r"$H_E$"), ("#3df07a", r"$Y_E{+}J_E$"),
                                  ("#6aa8ff", "VIS")]):
        axes[0].text(0.04, 0.96 - 0.09 * t, txt, transform=axes[0].transAxes, ha="left",
                     va="top", color=c, fontsize=6.5)

    bot = fig.add_gridspec(1, 5, wspace=0.12, left=0.005, right=0.995, top=0.45, bottom=0.075)

    # (f) noise texture: white vs dithered stacks, unit large-scale variance
    axf = fig.add_subplot(bot[0, 0])
    rng = np.random.default_rng(1)
    n = 48
    strips = [rng.standard_normal((n, n // 3 * 1 + 0)),
              dithered_unit_noise((n, n // 3), Config.get_band("VIS"), rng),
              dithered_unit_noise((n, n // 3), Config.get_band("H_E"), rng)]
    strips = [x / x.std() for x in strips]
    sep = np.full((n, 1), np.nan)
    tex = np.hstack([strips[0], sep, strips[1], sep, strips[2]])
    tcm = DIVERGE.copy()
    tcm.set_bad("white")
    axf.imshow(tex, origin="lower", cmap=tcm, vmin=-2.6, vmax=2.6, interpolation="nearest")
    bare(axf)
    for t, txt in enumerate(["white", "VIS", r"$H_E$"]):
        axf.text((t + 0.5) / 3, 1.015, txt, transform=axf.transAxes, ha="center",
                 va="bottom", fontsize=6.8)
    axf.set_title("(f) dithered-stack texture", fontsize=7.6, pad=10)
    axf.text(0.5, -0.03, "pixel-normalised; NISP shares\neach 0.3″ pixel over 3×3",
             transform=axf.transAxes, ha="center", va="top", fontsize=6, color=MUTED)

    # (g) local noise sigma: Q1 sky level, source shot noise and the seam
    axg = fig.add_subplot(bot[0, 1])
    resid = sc.field["VIS"] - sc.optical["VIS"]
    blk = 8
    loc = resid.reshape(32, blk, 32, blk).std(axis=(1, 3)) / sc.sigma["VIS"]
    seq = LinearSegmentedColormap.from_list("sigmap", [NIGHT, "#2b1a6b", "#7a2fd0",
                                                      "#ff4fa3", "#ffb347", "#fff3c4"])
    im = axg.imshow(loc, origin="lower", cmap=seq, vmin=0.7, vmax=2.4,
                    interpolation="bicubic", extent=[0, 256, 0, 256])
    bare(axg)
    axg.set_title("(g) local noise σ (VIS)", fontsize=7.6, pad=2.5)
    cax = axg.inset_axes([0.08, 0.07, 0.5, 0.045])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([1, 1.5, 2])
    cb.ax.tick_params(labelsize=5.5, colors="white", length=1.5, pad=1)
    cb.outline.set_edgecolor("white")
    cb.outline.set_linewidth(0.4)
    panel_label(axg, "seam ×%.2f" % (sc.scale_map.max() / sc.scale_map.min()),
                loc="ur", size=6.5)
    panel_label(axg, "σ / σ$_{\\rm sky}$", loc="ul", size=6.5)

    # (h) Q1 sky levels: field-to-field spread per band, as a ridgeline
    axh = fig.add_subplot(bot[0, 2])
    tab = json.loads(NOISE_TABLE.read_text())
    lv = np.array([r["levels_e"] for r in tab["rows"]], dtype=float)
    xs = np.linspace(0.55, 1.75, 400)
    for i, b in enumerate(BANDS):
        v = lv[:, i] / np.median(lv[:, i])
        bw = 0.9 * min(v.std(), np.subtract(*np.percentile(v, [75, 25])) / 1.34) * len(v) ** -0.2
        dens = np.exp(-0.5 * ((xs[:, None] - v[None, :]) / bw) ** 2).sum(1)
        dens = dens / dens.max() * 1.25
        off = (3 - i) * 0.95
        axh.fill_between(xs, off, off + dens, color=HUE[b], alpha=0.55, lw=0, zorder=4 - i)
        axh.plot(xs, off + dens, color=HUE[b], lw=1.0, zorder=4 - i)
        axh.plot(xs, np.full_like(xs, off), color=HUE[b], lw=0.5, alpha=0.6, zorder=4 - i)
        sd = np.std(np.log(v))
        axh.text(0.57, off + 0.18, SHORT[b], fontsize=6.6, color=INK, ha="left",
                 va="bottom", zorder=10)
        axh.text(1.73, off + 0.18, f"{100 * sd:.0f}%", fontsize=6.2, color=MUTED,
                 ha="right", va="bottom", zorder=10)
    axh.axvline(1.0, color=INK, lw=0.6, ls=(0, (2, 2)), zorder=0)
    axh.set_xlim(0.55, 1.75)
    axh.set_ylim(-0.05, 4.2)
    axh.set_yticks([])
    axh.set_xticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
    axh.set_xlabel("sky \u03c3 / band median", fontsize=7, labelpad=1)
    axh.tick_params(labelsize=6.3, pad=1.5)
    for sp in ("top", "right", "left"):
        axh.spines[sp].set_visible(False)
    axh.set_title(f"(h) Q1 sky \u03c3, {len(lv)} tiles", fontsize=7.6, pad=2.5)
    axh.text(1.73, 4.1, "scatter", fontsize=5.8, color=MUTED, ha="right", va="top")

    # (i) artifacts close-up (VIS)
    axi = fig.add_subplot(bot[0, 3])
    m = sc.artifact_masks
    x0, y0, w = _artifact_window(m, sc.optical["VIS"] / sc.sigma["VIS"])
    cut = sc.final["VIS"][y0:y0 + w, x0:x0 + w]
    axi.imshow(asinh_sigma(cut, sc.sigma["VIS"], soft=1.5, top=60), origin="lower",
               cmap=CMAP["VIS"], vmin=0, vmax=1, interpolation="nearest",
               extent=[x0, x0 + w, y0, y0 + w])
    marks = {"cr": ("#ffd23d", "cosmic ray"), "hot": ("#ff3d5a", "hot pixel"),
             "dead": ("#19c3a6", "dead pixel"), "streak": ("#ffffff", "masked trail")}
    for key, (col, _) in marks.items():
        if key == "streak":
            continue
        ys, xs = np.nonzero(m[key][y0:y0 + w, x0:x0 + w])
        for yy, xx in _cluster_points(xs, ys):
            axi.add_patch(Circle((x0 + xx + 0.5, y0 + yy + 0.5), 2.6, fill=False, ec=col,
                                 lw=0.8))
    # trails are sub-sigma by construction: mark each one's centre line
    lab, nlab = label(m["streak"])
    for t in range(1, nlab + 1):
        ys, xs = np.nonzero(lab == t)
        if len(xs) < 20:
            continue
        pts = np.c_[xs, ys].astype(float) + 0.5
        cen = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - cen, full_matrices=False)
        proj = (pts - cen) @ vt[0]
        a, b = cen + proj.min() * vt[0], cen + proj.max() * vt[0]
        axi.plot([a[0], b[0]], [a[1], b[1]], color="white", lw=0.7, ls=(0, (3, 2.5)),
                 alpha=0.85)
    axi.set_xlim(x0, x0 + w)
    axi.set_ylim(y0, y0 + w)
    bare(axi)
    axi.set_title(f"(i) artifacts, VIS {w // 10}″ zoom", fontsize=7.6, pad=2.5)
    legend_col = {"cr": "#e0a800", "hot": "#ff3d5a", "dead": "#13a68d", "streak": INK}
    for t, key in enumerate(["cr", "hot", "dead", "streak"]):
        lab = marks[key][1]
        axi.text(0.0 + 0.52 * (t % 2), -0.04 - 0.085 * (t // 2),
                 ("\u25cb " if key != "streak" else "- - ") + lab,
                 transform=axi.transAxes, color=legend_col[key], fontsize=6, ha="left",
                 va="top")
    axi.text(0.5, -0.23, "(rates boosted for visibility)", transform=axi.transAxes,
             color=MUTED, fontsize=5.6, ha="center", va="top")

    # (j) saturation: simulated blackout vs a real Q1 blanked nucleus
    sub = bot[0, 4].subgridspec(2, 2, wspace=0.04, hspace=0.04)
    bx, by = sc.bright_star_lr
    real = fits.getdata(BRIGHT_GALAXY).astype(float)
    hh = 14
    for col, b in enumerate(["VIS", "H"]):
        ax = fig.add_subplot(sub[0, col])
        cut = sc.final[b][by - hh:by + hh, bx - hh:bx + hh]
        ax.imshow(asinh_sigma(cut, sc.sigma[b], soft=3, top=3e4), origin="lower",
                  cmap=CMAP[b], vmin=0, vmax=1, interpolation="nearest")
        bare(ax)
        frame(ax, HUE[b], 0.8)
        if col == 0:
            panel_label(ax, "sim.", size=6)
        panel_label(ax, SHORT[b], loc="ur", size=6.3)
        ax2 = fig.add_subplot(sub[1, col])
        i = 0 if b == "VIS" else 3
        rc = real[i, 127 - 2 * hh:127 + 2 * hh, 125 - 2 * hh:125 + 2 * hh]
        sky = np.nanstd(real[i, :40, :40])
        ax2.imshow(asinh_sigma(rc, sky, soft=3, top=3e4), origin="lower", cmap=CMAP[b],
                   vmin=0, vmax=1, interpolation="nearest")
        bare(ax2)
        frame(ax2, HUE[b], 0.8)
        if col == 0:
            panel_label(ax2, "Q1", size=6)
        panel_label(ax2, SHORT[b], loc="ur", size=6.3)
    fig.text(0.9, 0.457, "(j) saturation blackout", ha="center", va="bottom", fontsize=7.6)
    fig.text(0.9, 0.06, "Q1: blanked galaxy nucleus", ha="center", va="top", fontsize=6,
             color=MUTED)
    save(fig, "noise_artifacts")


def _cluster_points(xs: np.ndarray, ys: np.ndarray, r: float = 4.0) -> list[tuple[float, float]]:
    """Merge neighbouring artifact pixels into one marker each."""
    pts: list[list[float]] = []
    for x, y in zip(xs, ys, strict=True):
        for p in pts:
            if abs(p[0] - x) <= r and abs(p[1] - y) <= r:
                break
        else:
            pts.append([float(x), float(y)])
    return [(p[1], p[0]) for p in pts]


def _artifact_window(m: dict[str, np.ndarray], snr: np.ndarray,
                     w: int = 64) -> tuple[int, int, int]:
    """The w x w window holding the most distinct artifact types/hits, on
    empty sky (no source above 30 sigma)."""
    best, best_xy = -1.0, (0, 0)
    n = m["cr"].shape[0]
    for y0 in range(0, n - w + 1, 8):
        for x0 in range(0, n - w + 1, 8):
            sl = (slice(y0, y0 + w), slice(x0, x0 + w))
            if snr[sl].max() > 30:
                continue
            kinds = sum(bool(m[k][sl].any()) for k in m)
            hits = sum(min(int(m[k][sl].sum()), 6) for k in ("cr", "hot", "dead"))
            score = kinds * 10 + hits
            if score > best:
                best, best_xy = score, (x0, y0)
    return best_xy[0], best_xy[1], w


FIGURES = {
    "psf_gallery": fig_psf_gallery,
    "psf_measurement": fig_psf_measurement,
    "psf_rgb": fig_psf_rgb,
    "psf_variation": fig_psf_variation,
    "psf_profiles": fig_psf_profiles,
    "noise_artifacts": fig_noise_artifacts,
}


def main(argv: list[str]) -> None:
    names = argv or list(FIGURES)
    for n in names:
        FIGURES[n]()


if __name__ == "__main__":
    main(sys.argv[1:])
