"""Paper figure: the Euclid-only stellar prior (counts + colours), article style.

(a) Q1 PHZ-weighted VIS PSF star counts, the straight law fitted in its
    straight region and extrapolated to VIS 12-25, Gaia DR3 counts as a
    comparison only (never a fit input).
(b, c) VIS-Y and Y-J against VIS magnitude: Q1 point sources as measured
    against model draws re-observed with each Q1 star's own flux errors.
(d) VIS-Y vs Y-J plane, Q1 and model 50/90% contours.

Run with the project env from the repo root:
    PYTHONPATH=. python paper_figures/build_star_figure.py
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullLocator

from euclid_polish.population.magnitude_law import StraightMagnitudeLaw
from euclid_polish.sky.generation.stellar_sed import StarColorModel
from euclid_polish.web.helpers.star_population import (
    _read_rows,
    _renoised_colors,
    _star_color_rows,
    euclid_catalog_path,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_figures" / "overleaf_model_stars" / "images"
PRIOR = ROOT / "data/population_comparison/calibrations/star_population_active.json"
COUNTS = ROOT / "data/population_comparison/q1_phz_star_counts.json"

INK, MUTED, GRID = "0.10", "0.45", "0.88"
Q1 = "#184f95"
MODEL = "#e0662f"
GAIA = "0.55"
RC = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}",
    "font.family": "serif",
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
}
FIG_W = 3.35  # AASTeX two-column \columnwidth (242.3 pt), inches


def weighted_quantile(x, w, q):
    order = np.argsort(x)
    cum = np.cumsum(w[order])
    return np.interp(q * cum[-1], cum, x[order])


def hpd_level(h, frac):
    v = np.sort(h.ravel())[::-1]
    return v[np.searchsorted(np.cumsum(v), frac * v.sum())]


def main() -> None:
    rng = np.random.default_rng(20261109)
    prior = json.loads(PRIOR.read_text())
    law = StraightMagnitudeLaw.from_payload(prior["population"]["magnitude_distribution"])
    model = StarColorModel.from_payload(prior["color_model"])
    density = prior["diagnostics"]["stellar_density_by_magnitude"]

    sel = _star_color_rows(_read_rows(euclid_catalog_path()))
    mag, col, w, flux = sel["magnitude"], sel["color"], sel["weight"], sel["fluxes"]

    # Model draws at each Q1 star's magnitude, measured with that star's errors.
    k_draws = 20
    mm, mc, mw = [], [], []
    for i in range(mag.size):
        for _ in range(k_draws):
            obs = None
            while obs is None:
                obs = _renoised_colors(model.sample_colors(rng, mag[i]), flux[i], rng)
            mm.append(mag[i])
            mc.append(obs)
            mw.append(w[i] / k_draws)
    mm, mc, mw = np.asarray(mm), np.asarray(mc), np.asarray(mw)

    counts = json.loads(COUNTS.read_text())["bins"]
    mid = np.array([(b["mag_lo"] + b["mag_hi"]) / 2 for b in counts])
    phz = np.array([b["density_arcmin2_mag"] for b in counts])
    phz_err = np.array([b["classification_sigma_arcmin2_mag"] for b in counts])

    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, 3.8))
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.195, top=0.985,
                        wspace=0.42, hspace=0.45)

    # (a) counts
    ax = axes[0, 0]
    ax.axvspan(law.fit_bright, law.fit_faint, color="0.93", lw=0, zorder=0)
    ok = phz > 0
    ax.errorbar(mid[ok], phz[ok], yerr=phz_err[ok], fmt="o", ms=1.4, color=Q1,
                elinewidth=0.5, zorder=3)
    gx = np.asarray(density["gaia_x"])
    gy = np.asarray(density["gaia_observed"])
    gaia_ok = gy > 0
    ax.plot(gx[gaia_ok], gy[gaia_ok], "s", ms=2.0, mfc="none", mec=GAIA, mew=0.5, zorder=2)
    m = np.linspace(law.mag_bright, law.mag_faint, 300)
    inside = (m >= law.fit_bright) & (m <= law.fit_faint)
    ax.semilogy(m, law.density(m), color=MODEL, lw=0.8, ls=(0, (3, 1.5)), zorder=4)
    ax.semilogy(m[inside], law.density(m[inside]), color=MODEL, lw=1.6, zorder=5)
    ax.set_xlim(12, 25)
    ax.set_ylim(5e-5, 30)
    ax.set_yticks([1e-4, 1e-2, 1e0])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r"VIS magnitude [AB]")
    ax.set_ylabel(r"$N$ [arcmin$^{-2}$\,mag$^{-1}$]")
    # Direct labels instead of a legend (the caption spells out solid/dashed).
    small = {"fontsize": 6.5, "va": "center"}
    ax.text(0.5 * (law.fit_bright + law.fit_faint), 9, "fit", ha="center",
            color=MUTED, **small)
    ax.text(24.9, 6, "law", ha="right", color=MODEL, **small)
    ax.text(0.5 * (law.fit_bright + law.fit_faint), 0.025, "Q1", ha="center", color=Q1, **small)
    ax.text(12.3, 6e-4, r"\textit{Gaia}", ha="left", color=GAIA, **small)

    # (b, c) colours against magnitude
    edges = np.arange(17.0, 23.51, 0.5)
    centres = 0.5 * (edges[1:] + edges[:-1])

    def bands(m_, c_, w_):
        out = np.full((centres.size, 3), np.nan)
        for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            s = (m_ >= lo) & (m_ < hi)
            if s.sum() > 5:
                out[j] = [weighted_quantile(c_[s], w_[s], q) for q in (0.16, 0.5, 0.84)]
        return out

    for ax, k, label in ((axes[0, 1], 0, r"VIS$\,-\,Y_{\rm E}$"),
                         (axes[1, 0], 1, r"$Y_{\rm E}-J_{\rm E}$")):
        qm = bands(mm, mc[:, k], mw)
        qq = bands(mag, col[:, k], w)
        ax.fill_between(centres, qm[:, 0], qm[:, 2], color=MODEL, alpha=0.22, lw=0)
        ax.plot(centres, qm[:, 1], color=MODEL, lw=1.3)
        ax.plot(centres, qq[:, 1], color=Q1, lw=0.9, marker="o", ms=1.6)
        ax.plot(centres, qq[:, 0], color=Q1, lw=0.6, ls=(0, (2, 1.2)))
        ax.plot(centres, qq[:, 2], color=Q1, lw=0.6, ls=(0, (2, 1.2)))
        ax.set_xlim(17, 23.5)
        ax.set_xlabel(r"VIS magnitude [AB]")
        ax.set_ylabel(label + " [mag]")

    # (d) colour-colour plane
    ax = axes[1, 1]
    xb = np.linspace(-0.1, 2.3, 49)
    yb = np.linspace(-0.25, 0.35, 49)
    xc, yc = 0.5 * (xb[1:] + xb[:-1]), 0.5 * (yb[1:] + yb[:-1])
    hq, _, _ = np.histogram2d(col[:, 0], col[:, 1], bins=[xb, yb], weights=w)
    hm, _, _ = np.histogram2d(mc[:, 0], mc[:, 1], bins=[xb, yb], weights=mw)
    hq, hm = gaussian_filter(hq, 1.0), gaussian_filter(hm, 1.0)
    ax.scatter(col[:, 0], col[:, 1], s=0.6, color=Q1, alpha=0.25, lw=0, rasterized=True)
    ax.contour(xc, yc, hm.T, levels=[hpd_level(hm, 0.9), hpd_level(hm, 0.5)],
               colors=MODEL, linewidths=[0.8, 1.3])
    ax.contour(xc, yc, hq.T, levels=[hpd_level(hq, 0.9), hpd_level(hq, 0.5)],
               colors=Q1, linewidths=[0.6, 0.9], linestyles=[(0, (2, 1.2)), "solid"])
    ax.set_xlim(-0.1, 2.3)
    ax.set_ylim(-0.25, 0.35)
    ax.set_xlabel(r"VIS$\,-\,Y_{\rm E}$ [mag]")
    ax.set_ylabel(r"$Y_{\rm E}-J_{\rm E}$ [mag]")
    fig.legend(handles=[
        Line2D([], [], color=Q1, lw=0.9, marker="o", ms=1.6, label=r"Q1 median; 50\%"),
        Line2D([], [], color=Q1, lw=0.6, ls=(0, (2, 1.2)), label=r"Q1 16--84\%; 90\%"),
        Line2D([], [], color=MODEL, lw=1.3, label=r"model median; 50, 90\%"),
        Patch(color=MODEL, alpha=0.22, lw=0, label=r"model 16--84\%"),
    ], loc="lower center", ncol=2, handlelength=1.8, columnspacing=1.5,
        labelspacing=0.25, bbox_to_anchor=(0.55, 0.0))

    for ax, tag in zip(axes.ravel(), "abcd", strict=True):
        ax.text(0.03, 0.97, f"({tag})", transform=ax.transAxes, va="top", ha="left",
                fontsize=7.5, color=INK, zorder=10,
                bbox={"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none"})
        ax.grid(color=GRID, lw=0.35)
        ax.set_axisbelow(True)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "star_population.pdf", facecolor="white")
    fig.savefig(OUT / "star_population.png", dpi=400, facecolor="white")
    print(f"{mag.size} Q1 colour rows, {mm.size} model draws, "
          f"density {law.integrated_density():.2f} arcmin^-2, slope {law.slope:.3f}")


if __name__ == "__main__":
    main()
