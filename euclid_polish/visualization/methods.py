"""
High-level visualization functions for EuclidPolish.

For sky / training data (raw electrons, sky-subtracted) the standard panel
layout is **linear (percentile-clipped) | asinh | stats**. The asinh panel
preserves the noise structure including negative pixels and compresses bright
stars; the linear panel is clipped to the 1–99.5 percentile range so a
single saturated pixel doesn't flatten the colourbar.

For strictly-positive domains (PSFs, Q1 cutouts) the second panel is log10
instead of asinh.
"""

import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.ndimage import gaussian_filter

from euclid_polish.visualization.base import (
    BaseVisualizer,
    _asinh_scale,
    _asinh_scale_mad,
)
from euclid_polish.visualization.presentation_style import (
    AXIS_LABEL_SIZE,
    LEGEND_SIZE,
    PANEL_TITLE_SIZE,
    TICK_LABEL_SIZE,
    apply_presentation_figure,
)


def _smooth_for_display(data: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Blur sparse delta-function inputs so point sources become visible.

    A clean HR image is mostly zeros with a handful of single-pixel stars;
    rendering it directly produces a near-blank figure. A small Gaussian
    smear (sigma ≈ 1–2 HR pixels) makes each source a visible blob without
    changing the integrated flux.
    """
    return gaussian_filter(data.astype(np.float64), sigma=sigma).astype(np.float32)


def _is_sparse(data: np.ndarray, threshold: float = 0.99) -> bool:
    """True if more than ``threshold`` fraction of pixels are zero."""
    return float((data == 0).mean()) > threshold


def _draw_single_electrons(
    data: np.ndarray,
    output_path: str,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
    asinh_scale: float | None = None,
) -> None:
    """1×3 figure (linear-clipped | asinh | stats) for sky/training data."""
    vis = BaseVisualizer(rows=1, cols=3, figsize=(22, 7), vmin=vmin, vmax=vmax)
    vis.add_scale_panel(data, stretch="linear")
    vis.add_scale_panel(data, stretch="asinh", asinh_scale=asinh_scale)
    vis.add_statistics_panel(data, {"title": "Statistics:", "include_data_stats": True})
    plt.suptitle(title, fontsize=14)
    vis.save_figure(output_path)


def _draw_single_positive(
    data: np.ndarray,
    output_path: str,
    title: str,
) -> None:
    """1×3 figure (linear-clipped | log10 | stats) for strictly-positive data."""
    vis = BaseVisualizer(rows=1, cols=3, figsize=(22, 7))
    vis.add_scale_panel(data, stretch="linear")
    vis.add_scale_panel(data, stretch="log10")
    vis.add_statistics_panel(data, {"title": "Statistics:", "include_data_stats": True})
    plt.suptitle(title, fontsize=14)
    vis.save_figure(output_path)


def draw_clean_image(
    data: np.ndarray,
    output_path: str,
    index: int | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Visualize a clean (HR) sky image (electrons; mostly zero with sources).

    Sparse delta-function frames (e.g. fields with only stars) get a small
    Gaussian smear for display so point sources are visible. The smear is
    cosmetic only.
    """
    title = f"Clean Sky Image {index:04d}" if index is not None else "Clean Sky Image"
    display = _smooth_for_display(data) if _is_sparse(data) else data
    _draw_single_electrons(display, output_path, title, vmin=vmin, vmax=vmax)


def draw_dirty_image(
    data: np.ndarray,
    output_path: str,
    index: int | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Visualize a dirty (LR, PSF-convolved + noise) image (sky-subtracted electrons)."""
    title = f"Dirty Image {index:04d}" if index is not None else "Dirty Image"
    _draw_single_electrons(data, output_path, title, vmin=vmin, vmax=vmax)


def draw_cutout(
    data: np.ndarray,
    output_path: str,
    star_id: int | None = None,
) -> None:
    """Visualize a Euclid Q1 cutout (ADU/s, sky-subtracted — has negatives).

    Cutouts are in different units from the simulator (ADU/s vs electrons),
    so the simulator's STRETCH_SCALE_E doesn't apply dimensionally. Use a
    per-image MAD scale instead.
    """
    title = f"Cutout — Star {star_id:04d}" if star_id is not None else "Cutout"
    _draw_single_electrons(data, output_path, title, asinh_scale=_asinh_scale_mad(data))


def draw_psf(data: np.ndarray, output_path: str) -> None:
    """Visualize a PSF (strictly positive, large dynamic range → log10)."""
    _draw_single_positive(data, output_path, "Euclid VIS PSF")


def draw_clean_dirty_pair(
    hr_data: np.ndarray,
    lr_data: np.ndarray,
    output_path: str,
    index: int | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """
    Visualize a clean / dirty pair on a shared asinh scale.

    Layout (2×3):
        [HR linear]  [LR linear]  [HR stats]
        [HR asinh]   [LR asinh]   [LR stats]

    The asinh ``scale`` is taken from the *LR* MAD (the dirty image carries
    the noise floor that defines the natural unit of "small"). Both panels
    use the same scale so brightness is comparable across HR/LR.
    """
    # Use the same asinh scale the network trains in, so the viz is
    # directly comparable to the loss/PSNR metrics.
    shared_scale = _asinh_scale(lr_data)  # = Config.STRETCH_SCALE_E
    hr_display = _smooth_for_display(hr_data) if _is_sparse(hr_data) else hr_data

    vis = BaseVisualizer(rows=2, cols=3, figsize=(22, 12), vmin=vmin, vmax=vmax)
    vis.add_scale_panel(hr_display, stretch="linear", title_suffix="\nHR Clean")
    vis.add_scale_panel(lr_data, stretch="linear", title_suffix="\nLR Dirty")
    vis.add_statistics_panel(hr_data, {"title": "HR Clean Stats:", "include_data_stats": True})
    vis.add_scale_panel(hr_display, stretch="asinh", asinh_scale=shared_scale, title_suffix="\nHR Clean")
    vis.add_scale_panel(lr_data, stretch="asinh", asinh_scale=shared_scale, title_suffix="\nLR Dirty")
    vis.add_statistics_panel(lr_data, {"title": "LR Dirty Stats:", "include_data_stats": True})

    title = f"HR Clean vs LR Dirty — Image {index:05d}" if index is not None else "HR Clean vs LR Dirty"
    plt.suptitle(title, fontsize=14)
    vis.save_figure(output_path)


def _star_arrays(stars: list[dict]):
    """Decompose the catalog into numpy arrays for the two plots below."""
    ra   = np.array([s['ra']        for s in stars], dtype=np.float64)
    dec  = np.array([s['dec']       for s in stars], dtype=np.float64)
    mag  = np.array([s['magnitude'] for s in stars], dtype=np.float64)
    corr = np.array([bool(s.get('corrupted', False)) for s in stars])
    return ra, dec, mag, corr


def plot_star_positions(stars: list[dict], fig=None):
    """Render the catalog's sky positions on an Aitoff projection.

    If all stars cluster in a small region (≲ 10° in RA or Dec) we add
    a tangent-plane zoom panel on the right; otherwise the Aitoff
    full-sky view is the whole figure.

    Returns the populated matplotlib figure (caller saves / displays).
    """

    ra, dec, mag, corr = _star_arrays(stars)
    if ra.size == 0:
        if fig is None:
            fig = plt.figure(figsize=(12, 6.5))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "no stars in catalog",
                ha="center", va="center", color="#666",
                fontsize=PANEL_TITLE_SIZE)
        ax.set_axis_off()
        return fig

    # Build SkyCoord once and reuse its wrapped values for Aitoff.
    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    # Negate wrapped longitude so right ascension increases to the left, as
    # required for conventional astronomical sky maps.
    ra_rad  = -coords.ra.wrap_at(180 * u.deg).radian
    dec_rad = coords.dec.radian
    # Visual sizing — brighter stars (lower mag) → larger dots, capped.
    # Range floored so even faint stars are visible against the sky grid.
    crowding_scale = max(0.22, min(1.0, (5000.0 / len(stars)) ** 0.5))
    sizes = np.clip((21.0 - mag) ** 2 * 2.0, 10.0, 80.0) * crowding_scale

    # Decide whether to show a zoom panel: yes if the cluster is small.
    ra_span  = float(ra.max()  - ra.min())
    dec_span = float(dec.max() - dec.min())
    show_zoom = (ra_span < 10.0 and dec_span < 10.0 and len(stars) > 1)

    if fig is None:
        fig = plt.figure(figsize=(16, 7.5) if show_zoom else (13, 7.5))
    if show_zoom:
        ax_sky  = fig.add_subplot(1, 2, 1, projection="aitoff")
        ax_zoom = fig.add_subplot(1, 2, 2)
    else:
        ax_sky  = fig.add_subplot(1, 1, 1, projection="aitoff")
        ax_zoom = None

    # ---- Full-sky Aitoff scatter ----
    valid = ~corr
    corrupted_scatter = None
    if corr.any():
        corrupted_scatter = ax_sky.scatter(
            ra_rad[corr], dec_rad[corr], s=sizes[corr],
            c="#4b5563", marker="x", linewidths=0.75, alpha=0.28,
            label=f"corrupted ({int(corr.sum())})",
            rasterized=True,
        )
    sc = ax_sky.scatter(
        ra_rad[valid], dec_rad[valid], s=sizes[valid], c=mag[valid],
        cmap="YlOrRd_r", alpha=0.72, edgecolors="none",
        label=f"valid ({int(valid.sum())})", rasterized=True,
    )
    colorbar = fig.colorbar(sc, ax=ax_sky, shrink=0.72, pad=0.05)
    colorbar.set_label("VIS magnitude (AB)", fontsize=AXIS_LABEL_SIZE)
    colorbar.ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax_sky.set_title(
        f"Stellar catalog — {len(stars):,} sources (ICRS Aitoff)",
        pad=24, fontsize=PANEL_TITLE_SIZE,
    )
    ax_sky.grid(True, color="#888", alpha=0.3)
    # Matplotlib places Aitoff ticks at −150°…150°. With the negated longitude
    # above, label those positions as ICRS RA in the conventional 0°…360° form.
    ax_sky.set_xticklabels(
        ["150°", "120°", "90°", "60°", "30°", "0°",
         "330°", "300°", "270°", "240°", "210°"],
    )
    legend_handles = [sc]
    if corrupted_scatter is not None:
        legend_handles.append(corrupted_scatter)
    ax_sky.legend(
        handles=legend_handles, loc="lower left",
        fontsize=LEGEND_SIZE, frameon=False,
    )

    # ---- Optional tangent-plane zoom on the cluster ----
    if ax_zoom is not None:
        ax_zoom.scatter(
            ra[valid], dec[valid], s=sizes[valid], c=mag[valid],
            cmap="YlOrRd_r", alpha=0.8, edgecolors="none",
        )
        if corr.any():
            ax_zoom.scatter(
                ra[corr], dec[corr], s=sizes[corr], c="#4b5563",
                marker="x", linewidths=0.75, alpha=0.28,
                rasterized=True,
            )
        ax_zoom.invert_xaxis()                      # RA increases leftward
        ax_zoom.set_xlabel("ICRS RA (deg)", fontsize=AXIS_LABEL_SIZE)
        ax_zoom.set_ylabel("ICRS Dec (deg)", fontsize=AXIS_LABEL_SIZE)
        ax_zoom.set_aspect("equal", adjustable="datalim")
        ax_zoom.grid(True, alpha=0.3)
        ax_zoom.set_title(
            f"Zoom — ΔRA≈{ra_span:.2f}°  ΔDec≈{dec_span:.2f}°",
            fontsize=PANEL_TITLE_SIZE,
        )

    apply_presentation_figure(fig)
    fig.tight_layout()
    return fig


def draw_star_positions(
    stars: list[dict],
    output_path: str,
) -> None:
    """Save :func:`plot_star_positions` to ``output_path``."""
    fig = plot_star_positions(stars)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
