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

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from euclid_polish.visualization.base import (
    BaseVisualizer,
    _asinh_scale,
    _asinh_scale_mad,
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


def draw_star_positions(
    stars: list[dict],
    output_path: str,
) -> None:
    """RA/Dec scatter of catalog stars (unchanged)."""
    ra = [s['ra'] for s in stars]
    dec = [s['dec'] for s in stars]
    mag = [s['magnitude'] for s in stars]
    corrupted = [s.get('corrupted', False) for s in stars]

    sizes = [(21 - m) ** 2 * 2 for m in mag]

    fig, ax = plt.subplots(figsize=(12, 8))

    valid_ra = [r for r, c in zip(ra, corrupted) if not c]
    valid_dec = [d for d, c in zip(dec, corrupted) if not c]
    valid_sizes = [sz for sz, c in zip(sizes, corrupted) if not c]
    valid_mag = [m for m, c in zip(mag, corrupted) if not c]
    n_valid = len(valid_ra)

    sc = ax.scatter(
        valid_ra, valid_dec, s=valid_sizes, c=valid_mag,
        cmap='YlOrRd_r', alpha=0.7, edgecolors='none',
        label=f'Valid ({n_valid})',
    )

    corr_ra = [r for r, c in zip(ra, corrupted) if c]
    corr_dec = [d for d, c in zip(dec, corrupted) if c]
    corr_sizes = [sz for sz, c in zip(sizes, corrupted) if c]
    n_corrupted = len(corr_ra)
    if corr_ra:
        ax.scatter(
            corr_ra, corr_dec, s=[sz * 1.5 for sz in corr_sizes],
            c='red', marker='x', linewidths=1.5,
            label=f'Corrupted ({n_corrupted})',
        )

    cbar = plt.colorbar(sc, ax=ax, label='Magnitude')
    ax.set_xlabel('RA (degrees)')
    ax.set_ylabel('Dec (degrees)')
    ax.set_title(f'Euclid Star Positions ({len(stars)} stars)')
    ax.invert_xaxis()

    ax.set_facecolor('#0a0a2a')
    fig.patch.set_facecolor('#1a1a3a')
    for item in [cbar.ax.yaxis.label, ax.xaxis.label, ax.yaxis.label, ax.title]:
        item.set_color('white')
    for tick_ax in [cbar.ax, ax]:
        tick_ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.legend(
        loc='upper left', facecolor='#2a2a4a',
        edgecolor='white', labelcolor='white',
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
