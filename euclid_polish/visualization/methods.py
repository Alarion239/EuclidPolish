"""
High-level visualization functions for EuclidPolish.

Each function accepts numpy arrays and an output path, builds a figure using
BaseVisualizer, and saves it.  All single-image functions share the same
layout (linear + log10 side-by-side) and differ only in title text.
"""

import numpy as np
import matplotlib.pyplot as plt

from euclid_polish.visualization.base import BaseVisualizer


def _draw_single(
    data: np.ndarray,
    output_path: str,
    title: str,
    clip_percentile: float = 99.5,
) -> None:
    """Core helper: 1×3 figure (linear | log10 | statistics) for a single image."""
    vis = BaseVisualizer(clip_percentile=clip_percentile, rows=1, cols=3, figsize=(22, 7))
    vis.add_scale_panel(data)
    vis.add_scale_panel(data, log_scale=True)
    vis.add_statistics_panel(data, {'title': 'Statistics:', 'include_data_stats': True})
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    vis.save_figure(output_path)


def draw_clean_image(
    data: np.ndarray,
    output_path: str,
    index: int | None = None,
    clip_percentile: float = 99.5,
) -> None:
    """Visualize a clean (HR) sky image."""
    title = f'Clean Sky Image {index:04d}' if index is not None else 'Clean Sky Image'
    _draw_single(data, output_path, title, clip_percentile)


def draw_dirty_image(
    data: np.ndarray,
    output_path: str,
    index: int | None = None,
    clip_percentile: float = 99.5,
) -> None:
    """Visualize a dirty (LR, PSF-convolved) image."""
    title = f'Dirty Image {index:04d}' if index is not None else 'Dirty Image'
    _draw_single(data, output_path, title, clip_percentile)


def draw_cutout(
    data: np.ndarray,
    output_path: str,
    star_id: int | None = None,
    clip_percentile: float = 99.5,
) -> None:
    """Visualize a Euclid cutout around a star."""
    title = f'Cutout — Star {star_id:04d}' if star_id is not None else 'Cutout'
    _draw_single(data, output_path, title, clip_percentile)


def draw_psf(
    data: np.ndarray,
    output_path: str,
    clip_percentile: float = 99.5,
) -> None:
    """Visualize a PSF."""
    _draw_single(data, output_path, 'Euclid VIS PSF', clip_percentile)


def draw_clean_dirty_pair(
    hr_data: np.ndarray,
    lr_data: np.ndarray,
    output_path: str,
    index: int | None = None,
    clip_percentile: float = 95.0,
) -> None:
    """
    Visualize a clean/dirty image pair.

    Layout (2×2):
        [HR linear]  [LR linear]
        [HR log10]   [LR log10]
    """
    vis = BaseVisualizer(clip_percentile=clip_percentile, rows=2, cols=3, figsize=(22, 12))
    vis.add_scale_panel(hr_data, title_suffix='\nHR Clean')
    vis.add_scale_panel(lr_data, title_suffix='\nLR Dirty')
    vis.add_statistics_panel(hr_data, {'title': 'HR Clean Stats:', 'include_data_stats': True})
    vis.add_scale_panel(hr_data, log_scale=True, title_suffix='\nHR Clean')
    vis.add_scale_panel(lr_data, log_scale=True, title_suffix='\nLR Dirty')
    vis.add_statistics_panel(lr_data, {'title': 'LR Dirty Stats:', 'include_data_stats': True})
    title = f'HR Clean vs LR Dirty — Image {index:05d}' if index is not None else 'HR Clean vs LR Dirty'
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    vis.save_figure(output_path)
