"""
Shared visualization base for EuclidPolish.

This module provides common visualization functionality used across multiple
visualization scripts, eliminating code duplication.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, Any


class BaseVisualizer:
    """Base class for visualizations with common plot layouts."""

    def __init__(
        self,
        clip_percentile: float = 99.5,
        rows: int = 2,
        cols: int = 3,
        figsize=(16, 12),
        hspace: float = 0.3,
        wspace: float = 0.3,
    ):
        """
        Initialize the visualizer and create the figure.

        Parameters:
        -----------
        clip_percentile : float
            Percentile for clipping in linear scale plots.
        rows : int
            Number of GridSpec rows.
        cols : int
            Number of GridSpec columns.
        figsize : tuple
            Figure size in inches.
        hspace : float
            Vertical spacing between subplots.
        wspace : float
            Horizontal spacing between subplots.
        """
        self.clip_percentile = clip_percentile
        self._ncols = cols
        self._next_panel = 0
        self._fig = plt.figure(figsize=figsize)
        self._gs = GridSpec(rows, cols, figure=self._fig, hspace=hspace, wspace=wspace)

    def _next_gs_position(self):
        """Return the next GridSpec slot and advance the internal counter."""
        row = self._next_panel // self._ncols
        col = self._next_panel % self._ncols
        self._next_panel += 1
        return self._gs[row, col]

    def add_scale_panel(self, data, title_suffix: str = "", log_scale: bool = False):
        """
        Add a scale plot panel (linear with clipping or log10).

        Parameters:
        -----------
        data : numpy.ndarray
            2D data array.
        title_suffix : str
            Additional suffix for the title.
        log_scale : bool
            If True, display log10 scale (FP16-safe). If False, display linear scale with clipping.
        """
        ax = self._fig.add_subplot(self._next_gs_position())
        if log_scale:
            fp16_min = np.finfo(np.float16).smallest_subnormal
            display_data = np.log10(np.maximum(data, fp16_min))
            title = f'Log10 Scale{title_suffix}'
            colorbar_label = 'log10(Flux)'
        else:
            clip_value = np.percentile(data, self.clip_percentile)
            display_data = np.clip(data, None, clip_value)
            title = f'Linear Scale (clipped at {self.clip_percentile}%){title_suffix}'
            colorbar_label = 'Flux'
        im = ax.imshow(display_data, cmap='viridis', origin='lower', interpolation='nearest')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax, label=colorbar_label)

    def add_statistics_panel(self, data, stats_dict: Dict[str, Any]):
        """
        Add statistics text panel.

        Parameters:
        -----------
        data : numpy.ndarray
            2D data array.
        stats_dict : dict
            Dictionary of statistics to display.
        """
        ax = self._fig.add_subplot(self._next_gs_position())
        ax.axis('off')

        # Build stats text
        lines = [stats_dict.get('title', 'Statistics:'), '=' * 30]

        for key, value in stats_dict.get('stats', {}).items():
            lines.append(f"{key}: {value}")

        # Add data stats if provided
        if stats_dict.get('include_data_stats', True):
            lines.append(f"\nShape: {data.shape[0]} x {data.shape[1]} pixels")
            lines.append(f"Peak: {np.max(data):.6e}")
            lines.append(f"Median: {np.median(data):.6e}")
            lines.append(f"Std Dev: {np.std(data):.6e}")

        stats_text = '\n'.join(lines)

        ax.text(
            0.1,
            0.5,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
        )

    def save_figure(self, output_path: str, dpi: int = 150, close: bool = True):
        """
        Save the current figure to file.

        Parameters:
        -----------
        output_path : str
            Output file path.
        dpi : int
            DPI for saved figure.
        close : bool
            Whether to close the figure after saving.
        """
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        self._fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        if close:
            plt.close(self._fig)
            self._fig = None
            self._gs = None

