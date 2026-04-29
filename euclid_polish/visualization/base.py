"""
Shared visualization base for EuclidPolish.

Two stretch modes are supported on intensity panels:

* ``linear``  — straight imshow with percentile-clipped colour bounds. Good
                for inspecting the noise floor and morphology of typical
                pixels; saturated bright peaks read as "clipped" but the
                surrounding structure stays visible.
* ``asinh``   — ``arcsinh(x / scale)`` (Lupton+ 2004). Linear near zero
                (preserves the sky-subtracted noise structure including
                negatives) and logarithmic for ``|x| >> scale`` (compresses
                bright stars). ``scale`` defaults to the median absolute
                deviation of the image so that ±1 in stretched units roughly
                corresponds to ±1σ of the noise.
* ``log10``   — legacy log of clamped data. Useful for strictly-positive
                domains like PSFs and cutouts; misleading on sky-subtracted
                images that contain negatives.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from euclid_polish.config import Config
from typing import Dict, Any, Tuple


def _percentile_bounds(
    data: np.ndarray,
    lo: float = 1.0,
    hi: float = 99.5,
) -> Tuple[float, float]:
    """Return (vmin, vmax) at the given percentiles, with sane fallbacks.

    For *sparse* images (e.g. clean-HR with isolated point sources, where
    >99.5% of pixels are zero), the [p1, p99.5] window collapses to a
    degenerate range. In that case, fall back to the actual data range so
    bright pixels remain visible.
    """
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    full_min = float(finite.min())
    full_max = float(finite.max())
    if full_max <= full_min:
        return full_min, full_min + 1.0

    vmin, vmax = (float(v) for v in np.percentile(finite, [lo, hi]))
    range_full = full_max - full_min
    range_perc = vmax - vmin
    # Degenerate percentile window (sparse outliers dominate the dynamic range)
    if range_perc < 0.01 * range_full:
        return full_min, full_max
    if vmax <= vmin:
        return full_min, full_max
    return vmin, vmax


def _asinh_scale(data: np.ndarray) -> float:
    """Default asinh scale used by all training-aligned visualization.

    Returns ``Config.STRETCH_SCALE_E`` — the same scale the network trains in.
    Using the same constant everywhere makes the viz directly comparable
    across images and to the loss/PSNR metrics.

    The legacy MAD-based behaviour is still available via
    :func:`_asinh_scale_mad` for callers that explicitly want a per-image
    adaptive stretch (used by domains where the data is in different units,
    e.g. Q1 ADU/s cutouts).
    """
    return float(Config.STRETCH_SCALE_E)


def _asinh_scale_mad(data: np.ndarray) -> float:
    """Per-image asinh scale based on median absolute deviation.

    For sparse images where MAD = 0 (most pixels identical), fall back to a
    fraction of the dynamic range so the stretch still compresses bright
    pixels meaningfully.
    """
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 1.0
    mad = float(np.median(np.abs(finite - np.median(finite))))
    if mad > 1e-6:
        return mad
    full = float(finite.max() - finite.min())
    if full > 0:
        return full * 0.01
    std = float(finite.std())
    return std if std > 1e-6 else 1.0


class BaseVisualizer:
    """Thin wrapper around a matplotlib GridSpec figure."""

    def __init__(
        self,
        rows: int = 2,
        cols: int = 3,
        figsize=(16, 12),
        hspace: float = 0.3,
        wspace: float = 0.3,
        vmin: float | None = None,
        vmax: float | None = None,
    ):
        """
        Parameters
        ----------
        vmin, vmax : float, optional
            Linear-scale colour bounds. ``None`` triggers percentile-clipped
            bounds (1% / 99.5%) computed from the data of each panel.
        """
        self.vmin = vmin
        self.vmax = vmax
        self._ncols = cols
        self._next_panel = 0
        self._fig = plt.figure(figsize=figsize)
        self._gs = GridSpec(rows, cols, figure=self._fig, hspace=hspace, wspace=wspace)

    def _next_gs_position(self):
        row = self._next_panel // self._ncols
        col = self._next_panel % self._ncols
        self._next_panel += 1
        return self._gs[row, col]

    def add_scale_panel(
        self,
        data: np.ndarray,
        title_suffix: str = "",
        stretch: str = "linear",
        asinh_scale: float | None = None,
        colorbar_label: str = "Electrons",
        log_scale: bool | None = None,    # legacy — prefer ``stretch``
    ) -> None:
        """Add an intensity panel.

        Parameters
        ----------
        stretch : {"linear", "asinh", "log10"}
            How pixel values are mapped to colour. ``"asinh"`` is recommended
            for sky-subtracted electron data. ``"log10"`` is a legacy clamp-
            to-positive log for PSF/cutout-like inputs.
        asinh_scale : float, optional
            ``scale`` parameter for the asinh stretch. ``None`` picks the
            image's MAD (median absolute deviation).
        log_scale : bool, optional
            Deprecated. ``True`` is equivalent to ``stretch="log10"``.
        """
        if log_scale is not None:
            stretch = "log10" if log_scale else stretch

        ax = self._fig.add_subplot(self._next_gs_position())

        if stretch == "log10":
            fp16_min = float(np.finfo(np.float16).smallest_subnormal)
            display = np.log10(np.maximum(data, fp16_min))
            vmin, vmax = _percentile_bounds(display)
            title = f"log10{title_suffix}"
            cbar_label = f"log10({colorbar_label})"
            im = ax.imshow(display, cmap="viridis", origin="lower",
                           interpolation="nearest", vmin=vmin, vmax=vmax)
        elif stretch == "asinh":
            scale = asinh_scale if asinh_scale is not None else _asinh_scale(data)
            display = np.arcsinh(data / scale)
            vmin, vmax = _percentile_bounds(display)
            title = f"asinh (scale={scale:.3g}){title_suffix}"
            cbar_label = f"asinh({colorbar_label} / {scale:.3g})"
            im = ax.imshow(display, cmap="viridis", origin="lower",
                           interpolation="nearest", vmin=vmin, vmax=vmax)
        elif stretch == "linear":
            if self.vmin is None or self.vmax is None:
                vmin, vmax = _percentile_bounds(data)
            else:
                vmin, vmax = self.vmin, self.vmax
            title = f"linear [p1, p99.5]{title_suffix}" if self.vmax is None \
                else f"linear [{vmin:.3g}, {vmax:.3g}]{title_suffix}"
            cbar_label = colorbar_label
            im = ax.imshow(data, cmap="viridis", origin="lower",
                           interpolation="nearest", vmin=vmin, vmax=vmax)
        else:
            raise ValueError(f"Unknown stretch: {stretch!r}")

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        plt.colorbar(im, ax=ax, label=cbar_label)

    def add_diverging_panel(
        self,
        data: np.ndarray,
        asinh_scale: float | None = None,
        title_suffix: str = "",
        colorbar_label: str = "Residual (e⁻)",
        stretch: str = "asinh",
    ) -> None:
        """Add a signed/divergent panel (e.g. residual = HR − SR).

        Parameters
        ----------
        stretch : {"asinh", "linear"}
            ``asinh``: apply ``asinh(x / asinh_scale)`` first, then divergent
            colourmap with symmetric limits.
            ``linear``: plot raw values directly with symmetric percentile-
            based limits.

        Negative values (model over-predicted) render blue; positive
        (under-predicted) render red; zero is white.
        """
        if stretch == "asinh":
            scale = asinh_scale if asinh_scale is not None else _asinh_scale(data)
            display = np.arcsinh(data / scale)
            cbar_label = f"asinh({colorbar_label} / {scale:.3g})"
            title = f"residual asinh (scale={scale:.3g}){title_suffix}"
        elif stretch == "linear":
            display = data
            cbar_label = colorbar_label
            title = f"residual linear{title_suffix}"
        else:
            raise ValueError(f"Unknown stretch: {stretch!r}")

        # Symmetric vmin/vmax around 0 — high percentile of |display| so a
        # single extreme outlier doesn't flatten the colourbar.
        finite = display[np.isfinite(display)]
        v = float(np.percentile(np.abs(finite), 99.5)) if finite.size else 1.0
        if v <= 0:
            v = 1.0

        ax = self._fig.add_subplot(self._next_gs_position())
        im = ax.imshow(display, cmap="RdBu_r", origin="lower",
                       interpolation="nearest", vmin=-v, vmax=+v)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        plt.colorbar(im, ax=ax, label=cbar_label)

    def add_pixel_psnr_panel(
        self,
        residual_stretched: np.ndarray,
        max_val: float = 10.0,
        title_suffix: str = "",
        clip_db: tuple = (10.0, 80.0),
    ) -> None:
        """Add a per-pixel PSNR heatmap from a stretched-space residual.

        ``PSNR_i = 20 · log10(max_val / (|residual_i| + ε))``. Larger values
        (yellow) are pixels the model reconstructs well; smaller (purple) are
        the worst-fit pixels. Clipped to ``clip_db`` for a useful colourbar.
        """
        eps = 1e-7
        psnr = 20.0 * np.log10(max_val / (np.abs(residual_stretched) + eps))
        psnr = np.clip(psnr, clip_db[0], clip_db[1])

        ax = self._fig.add_subplot(self._next_gs_position())
        im = ax.imshow(psnr, cmap="viridis", origin="lower",
                       interpolation="nearest", vmin=clip_db[0], vmax=clip_db[1])
        ax.set_title(f"PSNR per pixel{title_suffix}", fontsize=12)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        plt.colorbar(im, ax=ax, label="PSNR (dB)")

    def add_residual_histogram_panel(
        self,
        residual_stretched: np.ndarray,
        title_suffix: str = "",
    ) -> None:
        """Add a histogram of stretched residuals + overlay simple stats."""
        ax = self._fig.add_subplot(self._next_gs_position())
        flat = residual_stretched[np.isfinite(residual_stretched)].ravel()
        if flat.size == 0:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
            return
        # Clip the tail for binning so the histogram isn't dominated by outliers
        lo, hi = np.percentile(flat, [0.5, 99.5])
        ax.hist(np.clip(flat, lo, hi), bins=80, color="tab:gray", edgecolor="none")
        ax.axvline(0.0, color="red", lw=1.0, ls="--", alpha=0.8)
        ax.set_xlabel("residual (asinh space)")
        ax.set_ylabel("count")
        ax.set_title(f"residual histogram{title_suffix}", fontsize=12)
        # Stats overlay
        mean = float(np.mean(flat))
        std  = float(np.std(flat))
        med  = float(np.median(flat))
        ax.text(0.02, 0.95,
                f"mean = {mean:+.3g}\nmedian = {med:+.3g}\nstd  = {std:.3g}\nN = {flat.size}",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=9, fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))

    def add_statistics_panel(self, data: np.ndarray, stats_dict: Dict[str, Any]) -> None:
        ax = self._fig.add_subplot(self._next_gs_position())
        ax.axis("off")

        lines = [stats_dict.get("title", "Statistics:"), "=" * 30]

        for key, value in stats_dict.get("stats", {}).items():
            lines.append(f"{key}: {value}")

        if stats_dict.get("include_data_stats", True):
            finite = data[np.isfinite(data)]
            p1, p50, p99, p999 = np.percentile(finite, [1, 50, 99, 99.9]) if finite.size else (0, 0, 0, 0)
            lines.append(f"\nShape: {data.shape[0]} x {data.shape[1]} pixels")
            lines.append(f"Min:    {finite.min():.4g}")
            lines.append(f"p1:     {p1:.4g}")
            lines.append(f"Median: {p50:.4g}")
            lines.append(f"p99:    {p99:.4g}")
            lines.append(f"p99.9:  {p999:.4g}")
            lines.append(f"Max:    {finite.max():.4g}")
            lines.append(f"Std:    {finite.std():.4g}")
            lines.append(f"% < 0:  {(data < 0).mean() * 100:.1f}%")

        ax.text(
            0.1, 0.5, "\n".join(lines),
            transform=ax.transAxes, fontsize=10,
            verticalalignment="center", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

    def save_figure(self, output_path: str, dpi: int = 150, close: bool = True) -> None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        self._fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        if close:
            plt.close(self._fig)
            self._fig = None
            self._gs = None
