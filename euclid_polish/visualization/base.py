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

from euclid_polish.config import Config as _Cfg
from euclid_polish.visualization.color import (
    calibrated_rgb_panel, eye_rgb, planck_color_strip,
)


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
    across images and to the loss / PSNR metrics.

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
        cmap: str = "viridis",
        colorbar_inset: bool = False,
        hi_percentile: float | None = None,
    ) -> None:
        """Add an intensity panel.

        ``colorbar_inset``: draw the colorbar as a slim inset to the RIGHT of
        the image (like :meth:`add_rgb_scale_panel`'s temperature legend)
        instead of letting matplotlib steal space from the axes. This keeps
        the image filling its full grid cell, so a grayscale panel sits the
        same on-screen size as a neighbouring colour (RGB) panel that has no
        colorbar — used for the LR-vs-SR side-by-side cutout view.

        ``hi_percentile``: upper percentile for the colour-bound clip (default
        99.5). Raising it toward 100 lowers the displayed contrast ceiling so a
        bright compact source (e.g. a lens's central galaxy) stops saturating
        to flat white and its internal structure becomes visible. Applies in
        every stretch (the bound is taken in the post-stretch display space for
        asinh / log10).

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
        cmap : str
            Matplotlib colormap. Default ``"viridis"``; pass ``"gray"`` or
            ``"gray_r"`` to render a panel as true grayscale (used for the
            VIS-only SR cells in the reconstruction plot, since the model
            has no multi-band information to colour with).
        """
        if log_scale is not None:
            stretch = "log10" if log_scale else stretch

        ax = self._fig.add_subplot(self._next_gs_position())
        _hi = hi_percentile if hi_percentile is not None else 99.5

        if stretch == "log10":
            fp16_min = float(np.finfo(np.float16).smallest_subnormal)
            display = np.log10(np.maximum(data, fp16_min))
            vmin, vmax = _percentile_bounds(display, hi=_hi)
            title = f"log10{title_suffix}"
            cbar_label = f"log10({colorbar_label})"
            im = ax.imshow(display, cmap=cmap, origin="lower",
                           interpolation="nearest", vmin=vmin, vmax=vmax)
        elif stretch == "asinh":
            scale = asinh_scale if asinh_scale is not None else _asinh_scale(data)
            display = np.arcsinh(data / scale)
            vmin, vmax = _percentile_bounds(display, hi=_hi)
            title = f"asinh (scale={scale:.3g}){title_suffix}"
            cbar_label = f"asinh({colorbar_label} / {scale:.3g})"
            im = ax.imshow(display, cmap=cmap, origin="lower",
                           interpolation="nearest", vmin=vmin, vmax=vmax)
        elif stretch == "linear":
            if self.vmin is None or self.vmax is None:
                vmin, vmax = _percentile_bounds(data, hi=_hi)
            else:
                vmin, vmax = self.vmin, self.vmax
            title = f"linear [p1, p{_hi:g}]{title_suffix}" if self.vmax is None \
                else f"linear [{vmin:.3g}, {vmax:.3g}]{title_suffix}"
            cbar_label = colorbar_label
            im = ax.imshow(data, cmap=cmap, origin="lower",
                           interpolation="nearest", vmin=vmin, vmax=vmax)
        else:
            raise ValueError(f"Unknown stretch: {stretch!r}")

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        if colorbar_inset:
            # Slim inset to the right — does NOT shrink the image axes, so the
            # panel stays the same size as a colorbar-less RGB panel beside it.
            cax = ax.inset_axes([1.03, 0.0, 0.05, 1.0])
            self._fig.colorbar(im, cax=cax, label=cbar_label)
        else:
            plt.colorbar(im, ax=ax, label=cbar_label)

    def add_rgb_scale_panel(
        self,
        cube: np.ndarray,
        band_names: tuple = None,
        stretch: str = "asinh",
        asinh_scale: float | None = None,
        title_suffix: str = "",
        rgb_mode: str = "calibrated",
        temp_legend: bool = False,
    ) -> None:
        """Color version of :meth:`add_scale_panel` for 4-band cubes.

        ``rgb_mode``:

          * ``"calibrated"`` (default) — solar-balanced per-channel
            stretch with per-image ``[p1, p99.5]`` windows, matching the
            grayscale panels' adaptive dynamic-range convention.
          * ``"eye"`` — the PHYSICAL mode: per-pixel blackbody color
            temperature → CIE Planckian-locus chromaticity → sRGB, with
            an ABSOLUTE luminance transfer keyed to ``asinh_scale``
            (no per-image normalisation). Same (SED, brightness) →
            same color in every image, so hues are directly comparable
            across panels/scenes/runs — and read like the night sky
            (Sun-like ≈ white, cool ≈ orange, hot ≈ blue).

        ``temp_legend`` (eye mode only) draws a slim blackbody-T → hue
        colorbar to the RIGHT of the image so the hue can be read back as
        a physical SED temperature.
        """
        if band_names is None:
            band_names = _Cfg.LR_INPUT_BAND_NAMES
        scale = float(asinh_scale) if asinh_scale is not None \
            else float(_Cfg.STRETCH_SCALE_E)
        if rgb_mode == "eye":
            rgb = eye_rgb(cube, band_names=band_names,
                          stretch=stretch, asinh_scale_e=scale)
            title = (f"eye color (Planck T · {stretch}, "
                     f"knee={scale:.3g} e⁻){title_suffix}")
        elif rgb_mode == "calibrated":
            rgb = calibrated_rgb_panel(
                cube, band_names=band_names,
                scheme="vis_nisp", reference="solar",
                stretch=stretch, asinh_scale_e=scale,
            )
            if stretch == "linear":
                title = f"linear (4-band solar) [p1, p99.5]{title_suffix}"
            else:
                title = (f"asinh (scale={scale:.3g}, 4-band solar) "
                         f"[p1, p99.5]{title_suffix}")
        else:
            raise ValueError(
                f"rgb_mode must be 'calibrated' or 'eye'; got {rgb_mode!r}"
            )
        ax = self._fig.add_subplot(self._next_gs_position())
        ax.imshow(np.clip(rgb, 0.0, 1.0), origin="lower",
                  interpolation="nearest")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        if temp_legend and rgb_mode == "eye":
            # Blackbody-temperature legend: the hue ↔ T_eff dictionary
            # for reading physics back out of the panel. Drawn as a slim
            # VERTICAL colorbar to the right of the image (cool at the
            # bottom, hot at the top), like a standard matplotlib cbar.
            strip, temps = planck_color_strip()
            # planck_color_strip returns (1, n, 3); transpose to a column
            # (n, 1, 3) so the bar runs vertically with origin="lower".
            strip_col = np.transpose(strip, (1, 0, 2))
            lax = ax.inset_axes([1.03, 0.0, 0.05, 1.0])
            lax.imshow(strip_col, aspect="auto", origin="lower",
                       extent=[0, 1, 0, len(temps) - 1])
            ticks_k = [3000, 4000, 6000, 10000, 20000]
            log_t = np.log(temps)
            lax.set_yticks([
                float(np.interp(np.log(t), log_t, np.arange(len(temps))))
                for t in ticks_k
            ])
            lax.set_yticklabels([f"{t // 1000}k" for t in ticks_k],
                                fontsize=7)
            lax.yaxis.tick_right()
            lax.yaxis.set_label_position("right")
            lax.set_xticks([])
            lax.set_ylabel("blackbody T (K)", fontsize=8, labelpad=2)

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

    def add_relative_error_panel(
        self,
        residual: np.ndarray,
        signal: np.ndarray,
        floor: float,
        title_suffix: str = "",
        clip: tuple = (1e-3, 1.0),
        cmap: str = "magma",
    ) -> None:
        """Add a per-pixel **relative error** map: ``|residual| / max(|signal|, floor)``.

        For pixels well above the noise floor, this reads as the fraction of
        signal the model gets wrong (e.g. 0.05 = 5% photometric error). For
        sky-floor pixels where ``|signal| < floor``, the denominator clamps
        to ``floor`` so the display reduces to a noise-normalised z-score
        instead of blowing up to ∞.

        Parameters
        ----------
        residual : ndarray
            ``HR − SR`` in the same units as ``signal``.
        signal : ndarray
            Reference signal (typically HR) in the same units as ``residual``.
        floor : float
            Minimum value used for the denominator — typically the noise std
            in the same units as ``residual``. Anything ≥ noise level works.
        clip : (low, high)
            Log-display range. Default ``[1e-3, 1.0]`` shows three decades:
            <0.1% (yellow → black), 1% (mid), 10% (orange), ≥100% (saturated
            yellow / model failure).
        cmap : str
            ``magma``: black at low rel-err (good), bright yellow at high
            (bad).
        """
        eps = 1e-7
        denom = np.maximum(np.abs(signal), float(floor))
        rel = np.abs(residual) / (denom + eps)
        rel = np.clip(rel, clip[0], clip[1])

        ax = self._fig.add_subplot(self._next_gs_position())
        im = ax.imshow(rel, cmap=cmap, origin="lower", interpolation="nearest",
                       norm=plt.matplotlib.colors.LogNorm(vmin=clip[0], vmax=clip[1]))
        ax.set_title(
            f"|residual| / max(|signal|, floor)  (floor={floor:.3g}){title_suffix}",
            fontsize=12,
        )
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        plt.colorbar(im, ax=ax, label="|Δ| / |signal|")

    def add_statistics_panel(self, data: np.ndarray, stats_dict: Dict[str, Any]) -> None:
        """Render a fixed-width key/value statistics panel.

        Caller-provided ``stats_dict["stats"]`` is rendered as ``key  value``
        with the keys left-padded and the values right-padded so that all
        rows line up in monospace. Numeric values that come in as plain
        floats / ints are auto-formatted (``.4e`` for floats, ``d`` for
        ints); pre-formatted strings are passed through unchanged.
        """
        ax = self._fig.add_subplot(self._next_gs_position())
        ax.axis("off")

        title = stats_dict.get("title", "Statistics:")
        stats_items = list(stats_dict.get("stats", {}).items())

        if stats_dict.get("include_data_stats", True):
            finite = data[np.isfinite(data)]
            if finite.size:
                p1, p50, p99, p999 = np.percentile(finite, [1, 50, 99, 99.9])
            else:
                p1 = p50 = p99 = p999 = 0.0
            stats_items += [
                ("Shape",  f"{data.shape[0]} × {data.shape[1]}"),
                ("min",    float(finite.min()) if finite.size else 0.0),
                ("p1",     float(p1)),
                ("median", float(p50)),
                ("p99",    float(p99)),
                ("p99.9",  float(p999)),
                ("max",    float(finite.max()) if finite.size else 0.0),
                ("std",    float(finite.std()) if finite.size else 0.0),
                ("% < 0",  f"{(data < 0).mean() * 100:>9.2f} %"),
            ]

        # Auto-format raw numeric values; leave pre-formatted strings alone.
        formatted = []
        for k, v in stats_items:
            if isinstance(v, str):
                s = v
            elif isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                s = f"{int(v):>11d}"
            elif isinstance(v, (float, np.floating)):
                s = f"{float(v):>+11.4e}"
            else:
                s = str(v)
            formatted.append((str(k), s))

        # Column widths for clean alignment.
        key_w = max((len(k) for k, _ in formatted), default=0)
        val_w = max((len(v) for _, v in formatted), default=0)
        bar_w = key_w + 2 + val_w
        lines = [title, "─" * bar_w]
        for k, v in formatted:
            lines.append(f"{k:<{key_w}}  {v:>{val_w}}")

        ax.text(
            0.05, 0.95, "\n".join(lines),
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

    def save_figure(self, output_path: str, dpi: int = 150, close: bool = True) -> None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        self._fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        if close:
            plt.close(self._fig)
            self._fig = None
            self._gs = None
