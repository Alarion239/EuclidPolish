"""LR→SR(→HR) reconstruction figures.

The visualization layer sits *above* the data layer: it imports
:class:`~euclid_polish.image.Image` / :class:`~euclid_polish.image.ImageSet` and
builds rich figures on top of them. (The image package stays a leaf and never
imports visualization.) Two entry points:

* :func:`plot_reconstruction` — the array-level renderer (used across the CLI,
  WebUI and eval runners; ``training.inference`` re-exports it for back-compat).
* :func:`plot_imageset` — the OO entry: hand it an ``ImageSet`` and it picks the
  LR / SR / HR images by role and renders them.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from euclid_polish.config import Config
from euclid_polish.image import Image, ImageSet, Role
from euclid_polish.visualization.base import BaseVisualizer


def _image_stats(data: np.ndarray) -> dict:
    """Per-image statistics. Values are raw numerics — the stats panel
    auto-formats them consistently (``.4e`` for floats)."""
    finite = data[np.isfinite(data)]
    p1, p10, p50, p90, p99, p999 = np.percentile(finite, [1, 10, 50, 90, 99, 99.9])
    return {
        "Shape":           f"{data.shape[0]} × {data.shape[1]}",
        "Dtype":           str(data.dtype),
        "min":             float(finite.min()),
        "p1":              float(p1),
        "p10":             float(p10),
        "median":          float(p50),
        "p90":             float(p90),
        "p99":             float(p99),
        "p99.9":           float(p999),
        "max":             float(finite.max()),
        "mean":            float(np.mean(finite)),
        "std":             float(np.std(finite)),
        "% < 0":           f"{(data < 0).mean() * 100:>9.2f} %",
        "Total flux":      float(np.sum(finite)),
        "# bright (>p99)": int((finite > p99).sum()),
    }


def _residual_asinh_stats(residual_stretched: np.ndarray, residual_e: np.ndarray) -> dict:
    """Stats describing the residual structure in asinh space."""
    finite = residual_stretched[np.isfinite(residual_stretched)]
    p1, p50, p99 = np.percentile(finite, [1, 50, 99])
    abs_err = np.abs(finite)
    return {
        "mean (bias)":           float(np.mean(finite)),
        "median":                float(p50),
        "std (noise level)":     float(np.std(finite)),
        "p1":                    float(p1),
        "p99":                   float(p99),
        "median |Δ|":            float(np.median(abs_err)),
        "mean |Δ| (MAE)":        float(np.mean(abs_err)),
        "max |Δ|":               float(abs_err.max()),
        "# pixels |Δ|>3σ":       int((abs_err > 3 * np.std(finite)).sum()),
        "RMSE":                  float(np.sqrt(np.mean(finite ** 2))),
        "Median e⁻ equiv":       float(np.sinh(p50) * Config.STRETCH_SCALE_E),
    }


def _noise_floor_vis_e() -> float:
    """Expected per-VIS-pixel noise std (e⁻) from the analytical noise model.

    Used as the denominator floor in relative-error panels — keeps the plot
    finite where signal → 0.
    """
    t_total = Config.EXPOSURE_TIME_S * Config.N_EXPOSURES
    pixel_area = Config.VIS_PIXEL_SCALE_ARCSEC ** 2
    sky_e   = Config.SKY_E_PER_S_PER_ARCSEC2 * pixel_area * t_total
    dark_e  = Config.DARK_E_PER_S_PER_PIX * t_total
    read_var = Config.READ_NOISE_E ** 2 * Config.N_EXPOSURES
    return float(np.sqrt(sky_e + dark_e + read_var))


def _residual_metrics(residual_stretched: np.ndarray,
                      residual_e: np.ndarray,
                      hr_stretched: np.ndarray,
                      hr_e: np.ndarray) -> dict:
    """PSNR / RMSE / MAE numbers for the reconstruction in both spaces.

    Peaks come from ``Config.PSNR_PEAK_*`` (mag-17 star).
    """
    eps = 1e-7
    rmse_str = float(np.sqrt(np.mean(residual_stretched ** 2)) + eps)
    rmse_raw = float(np.sqrt(np.mean(residual_e ** 2)) + eps)

    peak_str = float(Config.PSNR_PEAK_STRETCHED)
    peak_raw = float(Config.PSNR_PEAK_E)
    psnr_str = 20.0 * np.log10(peak_str / rmse_str)
    psnr_raw = 20.0 * np.log10(peak_raw / rmse_raw)

    return {
        "PSNR str (peak=mag17)":   f"{psnr_str:>9.3f} dB",
        "PSNR raw (peak=mag17)":   f"{psnr_raw:>9.3f} dB",
        "MAE (asinh)":             float(np.mean(np.abs(residual_stretched))),
        "MAE (raw e⁻)":           float(np.mean(np.abs(residual_e))),
        "RMSE (asinh)":            float(rmse_str),
        "RMSE (raw e⁻)":          float(rmse_raw),
        "Worst |Δ| (asinh)":       float(np.max(np.abs(residual_stretched))),
        "Worst |Δ| (raw e⁻)":      float(np.max(np.abs(residual_e))),
        "Best |Δ| (asinh)":        float(np.min(np.abs(residual_stretched))),
    }


def plot_reconstruction(
    lr_data: np.ndarray,
    sr_data: np.ndarray,
    hr_data: np.ndarray | None = None,
    output_path: str = "reconstruction.png",
    vmax: float | None = None,
    lr_cube: np.ndarray | None = None,
    hr_cube: np.ndarray | None = None,
    asinh_scale: float | None = None,
    show_all_bands: bool = False,
    predicted_dirty: np.ndarray | None = None,
    residual: np.ndarray | None = None,
    rgb_mode: str = "eye",
    dirty_hi_pct: float | None = None,
) -> None:
    """
    Visualize LR input, SR output, and (optionally) HR ground truth.

    Layout when HR is provided — 3 rows × 5 cols (4 rows when colour
    composites are added):

        Row 0 (color, optional):    LR color | (blank) | HR color | ...
        Row 1 (raw / linear):       LR raw | SR raw | HR raw | residual raw | rel-err raw
        Row 2 (asinh):              LR asinh | SR asinh | HR asinh | residual asinh | rel-err asinh
        Row 3 (stats):              LR stats | SR stats | HR stats | asinh-residual | PSNR

    All asinh panels share ``Config.STRETCH_SCALE_E`` (the network's
    training scale), so brightness is directly comparable to the loss.

    When HR is missing, falls back to a 2 × 3 LR/SR layout (raw + asinh
    + optional LR color).

    Color composites: SR and HR render in the regime picked by
    ``rgb_mode``:

      * ``"eye"`` (default) — the PHYSICAL mode (per-pixel blackbody
        color temperature → CIE Planckian-locus hue, absolute luminance
        keyed to the asinh knee — see
        :func:`euclid_polish.visualization.color.eye_rgb`). The
        transform is image-independent, so an SR-vs-HR hue difference is
        a reconstruction error, never a rendering artifact; a
        blackbody-T legend strip under the asinh HR (or SR) panel
        translates hue back to SED temperature.
      * ``"calibrated"`` — the solar-balanced adaptive mode (per-image
        [p1, p99.5] windows), the depth-adaptive rendering the /sky
        record viewer's "solar" chip uses.

    Inference jobs render the SAME figure once per regime so both
    color readings sit side-by-side in the gallery. The residual /
    metric panels stay on the VIS channel (channel 0) regardless, so
    the numbers remain comparable with historical single-band runs.
    """
    if rgb_mode not in ("eye", "calibrated"):
        raise ValueError(
            f"rgb_mode must be 'eye' or 'calibrated'; got {rgb_mode!r}"
        )
    # Asinh-stretch knee used in every "asinh" panel of this plot. The
    # caller can override per-run from the UI (especially useful on
    # real Euclid cutouts where the source-to-sky brightness ratio is
    # different from the simulator's). ``None`` → fall back to the
    # training scale ``Config.STRETCH_SCALE_E`` so default behaviour is
    # unchanged for everyone who doesn't pass the parameter.
    shared_scale = float(asinh_scale) if asinh_scale and asinh_scale > 0 \
                   else float(Config.STRETCH_SCALE_E)

    # 4-band SR (the VIS+NISP model): keep the cube for color panels and
    # use the VIS plane (channel 0) for every residual / metric panel.
    _nbands = len(Config.LR_INPUT_BAND_NAMES)
    sr_cube = (np.asarray(sr_data)
               if (np.asarray(sr_data).ndim == 3
                   and np.asarray(sr_data).shape[-1] == _nbands)
               else None)
    sr_vis = sr_cube[..., 0] if sr_cube is not None else np.asarray(sr_data)

    # Regime tag in the figure title so the eye / solar renders of the
    # same scene are distinguishable in the gallery.
    regime_note = ("" if sr_cube is None else
                   (" — eye color" if rgb_mode == "eye" else " — solar color"))

    def _add_sr_panel(vis_obj, stretch, temp_legend=False):
        """SR panel — color in the requested ``rgb_mode`` when the 4-band
        cube exists, else grayscale. The blackbody-T legend only applies
        to the eye regime."""
        if sr_cube is not None:
            vis_obj.add_rgb_scale_panel(
                sr_cube, stretch=stretch, asinh_scale=shared_scale,
                title_suffix="\nReconstruction (SR)",
                rgb_mode=rgb_mode,
                temp_legend=temp_legend and rgb_mode == "eye")
        else:
            kw = {"asinh_scale": shared_scale} if stretch == "asinh" else {}
            vis_obj.add_scale_panel(sr_vis, stretch=stretch,
                                    title_suffix="\nReconstruction (SR)",
                                    cmap="gray", **kw)

    if hr_data is None:
        # Real-Euclid inference flow: no HR truth available. Two layout
        # modes:
        #
        #   (a) ``show_all_bands=True`` (and a 4-band LR cube is
        #       supplied) → 2 × 5 grid showing each LR band as its own
        #       grayscale panel + SR, both linear and asinh. Useful for
        #       diagnosing per-band issues (saturation, NISP CR
        #       artefacts, persistence) the colour composite would
        #       blend together.
        #
        #   (b) default → 2 × 2 LR (colour or VIS gray) + SR (gray).
        nbands = len(Config.LR_INPUT_BAND_NAMES)
        has_cube = (lr_cube is not None and lr_cube.ndim == 3
                    and lr_cube.shape[-1] == nbands)

        if show_all_bands and has_cube:
            # Third row for the per-band SR planes when the model emits
            # all four bands.
            nrows = 3 if sr_cube is not None else 2
            vis = BaseVisualizer(rows=nrows, cols=nbands + 1,
                                 figsize=(6 * (nbands + 1), 6 * nrows),
                                 vmax=vmax)
            # Row 1 — linear per-band LR + SR (color when 4-band).
            for k, name in enumerate(Config.LR_INPUT_BAND_NAMES):
                vis.add_scale_panel(
                    lr_cube[..., k], stretch="linear",
                    title_suffix=f"\nDirty (LR · {name})",
                    cmap="gray",
                )
            _add_sr_panel(vis, "linear")
            # Row 2 — asinh per-band LR + SR (color when 4-band).
            for k, name in enumerate(Config.LR_INPUT_BAND_NAMES):
                vis.add_scale_panel(
                    lr_cube[..., k], stretch="asinh",
                    asinh_scale=shared_scale,
                    title_suffix=f"\nDirty (LR · {name})",
                    cmap="gray",
                )
            _add_sr_panel(vis, "asinh", temp_legend=True)
            # Row 3 — per-band SR planes (asinh), one per output band.
            if sr_cube is not None:
                for k, name in enumerate(Config.LR_INPUT_BAND_NAMES):
                    vis.add_scale_panel(
                        sr_cube[..., k], stretch="asinh",
                        asinh_scale=shared_scale,
                        title_suffix=f"\nSR · {name}",
                        cmap="gray",
                    )
            plt.suptitle("Super-Resolution Reconstruction (per-band view)"
                         + regime_note, fontsize=16)
            vis.save_figure(output_path)
            return

        # Default 2 × 2 layout: Dirty LR | SR, in linear and asinh rows.
        # Dirty LR is always rendered in VIS-only grayscale (a 4-band colour
        # composite is dominated by the much-noisier NISP channels and
        # visually underplays the VIS plane the model targets); SR renders in
        # the chosen colour regime. The LR panel uses an inset colorbar so it
        # does NOT shrink relative to the colorbar-less SR panel — LR and SR
        # then sit the same on-screen size, which is what makes them pleasant
        # to compare side by side.
        #
        # There is intentionally no "predicted LR" / forward-model residual
        # column for real cutouts: those would forward-model SR through *our*
        # committed VIS PSF, but the true Euclid PSF is position-dependent and
        # unknown at an arbitrary (RA, Dec), so the residual measures the PSF
        # mismatch rather than the reconstruction. (predicted_dirty / residual
        # are still accepted for the synthetic / known-PSF callers.)
        has_predicted = predicted_dirty is not None
        has_residual  = residual is not None
        cols = 2 + int(has_predicted) + int(has_residual)
        vis = BaseVisualizer(rows=2, cols=cols, figsize=(9 * cols, 18),
                             vmax=vmax)

        # Row 1 — linear. ``dirty_hi_pct`` lets the caller raise the upper clip
        # so a bright central galaxy stops saturating and its internal (lens)
        # structure shows.
        vis.add_scale_panel(lr_data, stretch="linear",
                            title_suffix="\nDirty (LR, VIS)",
                            cmap="gray", colorbar_inset=True,
                            hi_percentile=dirty_hi_pct)
        _add_sr_panel(vis, "linear")
        if has_predicted:
            vis.add_scale_panel(
                predicted_dirty, stretch="linear",
                title_suffix="\nVIS PSF ⨂ SR + 2× rebin\n(predicted LR)",
                cmap="gray",
            )
        if has_residual:
            vis.add_diverging_panel(
                residual, stretch="linear",
                title_suffix="\nResidual = Dirty − Predicted",
                colorbar_label="LR e⁻",
            )

        # Row 2 — asinh (loss-aligned stretch).
        vis.add_scale_panel(lr_data, stretch="asinh",
                            asinh_scale=shared_scale,
                            title_suffix="\nDirty (LR, VIS)",
                            cmap="gray", colorbar_inset=True,
                            hi_percentile=dirty_hi_pct)
        _add_sr_panel(vis, "asinh", temp_legend=True)
        if has_predicted:
            vis.add_scale_panel(
                predicted_dirty, stretch="asinh", asinh_scale=shared_scale,
                title_suffix="\nVIS PSF ⨂ SR + 2× rebin\n(predicted LR)",
                cmap="gray",
            )
        if has_residual:
            vis.add_diverging_panel(
                residual, stretch="asinh", asinh_scale=shared_scale,
                title_suffix="\nResidual = Dirty − Predicted",
            )
        plt.suptitle("Super-Resolution Reconstruction" + regime_note,
                     fontsize=16)
        vis.save_figure(output_path)
        return

    # Pre-compute residuals & noise floors (used as denominator clamps in
    # the rel-err panels so they don't blow up at sky-floor pixels).
    # Residual / metric panels compare the VIS planes (channel 0) so the
    # numbers stay comparable with historical single-band runs.
    hr_stretched       = np.arcsinh(hr_data / shared_scale).astype(np.float32)
    sr_stretched       = np.arcsinh(sr_vis / shared_scale).astype(np.float32)
    residual_e         = (hr_data - sr_vis).astype(np.float32)
    residual_stretched = (hr_stretched - sr_stretched).astype(np.float32)
    floor_e            = _noise_floor_vis_e()
    floor_str          = float(np.arcsinh(floor_e / shared_scale))

    # 3 × 5 layout. SR / HR cells render in colour when their 4-band
    # cubes are available — directly comparable composites since the
    # 4-band model emits every band.
    vis = BaseVisualizer(rows=3, cols=5, figsize=(45, 24), vmax=vmax)

    nbands = len(Config.LR_INPUT_BAND_NAMES)
    # LR (dirty) is always rendered VIS-only grayscale. NISP channels
    # are much noisier than VIS and a 4-band colour composite makes
    # the dirty image look worse than what the model actually sees —
    # the user wants to inspect the VIS channel the network targets.
    # HR (clean truth) keeps the 4-band colour composite since it's
    # noise-free and the colour informs galaxy-SED interpretation.
    hr_color = (hr_cube is not None and hr_cube.ndim == 3
                and hr_cube.shape[-1] == nbands)

    # ---- Row 1: linear (raw electrons) ----
    vis.add_scale_panel(lr_data, stretch="linear",
                        title_suffix="\nDirty (LR, VIS)",
                        cmap="gray")
    _add_sr_panel(vis, "linear")
    if hr_color:
        vis.add_rgb_scale_panel(hr_cube, stretch="linear",
                                title_suffix="\nTrue Sky (HR)",
                                rgb_mode=rgb_mode)
    else:
        vis.add_scale_panel(hr_data, stretch="linear",
                            title_suffix="\nTrue Sky (HR)")
    vis.add_diverging_panel(residual_e, stretch="linear",
                            title_suffix="\nResidual = HR − SR (raw e⁻)",
                            colorbar_label="Residual (e⁻)")
    vis.add_relative_error_panel(residual_e, hr_data, floor=floor_e,
                                 title_suffix="\nraw e⁻")

    # ---- Row 2: asinh (loss-aligned) ----
    vis.add_scale_panel(lr_data, stretch="asinh", asinh_scale=shared_scale,
                        title_suffix="\nDirty (LR, VIS)",
                        cmap="gray")
    _add_sr_panel(vis, "asinh")
    if hr_color:
        # Temperature legend on the asinh HR panel (eye regime only) —
        # one hue ↔ T_eff dictionary serves every eye panel.
        vis.add_rgb_scale_panel(hr_cube, stretch="asinh",
                                asinh_scale=shared_scale,
                                title_suffix="\nTrue Sky (HR)",
                                rgb_mode=rgb_mode,
                                temp_legend=(rgb_mode == "eye"))
    else:
        vis.add_scale_panel(hr_data, stretch="asinh", asinh_scale=shared_scale,
                            title_suffix="\nTrue Sky (HR)")
    vis.add_diverging_panel(residual_e, stretch="asinh", asinh_scale=shared_scale,
                            title_suffix="\nResidual = HR − SR")
    vis.add_relative_error_panel(residual_stretched, hr_stretched, floor=floor_str,
                                 title_suffix="\nasinh space")

    # ---- Row 3: stats ----
    vis.add_statistics_panel(lr_data, {
        "title": "LR (dirty input):",
        "stats": _image_stats(lr_data),
        "include_data_stats": False,
    })
    vis.add_statistics_panel(sr_vis, {
        "title": "SR (model output, VIS):",
        "stats": _image_stats(sr_vis),
        "include_data_stats": False,
    })
    vis.add_statistics_panel(hr_data, {
        "title": "HR (ground truth):",
        "stats": _image_stats(hr_data),
        "include_data_stats": False,
    })
    vis.add_statistics_panel(residual_stretched, {
        "title": "Residual (asinh space):",
        "stats": _residual_asinh_stats(residual_stretched, residual_e),
        "include_data_stats": False,
    })
    vis.add_statistics_panel(residual_e, {
        "title": "PSNR / error metrics:",
        "stats": _residual_metrics(residual_stretched, residual_e,
                                   hr_stretched, hr_data),
        "include_data_stats": False,
    })

    plt.suptitle("Super-Resolution Reconstruction" + regime_note,
                 fontsize=16)
    vis.save_figure(output_path)


def _first(images: ImageSet, role: Role) -> Image | None:
    for im in images:
        if im.role is role:
            return im
    return None


def plot_imageset(images: ImageSet, output_path: str, *, regime: str = "eye",
                  asinh_scale: float | None = None) -> str:
    """Render the LR→SR(→HR) reconstruction figure for an :class:`ImageSet`.

        plot_imageset(ImageSet.from_images([lr, sr, hr]), "recon.png")

    Picks the SR image (``role='sr'``, required), the LR input (``role='lr'`` or
    ``'real'``, optional — a 2× pooled SR-VIS proxy is used when absent) and the
    HR truth (``role='hr'``, optional) from the set, then delegates to
    :func:`plot_reconstruction`. ``regime`` is the colour regime (``"eye"`` or
    ``"calibrated"``). Returns ``output_path``.
    """
    sr = _first(images, Role.SR)
    if sr is None:
        raise ValueError("plot_imageset needs an image with role='sr'")
    lr = _first(images, Role.LR) or _first(images, Role.REAL)
    hr = _first(images, Role.HR)

    sr_data = np.asarray(sr.data, dtype=np.float32)

    lr_data = lr_cube = None
    if lr is not None:
        a = np.asarray(lr.data, dtype=np.float32)
        if a.ndim == 3 and a.shape[-1] > 1:
            lr_cube, lr_data = a, a[..., 0]
        else:
            lr_data = a[..., 0] if a.ndim == 3 else a
    if lr_data is None:
        vis = sr_data[..., 0] if sr_data.ndim == 3 else sr_data
        h, w = vis.shape[:2]
        lr_data = vis[: h - h % 2, : w - w % 2].reshape(
            h // 2, 2, w // 2, 2).mean(axis=(1, 3))

    hr_data = hr_cube = None
    if hr is not None:
        a = np.asarray(hr.data, dtype=np.float32)
        if a.ndim == 3 and a.shape[-1] > 1:
            hr_cube, hr_data = a, a[..., 0]
        else:
            hr_data = a[..., 0] if a.ndim == 3 else a

    plot_reconstruction(
        lr_data=lr_data, sr_data=sr_data, hr_data=hr_data,
        output_path=output_path, lr_cube=lr_cube, hr_cube=hr_cube,
        asinh_scale=asinh_scale, rgb_mode=regime)
    return output_path
