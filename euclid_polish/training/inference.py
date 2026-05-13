"""
Inference / reconstruction module for EuclidPolish.

Provides utilities for applying a trained WDSR model to low-resolution
images and visualizing the results.
"""

import os
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.training.models.common import resolve_single
from euclid_polish.training.models.wdsr import wdsr
from euclid_polish.visualization.base import BaseVisualizer


def load_model_from_checkpoint(
    checkpoint_dir: str,
    scale: int,
    num_res_blocks: int = Config.DEFAULT_NUM_RES_BLOCKS,
    nchan: int = None,           # back-compat alias for nchan_in
    nchan_in: int = None,
    nchan_out: int = None,
):
    """Build a WDSR model and restore weights from a TF checkpoint directory.

    For the multi-band pipeline pass ``nchan_in=4, nchan_out=1``. The
    single-channel signature ``nchan=1`` is still accepted.
    """
    if nchan is not None and nchan_in is None:
        nchan_in = nchan
    if nchan_in is None:
        nchan_in = 1
    if nchan_out is None:
        nchan_out = nchan_in

    model = wdsr(
        scale=scale, num_res_blocks=num_res_blocks,
        nchan_in=nchan_in, nchan_out=nchan_out,
    )
    checkpoint = tf.train.Checkpoint(model=model)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    checkpoint.restore(latest).expect_partial()
    print(f"Model restored from checkpoint at {latest}.")
    return model


def load_model_from_weights(
    weights_path: str,
    scale: int,
    num_res_blocks: int = Config.DEFAULT_NUM_RES_BLOCKS,
    nchan: int = 1,
):
    """
    Build a WDSR model and load saved .h5 weights.

    Parameters
    ----------
    weights_path : str
        Path to a .h5 weights file.
    scale : int
        Super-resolution scale factor.
    num_res_blocks : int
        Number of residual blocks.
    nchan : int
        Number of image channels.

    Returns
    -------
    tf.keras.Model
    """
    model = wdsr(scale=scale, num_res_blocks=num_res_blocks, nchan=nchan)
    model.load_weights(weights_path)
    return model


def reconstruct(
    model,
    lr_input,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply super-resolution to a single LR image.

    Inputs and outputs are raw float32 electrons (over the stacked Euclid VIS
    integration). Internally this function

      1. asinh-stretches the input by Config.STRETCH_SCALE_E,
      2. runs the model (which operates entirely in stretched space),
      3. clips the stretched output for safety, and
      4. applies sinh × scale to recover electrons.

    Parameters
    ----------
    model : tf.keras.Model
        Trained WDSR model.
    lr_input : str or np.ndarray
        Either a ``.npy`` file path or a numpy array in raw electron units.

    Returns
    -------
    lr_data : ndarray, shape (H, W)
        The input LR image (2-D, raw electrons).
    sr_data : ndarray, shape (H', W')
        The super-resolved output (2-D, raw electrons).
    """
    if isinstance(lr_input, str):
        if lr_input.endswith(".npy"):
            lr_data = np.load(lr_input).astype(np.float32)
        else:
            raise ValueError(
                f"Unsupported file format: {lr_input}. "
                "Inputs must be .npy in raw electron units."
            )
    else:
        lr_data = np.asarray(lr_input, dtype=np.float32)

    # Normalize input to a (H, W, C) tensor for the model. Three shapes
    # we accept:
    #   2D (H, W)           — single-channel image
    #   3D (H, W, 1)        — single-channel with an explicit channel axis
    #   3D (H, W, C)        — multi-band cube (e.g. C=4 for VIS+Y_E+J_E+H_E)
    if lr_data.ndim == 2:
        lr_for_model = lr_data[:, :, None]              # → (H, W, 1)
    elif lr_data.ndim == 3:
        lr_for_model = lr_data                          # already (H, W, C)
    else:
        raise ValueError(
            f"reconstruct(): expected 2D or 3D input, got shape {lr_data.shape}"
        )

    scale_e = float(Config.STRETCH_SCALE_E)

    # Stretch → model → unstretch. ``clip(±20)`` is a defensive guard against
    # an untrained / pathological model output: ``sinh(20) ≈ 2.4×10⁸``, well
    # above any realistic source, but finite in float32 (sinh overflows at ~89).
    lr_stretched = np.arcsinh(lr_for_model / scale_e).astype(np.float32)
    sr_stretched = resolve_single(model, tf.constant(lr_stretched)).numpy().astype(np.float32)
    sr_stretched = np.clip(sr_stretched, -20.0, 20.0)
    sr_data = (np.sinh(sr_stretched.astype(np.float64)) * scale_e).astype(np.float32)
    # SR is always single-channel per the model spec (nchan_out=1); squeeze
    # the channel axis for the 2-D visualizers downstream.
    if sr_data.ndim == 3 and sr_data.shape[-1] == 1:
        sr_data = sr_data[..., 0]
    # LR returned for display: if the caller passed a multi-channel cube,
    # show only the VIS channel (band 0) — that's what the HR target is.
    if lr_data.ndim == 3 and lr_data.shape[-1] > 1:
        lr_display = lr_data[..., 0]
    elif lr_data.ndim == 3 and lr_data.shape[-1] == 1:
        lr_display = lr_data[..., 0]
    else:
        lr_display = lr_data

    return lr_display, sr_data


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


def _noise_floor_lr_std_e() -> float:
    """Expected per-LR-pixel noise std (e⁻) from the analytical noise model.

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


def _safe_lupton_rgb(cube: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Build a Lupton RGB from a 4-band cube, swallowing any rendering
    failure so the rest of the plot still renders. Returns None if the
    cube is missing or doesn't have all four bands."""
    if cube is None or cube.ndim != 3 or cube.shape[-1] != len(Config.LR_INPUT_BAND_NAMES):
        return None
    from euclid_polish.visualization.color import lupton_rgb
    try:
        return lupton_rgb(
            cube, band_names=Config.LR_INPUT_BAND_NAMES,
            scheme="vis_nisp", reference="solar", Q=8.0, stretch=1.0,
        )
    except Exception as e:    # pragma: no cover — color is best-effort
        print(f"  color composite skipped: {type(e).__name__}: {e}")
        return None


def plot_reconstruction(
    lr_data: np.ndarray,
    sr_data: np.ndarray,
    hr_data: Optional[np.ndarray] = None,
    output_path: str = "reconstruction.png",
    vmax: float | None = None,
    lr_cube: Optional[np.ndarray] = None,
    hr_cube: Optional[np.ndarray] = None,
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

    Color composites: pass ``lr_cube`` and/or ``hr_cube`` to render a
    Lupton RGB (solar-balanced) of the dirty LR and clean HR inputs.
    SR is *not* colorized — the model emits only VIS — so its slot in
    the color row stays empty as a deliberate "SR has no color info"
    visual cue.
    """
    shared_scale = float(Config.STRETCH_SCALE_E)

    lr_rgb = _safe_lupton_rgb(lr_cube)
    hr_rgb = _safe_lupton_rgb(hr_cube)

    if hr_data is None:
        # 2 × 3 layout when HR is unknown (real-Euclid inference flow):
        # row 1 = LR raw / SR raw / color LR ; row 2 = LR asinh / SR asinh / (blank).
        cols = 3 if lr_rgb is not None else 2
        vis = BaseVisualizer(rows=2, cols=cols, figsize=(8 * cols, 16), vmax=vmax)
        vis.add_scale_panel(lr_data, stretch="linear", title_suffix="\nDirty (LR)")
        vis.add_scale_panel(sr_data, stretch="linear", title_suffix="\nReconstruction (SR)")
        if lr_rgb is not None:
            vis.add_rgb_panel(lr_rgb, title_suffix="\nLR (4-band, solar-balanced)")
        vis.add_scale_panel(lr_data, stretch="asinh", asinh_scale=shared_scale,
                            title_suffix="\nDirty (LR)")
        vis.add_scale_panel(sr_data, stretch="asinh", asinh_scale=shared_scale,
                            title_suffix="\nReconstruction (SR)")
        # blank cell in the third column on row 2 is left empty if rgb exists.
        plt.suptitle("Super-Resolution Reconstruction", fontsize=16)
        vis.save_figure(output_path)
        return

    # Pre-compute residuals & noise floors (used as denominator clamps in
    # the rel-err panels so they don't blow up at sky-floor pixels).
    hr_stretched       = np.arcsinh(hr_data / shared_scale).astype(np.float32)
    sr_stretched       = np.arcsinh(sr_data / shared_scale).astype(np.float32)
    residual_e         = (hr_data - sr_data).astype(np.float32)
    residual_stretched = (hr_stretched - sr_stretched).astype(np.float32)
    floor_e            = _noise_floor_lr_std_e()
    floor_str          = float(np.arcsinh(floor_e / shared_scale))

    # If we have at least one color cube, add an extra row dedicated to
    # color composites. SR is intentionally left blank in this row — the
    # model only emits VIS, so it has no color information of its own.
    has_color_row = (lr_rgb is not None) or (hr_rgb is not None)
    rows = 4 if has_color_row else 3
    vis = BaseVisualizer(rows=rows, cols=5,
                         figsize=(45, 6 * rows), vmax=vmax)

    if has_color_row:
        # Row 0: LR color | (SR blank) | HR color | (blank) | (blank).
        if lr_rgb is not None:
            vis.add_rgb_panel(lr_rgb, title_suffix="\nDirty (LR), 4-band solar")
        else:
            vis._next_panel += 1   # skip the LR cell
        # SR slot is deliberately blank — the model emits VIS only.
        vis._next_panel += 1
        if hr_rgb is not None:
            vis.add_rgb_panel(hr_rgb, title_suffix="\nTrue Sky (HR), 4-band solar")
        else:
            vis._next_panel += 1
        # Fill remaining two cells in the row.
        vis._next_panel += 2

    # ---- Row 1: linear (raw electrons) ----
    vis.add_scale_panel(lr_data, stretch="linear", title_suffix="\nDirty (LR)")
    vis.add_scale_panel(sr_data, stretch="linear", title_suffix="\nReconstruction (SR)")
    vis.add_scale_panel(hr_data, stretch="linear", title_suffix="\nTrue Sky (HR)")
    vis.add_diverging_panel(residual_e, stretch="linear",
                            title_suffix="\nResidual = HR − SR (raw e⁻)",
                            colorbar_label="Residual (e⁻)")
    vis.add_relative_error_panel(residual_e, hr_data, floor=floor_e,
                                 title_suffix="\nraw e⁻")

    # ---- Row 2: asinh (loss-aligned) ----
    vis.add_scale_panel(lr_data, stretch="asinh", asinh_scale=shared_scale,
                        title_suffix="\nDirty (LR)")
    vis.add_scale_panel(sr_data, stretch="asinh", asinh_scale=shared_scale,
                        title_suffix="\nReconstruction (SR)")
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
    vis.add_statistics_panel(sr_data, {
        "title": "SR (model output):",
        "stats": _image_stats(sr_data),
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

    plt.suptitle("Super-Resolution Reconstruction", fontsize=16)
    vis.save_figure(output_path)
