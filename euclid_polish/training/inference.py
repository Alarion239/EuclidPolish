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
    nchan: int = 1,
):
    """
    Build a WDSR model and restore weights from a TF checkpoint directory.

    Parameters
    ----------
    checkpoint_dir : str
        Directory managed by tf.train.CheckpointManager.
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
    from euclid_polish.training.trainer import Trainer
    from tf_keras.losses import MeanAbsoluteError

    model = wdsr(scale=scale, num_res_blocks=num_res_blocks, nchan=nchan)
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

    if lr_data.ndim == 3 and lr_data.shape[-1] == 1:
        lr_data = lr_data[..., 0]

    scale_e = float(Config.STRETCH_SCALE_E)

    # Stretch → model → unstretch. ``clip(±20)`` is a defensive guard against
    # an untrained / pathological model output: ``sinh(20) ≈ 2.4×10⁸``, well
    # above any realistic source, but finite in float32 (sinh overflows at ~89).
    lr_stretched = np.arcsinh(lr_data / scale_e).astype(np.float32)
    lr_3d = tf.constant(lr_stretched[:, :, None])
    sr_stretched = resolve_single(model, lr_3d).numpy().astype(np.float32)
    sr_stretched = np.clip(sr_stretched, -20.0, 20.0)
    sr_data = (np.sinh(sr_stretched.astype(np.float64)) * scale_e).astype(np.float32)
    sr_data = sr_data[..., 0] if sr_data.ndim == 3 else sr_data

    return lr_data, sr_data


def _image_stats(data: np.ndarray) -> dict:
    """Compact dict of useful per-image statistics."""
    finite = data[np.isfinite(data)]
    p1, p10, p50, p90, p99, p999 = np.percentile(finite, [1, 10, 50, 90, 99, 99.9])
    pos = finite[finite > 0]
    return {
        "Shape":         f"{data.shape[0]} × {data.shape[1]}",
        "Dtype":         str(data.dtype),
        "min":           f"{finite.min():>+12.4g}",
        "p1":            f"{p1:>+12.4g}",
        "p10":           f"{p10:>+12.4g}",
        "median":        f"{p50:>+12.4g}",
        "p90":           f"{p90:>+12.4g}",
        "p99":           f"{p99:>+12.4g}",
        "p99.9":         f"{p999:>+12.4g}",
        "max":           f"{finite.max():>+12.4g}",
        "mean":          f"{np.mean(finite):>+12.4g}",
        "std":           f"{np.std(finite):>12.4g}",
        "% < 0":         f"{(data < 0).mean() * 100:>11.2f}%",
        "Total flux":    f"{np.sum(finite):>12.4g}",
        "# bright (>p99)": f"{(finite > p99).sum():>12d}",
    }


def _residual_asinh_stats(residual_stretched: np.ndarray, residual_e: np.ndarray) -> dict:
    """Stats describing the residual structure in asinh space."""
    finite = residual_stretched[np.isfinite(residual_stretched)]
    p1, p50, p99 = np.percentile(finite, [1, 50, 99])
    abs_err = np.abs(finite)
    return {
        "mean (bias)":           f"{np.mean(finite):>+12.4g}",
        "median":                f"{p50:>+12.4g}",
        "std (noise level)":     f"{np.std(finite):>12.4g}",
        "p1, p99":               f"[{p1:+.3g}, {p99:+.3g}]",
        "median |Δ|":            f"{np.median(abs_err):>12.4g}",
        "mean |Δ| (MAE)":        f"{np.mean(abs_err):>12.4g}",
        "max |Δ|":               f"{abs_err.max():>12.4g}",
        "# pixels |Δ|>3σ":       f"{(abs_err > 3 * np.std(finite)).sum():>12d}",
        "RMSE":                  f"{np.sqrt(np.mean(finite ** 2)):>12.4g}",
        "Median residual electron-equiv": f"{float(np.sinh(p50) * Config.STRETCH_SCALE_E):>+12.4g}",
    }


def _noise_floor_lr_var() -> float:
    """Expected per-LR-pixel noise variance (e⁻²) from the noise model."""
    t_total = Config.EXPOSURE_TIME_S * Config.N_EXPOSURES
    pixel_area = Config.VIS_PIXEL_SCALE_ARCSEC ** 2
    sky_e   = Config.SKY_E_PER_S_PER_ARCSEC2 * pixel_area * t_total
    dark_e  = Config.DARK_E_PER_S_PER_PIX * t_total
    read_var = Config.READ_NOISE_E ** 2 * Config.N_EXPOSURES
    return float(sky_e + dark_e + read_var)


def _expected_sigma_e(hr: np.ndarray) -> np.ndarray:
    """Per-pixel expected noise std σ in raw electrons given HR signal.

    σ²_i = max(HR_i, 0) (Poisson signal contribution)
         + σ²_floor / r²  (per-HR-pixel constant noise floor)

    where σ²_floor = sky + dark + N·read² is the LR-pixel variance and the
    factor r² = REBIN² accounts for the fact that HR pixels see 1/r² of
    the area-integrated noise that LR pixels do.
    """
    floor_var_lr = _noise_floor_lr_var()
    rebin = float(Config.DEFAULT_REBIN_FACTOR)
    floor_var_hr = floor_var_lr / (rebin ** 2)
    return np.sqrt(np.maximum(hr, 0.0) + floor_var_hr).astype(np.float32)


def _expected_sigma_stretched(hr: np.ndarray, hr_stretched: np.ndarray) -> np.ndarray:
    """σ in asinh-stretched space, propagated from raw σ via the asinh
    derivative: ``d asinh(x/k)/dx = 1 / sqrt(k² + x²)``.
    """
    sigma_e = _expected_sigma_e(hr)
    k = float(Config.STRETCH_SCALE_E)
    jacobian = 1.0 / np.sqrt(k * k + hr * hr).astype(np.float32)
    return (sigma_e * jacobian).astype(np.float32)


def _residual_metrics(residual_stretched: np.ndarray,
                      residual_e: np.ndarray,
                      hr_stretched: np.ndarray,
                      hr_e: np.ndarray) -> dict:
    """SNR / RMSE / MAE numbers for the reconstruction in both spaces."""
    eps = 1e-7
    rmse_str = float(np.sqrt(np.mean(residual_stretched ** 2)) + eps)
    rmse_raw = float(np.sqrt(np.mean(residual_e ** 2)) + eps)

    var_hr_str = float(np.var(hr_stretched))
    var_hr_raw = float(np.var(hr_e))
    snr_var_str = 10.0 * np.log10(var_hr_str / (np.var(residual_stretched) + eps))
    snr_var_raw = 10.0 * np.log10(var_hr_raw / (np.var(residual_e) + eps))

    return {
        "SNR_var (asinh, loss)":   f"{snr_var_str:>+10.3f} dB",
        "SNR_var (raw e⁻)":       f"{snr_var_raw:>+10.3f} dB",
        "MAE (asinh)":             f"{np.mean(np.abs(residual_stretched)):>12.4g}",
        "MAE (raw e⁻)":           f"{np.mean(np.abs(residual_e)):>12.4g}",
        "RMSE (asinh)":            f"{rmse_str:>12.4g}",
        "RMSE (raw e⁻)":          f"{rmse_raw:>12.4g}",
        "Worst pixel |Δ| (asinh)": f"{np.max(np.abs(residual_stretched)):>12.4g}",
        "Worst pixel |Δ| (raw e⁻)": f"{np.max(np.abs(residual_e)):>12.4g}",
        "Best pixel |Δ| (asinh)":  f"{np.min(np.abs(residual_stretched)):>12.4g}",
    }


def plot_reconstruction(
    lr_data: np.ndarray,
    sr_data: np.ndarray,
    hr_data: Optional[np.ndarray] = None,
    output_path: str = "reconstruction.png",
    vmax: float | None = None,
) -> None:
    """
    Visualize LR input, SR output, and (optionally) HR ground truth.

    Layout when HR is provided — 3 rows × 5 cols:

        Row 1 (raw / linear):
            LR raw | SR raw | HR raw | residual raw | z-score raw
        Row 2 (asinh):
            LR asinh | SR asinh | HR asinh | residual asinh | z-score asinh
        Row 3 (stats):
            LR stats | SR stats | HR stats | asinh-residual stats | SNR stats

    All asinh panels share ``Config.STRETCH_SCALE_E`` (the network's
    training scale), so brightness is directly comparable to the loss.

    When HR is missing, falls back to a 2 × 2 LR/SR layout (raw + asinh).
    """
    shared_scale = float(Config.STRETCH_SCALE_E)

    if hr_data is None:
        vis = BaseVisualizer(rows=2, cols=2, figsize=(22, 18), vmax=vmax)
        vis.add_scale_panel(lr_data, stretch="linear", title_suffix="\nDirty (LR)")
        vis.add_scale_panel(sr_data, stretch="linear", title_suffix="\nReconstruction (SR)")
        vis.add_scale_panel(lr_data, stretch="asinh", asinh_scale=shared_scale,
                            title_suffix="\nDirty (LR)")
        vis.add_scale_panel(sr_data, stretch="asinh", asinh_scale=shared_scale,
                            title_suffix="\nReconstruction (SR)")
        plt.suptitle("Super-Resolution Reconstruction", fontsize=16)
        vis.save_figure(output_path)
        return

    # Pre-compute residuals & expected per-pixel noise
    hr_stretched       = np.arcsinh(hr_data / shared_scale).astype(np.float32)
    sr_stretched       = np.arcsinh(sr_data / shared_scale).astype(np.float32)
    residual_e         = (hr_data - sr_data).astype(np.float32)
    residual_stretched = (hr_stretched - sr_stretched).astype(np.float32)
    sigma_e            = _expected_sigma_e(hr_data)
    sigma_str          = _expected_sigma_stretched(hr_data, hr_stretched)

    vis = BaseVisualizer(rows=3, cols=5, figsize=(45, 24), vmax=vmax)

    # ---- Row 1: linear (raw electrons) ----
    vis.add_scale_panel(lr_data, stretch="linear", title_suffix="\nDirty (LR)")
    vis.add_scale_panel(sr_data, stretch="linear", title_suffix="\nReconstruction (SR)")
    vis.add_scale_panel(hr_data, stretch="linear", title_suffix="\nTrue Sky (HR)")
    vis.add_diverging_panel(residual_e, stretch="linear",
                            title_suffix="\nResidual = HR − SR (raw e⁻)",
                            colorbar_label="Residual (e⁻)")
    vis.add_zscore_panel(residual_e, sigma_e,
                         title_suffix="\nraw e⁻, σ from noise model")

    # ---- Row 2: asinh (loss-aligned) ----
    vis.add_scale_panel(lr_data, stretch="asinh", asinh_scale=shared_scale,
                        title_suffix="\nDirty (LR)")
    vis.add_scale_panel(sr_data, stretch="asinh", asinh_scale=shared_scale,
                        title_suffix="\nReconstruction (SR)")
    vis.add_scale_panel(hr_data, stretch="asinh", asinh_scale=shared_scale,
                        title_suffix="\nTrue Sky (HR)")
    vis.add_diverging_panel(residual_e, stretch="asinh", asinh_scale=shared_scale,
                            title_suffix="\nResidual = HR − SR")
    vis.add_zscore_panel(residual_stretched, sigma_str,
                         title_suffix="\nasinh space, σ propagated")

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
        "title": "SNR / error metrics:",
        "stats": _residual_metrics(residual_stretched, residual_e,
                                   hr_stretched, hr_data),
        "include_data_stats": False,
    })

    plt.suptitle("Super-Resolution Reconstruction", fontsize=16)
    vis.save_figure(output_path)
