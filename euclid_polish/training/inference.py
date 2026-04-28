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


def plot_reconstruction(
    lr_data: np.ndarray,
    sr_data: np.ndarray,
    hr_data: Optional[np.ndarray] = None,
    output_path: str = "reconstruction.png",
    vmax: float | None = None,
) -> None:
    """
    Visualize LR input, SR output, and (optionally) HR ground truth.

    Layout when HR is provided — 2 rows × 4 cols (more horizontal):

        Row 1: LR asinh | SR asinh | HR asinh   | residual asinh
        Row 2: residual raw (asinh stretch, divergent) |
               PSNR-per-pixel asinh | PSNR-per-pixel raw | stats

    All asinh panels share ``Config.STRETCH_SCALE_E`` (= the network's
    training scale), so the viz is directly comparable to the
    loss/PSNR metrics.

    When HR is missing, falls back to a simple 1 × 2 (LR | SR) asinh
    layout — diagnostic panels need a ground truth.

    Parameters
    ----------
    lr_data, sr_data, hr_data : ndarray
        2-D images in raw electrons. ``hr_data`` is optional.
    output_path : str
        PNG output path.
    vmax : float, optional
        Override the linear-scale upper bound (unused in the asinh-only
        layout but kept for API stability).
    """
    shared_scale = float(Config.STRETCH_SCALE_E)

    if hr_data is None:
        vis = BaseVisualizer(rows=1, cols=2, figsize=(22, 9), vmax=vmax)
        vis.add_scale_panel(lr_data, stretch="asinh", asinh_scale=shared_scale,
                            title_suffix="\nDirty (LR)")
        vis.add_scale_panel(sr_data, stretch="asinh", asinh_scale=shared_scale,
                            title_suffix="\nPOLISH Reconstruction (SR)")
        plt.suptitle("Super-Resolution Reconstruction", fontsize=16)
        vis.save_figure(output_path)
        return

    # Pre-compute residuals once
    residual_e         = (hr_data - sr_data).astype(np.float32)
    residual_stretched = (np.arcsinh(hr_data / shared_scale)
                          - np.arcsinh(sr_data / shared_scale)).astype(np.float32)

    vis = BaseVisualizer(rows=2, cols=4, figsize=(40, 18), vmax=vmax)

    # Row 1: LR / SR / HR / residual — all in asinh space
    vis.add_scale_panel(lr_data, stretch="asinh", asinh_scale=shared_scale,
                        title_suffix="\nDirty (LR)")
    vis.add_scale_panel(sr_data, stretch="asinh", asinh_scale=shared_scale,
                        title_suffix="\nPOLISH Reconstruction (SR)")
    vis.add_scale_panel(hr_data, stretch="asinh", asinh_scale=shared_scale,
                        title_suffix="\nTrue Sky (HR)")
    vis.add_diverging_panel(residual_e, asinh_scale=shared_scale,
                            title_suffix="\nResidual asinh = HR − SR")

    # Row 2: residual raw (divergent) | PSNR asinh | PSNR raw | stats
    vis.add_diverging_panel(residual_e, asinh_scale=shared_scale,
                            title_suffix="\nResidual raw e⁻")
    vis.add_pixel_psnr_panel(residual_stretched, max_val=10.0,
                             title_suffix="\nasinh space, max_val=10")
    vis.add_pixel_psnr_panel(residual_e,
                             max_val=float(Config.RAW_PSNR_MAX_VAL),
                             title_suffix=f"\nraw e⁻, max_val={Config.RAW_PSNR_MAX_VAL:.0e}",
                             clip_db=(0.0, 100.0))

    # Stats summary in the last cell — global PSNR and residual statistics.
    eps = 1e-7
    psnr_str_global = 20.0 * np.log10(10.0 / (np.std(residual_stretched) + eps))
    psnr_raw_global = 20.0 * np.log10(float(Config.RAW_PSNR_MAX_VAL)
                                      / (np.std(residual_e) + eps))
    stats = {
        "PSNR (asinh)":   f"{psnr_str_global:.2f} dB",
        "PSNR (raw e⁻)":  f"{psnr_raw_global:.2f} dB",
        "Residual mean (e⁻)":   f"{np.mean(residual_e):+.3g}",
        "Residual std (e⁻)":    f"{np.std(residual_e):.3g}",
        "Residual mean (asinh)": f"{np.mean(residual_stretched):+.3g}",
        "Residual std (asinh)":  f"{np.std(residual_stretched):.3g}",
    }
    vis.add_statistics_panel(residual_e, {
        "title": "Reconstruction stats:",
        "stats": stats,
        "include_data_stats": False,
    })

    plt.suptitle("Super-Resolution Reconstruction", fontsize=16)
    vis.save_figure(output_path)
