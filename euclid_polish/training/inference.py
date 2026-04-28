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
    integration). The model wraps tanh/arctanh as its first/last layers, so no
    on-disk normalization is required.

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

    lr_3d = tf.constant(lr_data[:, :, None])
    sr = resolve_single(model, lr_3d).numpy().astype(np.float32)
    sr_data = sr[..., 0] if sr.ndim == 3 else sr

    return lr_data, sr_data


def plot_reconstruction(
    lr_data: np.ndarray,
    sr_data: np.ndarray,
    hr_data: Optional[np.ndarray] = None,
    output_path: str = "reconstruction.png",
    vmax: float | None = None,
) -> None:
    """
    Visualize LR input, SR output, and optionally HR ground truth.

    Parameters
    ----------
    lr_data : ndarray
        Low-resolution input (2-D, electrons).
    sr_data : ndarray
        Super-resolved output (2-D, electrons).
    hr_data : ndarray, optional
        High-resolution ground truth (2-D, electrons).
    output_path : str
        Path to save the figure.
    vmax : float, optional
        Upper bound for the linear colour scale. ``None`` (default) lets
        matplotlib auto-scale per panel.
    """
    from euclid_polish.visualization.base import _asinh_scale

    # Use a shared asinh scale across all panels (LR's noise floor sets it)
    # so brightness is directly comparable across LR/SR/HR.
    shared_scale = _asinh_scale(lr_data)

    ncols = 3 if hr_data is not None else 2
    vis = BaseVisualizer(
        rows=2,
        cols=ncols,
        figsize=(11 * ncols, 20),
        vmax=vmax,
    )

    # Row 1: linear (percentile-clipped)
    vis.add_scale_panel(lr_data, stretch="linear", title_suffix="\nDirty (LR)")
    vis.add_scale_panel(sr_data, stretch="linear", title_suffix="\nPOLISH Reconstruction (SR)")
    if hr_data is not None:
        vis.add_scale_panel(hr_data, stretch="linear", title_suffix="\nTrue Sky (HR)")

    # Row 2: shared asinh
    vis.add_scale_panel(lr_data, stretch="asinh", asinh_scale=shared_scale, title_suffix="\nDirty (LR)")
    vis.add_scale_panel(sr_data, stretch="asinh", asinh_scale=shared_scale, title_suffix="\nPOLISH Reconstruction (SR)")
    if hr_data is not None:
        vis.add_scale_panel(hr_data, stretch="asinh", asinh_scale=shared_scale, title_suffix="\nTrue Sky (HR)")

    plt.suptitle("Super-Resolution Reconstruction", fontsize=16)
    vis.save_figure(output_path)
