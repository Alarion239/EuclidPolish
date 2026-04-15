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

    The input must already be in [0, 65535] (the same range the model was
    trained on).  Normalized uint16 PNGs or the ``*_norm`` TFRecords
    satisfy this requirement directly.  No additional per-image
    normalization is applied here.

    Parameters
    ----------
    model : tf.keras.Model
        Trained WDSR model.
    lr_input : str or np.ndarray
        Either a file path (.npy or .png) or a numpy array in [0, 65535].

    Returns
    -------
    lr_data : ndarray, shape (H, W)
        The input LR image (2-D).
    sr_data : ndarray, shape (H', W')
        The super-resolved output (2-D, clipped to [0, 65535]).
    """
    # Load from file if needed
    if isinstance(lr_input, str):
        if lr_input.endswith(".npy"):
            lr_data = np.load(lr_input).astype(np.float32)
        elif lr_input.endswith(".png"):
            raw = tf.io.read_file(lr_input)
            lr_data = tf.image.decode_png(raw, dtype=tf.uint16).numpy().astype(np.float32)
            if lr_data.ndim == 3 and lr_data.shape[-1] == 1:
                lr_data = lr_data[..., 0]
        else:
            raise ValueError(f"Unsupported file format: {lr_input}")
    else:
        lr_data = np.asarray(lr_input, dtype=np.float32)

    # Ensure 2-D
    if lr_data.ndim == 3 and lr_data.shape[-1] == 1:
        lr_data = lr_data[..., 0]

    # Feed directly to model — input must already be in [0, 65535]
    lr_3d = tf.constant(lr_data[:, :, None])
    sr = resolve_single(model, lr_3d).numpy().astype(np.float32)
    sr_data = sr[..., 0] if sr.ndim == 3 else sr

    return lr_data, sr_data


def plot_reconstruction(
    lr_data: np.ndarray,
    sr_data: np.ndarray,
    hr_data: Optional[np.ndarray] = None,
    output_path: str = "reconstruction.png",
    clip_percentile: float = Config.DEFAULT_CLIP_PERCENTILE,
) -> None:
    """
    Visualize LR input, SR output, and optionally HR ground truth.

    Parameters
    ----------
    lr_data : ndarray
        Low-resolution input (2-D).
    sr_data : ndarray
        Super-resolved output (2-D).
    hr_data : ndarray, optional
        High-resolution ground truth (2-D).
    output_path : str
        Path to save the figure.
    clip_percentile : float
        Percentile for clipping in the linear scale panels.
    """
    ncols = 3 if hr_data is not None else 2
    vis = BaseVisualizer(
        clip_percentile=clip_percentile,
        rows=2,
        cols=ncols,
        figsize=(11 * ncols, 20),
    )

    # Row 1: linear scale
    vis.add_scale_panel(lr_data, title_suffix="\nDirty (LR)")
    vis.add_scale_panel(sr_data, title_suffix="\nPOLISH Reconstruction (SR)")
    if hr_data is not None:
        vis.add_scale_panel(hr_data, title_suffix="\nTrue Sky (HR)")

    # Row 2: log scale
    vis.add_scale_panel(lr_data, title_suffix="\nDirty (LR)", log_scale=True)
    vis.add_scale_panel(sr_data, title_suffix="\nPOLISH Reconstruction (SR)", log_scale=True)
    if hr_data is not None:
        vis.add_scale_panel(hr_data, title_suffix="\nTrue Sky (HR)", log_scale=True)

    plt.suptitle("Super-Resolution Reconstruction", fontsize=16)
    vis.save_figure(output_path)
