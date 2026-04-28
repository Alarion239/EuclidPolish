"""Visualize the training-time JSONL log written by ``Trainer.train``."""

import json
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from euclid_polish.training.trainer import TRAINING_LOG_FILENAME


def read_training_log(log_path: str) -> List[dict]:
    """Read a JSONL training log and return the list of records."""
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Training log not found: {log_path}")
    records = []
    with open(log_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"Training log is empty: {log_path}")
    return records


def plot_training_log(
    log_path: str,
    output_path: str,
    smooth_window: int = 0,
) -> Tuple[int, int]:
    """Plot loss and PSNR vs step on a twin-axis figure.

    Parameters
    ----------
    log_path : str
        Path to the JSONL log (typically ``<ckpt_dir>/training_log.jsonl``).
    output_path : str
        Where to save the PNG.
    smooth_window : int, optional
        If > 1, draw an additional moving-average curve with this window.

    Returns
    -------
    (n_records, last_step) : tuple of ints
    """
    records = read_training_log(log_path)
    steps  = np.array([r["step"]  for r in records])
    losses = np.array([r["loss"]  for r in records])
    psnrs  = np.array([r["psnr"]  for r in records])

    fig, ax_loss = plt.subplots(figsize=(11, 6))

    color_loss = "tab:blue"
    ax_loss.plot(steps, losses, color=color_loss, alpha=0.85, lw=1.4, label="loss (MAE)")
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss", color=color_loss)
    ax_loss.tick_params(axis="y", labelcolor=color_loss)
    if (losses > 0).all():
        ax_loss.set_yscale("log")

    ax_psnr = ax_loss.twinx()
    color_psnr = "tab:red"
    ax_psnr.plot(steps, psnrs, color=color_psnr, alpha=0.85, lw=1.4, label="PSNR (dB)")
    ax_psnr.set_ylabel("PSNR (dB)", color=color_psnr)
    ax_psnr.tick_params(axis="y", labelcolor=color_psnr)

    if smooth_window and smooth_window > 1 and len(steps) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        loss_s = np.convolve(losses, kernel, mode="valid")
        psnr_s = np.convolve(psnrs,  kernel, mode="valid")
        steps_s = steps[smooth_window - 1:]
        ax_loss.plot(steps_s, loss_s, color=color_loss, lw=2.4, label=f"loss (MA{smooth_window})")
        ax_psnr.plot(steps_s, psnr_s, color=color_psnr, lw=2.4, label=f"PSNR (MA{smooth_window})")

    # Combined legend
    h1, l1 = ax_loss.get_legend_handles_labels()
    h2, l2 = ax_psnr.get_legend_handles_labels()
    ax_loss.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.9)

    plt.title(
        f"Training log — {len(records)} evaluations, last step {int(steps[-1])}\n"
        f"({log_path})",
        fontsize=11,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return len(records), int(steps[-1])


def default_log_path(checkpoint_dir: str) -> str:
    """Convention for where the trainer writes its log."""
    return os.path.join(checkpoint_dir, TRAINING_LOG_FILENAME)
