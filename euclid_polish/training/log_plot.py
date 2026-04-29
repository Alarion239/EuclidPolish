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
    """Plot loss + PSNR (and gradient norm if logged) vs step.

    Layout:
      - Top: loss (left axis, log scale) + PSNR stretched/raw (right axis).
      - Bottom (only if log contains ``gnorm_avg``): gradient-norm trace
        with the configured ``clip_norm`` shown as a horizontal reference.

    Parameters
    ----------
    log_path : str
        Path to the JSONL log (typically ``<ckpt_dir>/training_log.jsonl``).
    output_path : str
        Where to save the PNG.
    smooth_window : int, optional
        If > 1, overlay a moving-average curve on top of the raw loss / PSNR.

    Returns
    -------
    (n_records, last_step) : tuple of ints
    """
    records = read_training_log(log_path)
    steps  = np.array([r["step"]  for r in records])
    losses = np.array([r["loss"]  for r in records])
    psnr_str = np.array([r["psnr_stretched"] for r in records])
    psnr_raw = np.array([r["psnr_raw"]       for r in records])

    has_gnorm = all("gnorm_avg" in r for r in records)
    if has_gnorm:
        gnorm_avg = np.array([r["gnorm_avg"] for r in records])
        gnorm_max = np.array([r["gnorm_max"] for r in records])
        clip_norm = float(records[-1].get("clip_norm", 0.0))

    if has_gnorm:
        fig, (ax_loss, ax_g) = plt.subplots(
            2, 1, figsize=(11, 8), sharex=True,
            gridspec_kw=dict(height_ratios=[3, 1], hspace=0.08),
        )
    else:
        fig, ax_loss = plt.subplots(figsize=(11, 6))
        ax_g = None

    color_loss = "tab:blue"
    ax_loss.plot(steps, losses, color=color_loss, alpha=0.85, lw=1.4, label="loss (MAE)")
    ax_loss.set_ylabel("Loss", color=color_loss)
    ax_loss.tick_params(axis="y", labelcolor=color_loss)
    if (losses > 0).all():
        ax_loss.set_yscale("log")

    ax_psnr = ax_loss.twinx()
    color_psnr_str = "tab:red"
    color_psnr_raw = "tab:orange"

    ax_psnr.plot(steps, psnr_str, color=color_psnr_str, alpha=0.85, lw=1.4,
                 label="PSNR stretched")
    ax_psnr.plot(steps, psnr_raw, color=color_psnr_raw, alpha=0.85, lw=1.4,
                 ls="--", label="PSNR raw e⁻")
    ax_psnr.set_ylabel("PSNR (dB)", color="black")
    ax_psnr.tick_params(axis="y", labelcolor="black")

    if smooth_window and smooth_window > 1 and len(steps) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        loss_s    = np.convolve(losses,   kernel, mode="valid")
        psnr_s_s  = np.convolve(psnr_str, kernel, mode="valid")
        psnr_r_s  = np.convolve(psnr_raw, kernel, mode="valid")
        steps_s   = steps[smooth_window - 1:]
        ax_loss.plot(steps_s, loss_s, color=color_loss, lw=2.4, label=f"loss (MA{smooth_window})")
        ax_psnr.plot(steps_s, psnr_s_s, color=color_psnr_str, lw=2.4,
                     label=f"PSNR str (MA{smooth_window})")
        ax_psnr.plot(steps_s, psnr_r_s, color=color_psnr_raw, lw=2.4, ls="--",
                     label=f"PSNR raw (MA{smooth_window})")

    h1, l1 = ax_loss.get_legend_handles_labels()
    h2, l2 = ax_psnr.get_legend_handles_labels()
    ax_loss.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.9)

    if ax_g is not None:
        ax_g.plot(steps, gnorm_avg, color="tab:gray",   lw=1.4, label="‖g‖ avg")
        ax_g.plot(steps, gnorm_max, color="tab:purple", lw=0.8, alpha=0.7, label="‖g‖ max")
        if clip_norm > 0 and np.isfinite(clip_norm):
            ax_g.axhline(clip_norm, color="red", ls="--", lw=0.8,
                         label=f"clip_norm={clip_norm:g}")
        ax_g.set_ylabel("|g|")
        ax_g.set_xlabel("Step")
        if (gnorm_avg > 0).all():
            ax_g.set_yscale("log")
        ax_g.legend(loc="upper right", fontsize=8, framealpha=0.9)
    else:
        ax_loss.set_xlabel("Step")

    fig.suptitle(
        f"Training log — {len(records)} evaluations, last step {int(steps[-1])}\n"
        f"({log_path})",
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return len(records), int(steps[-1])


def default_log_path(checkpoint_dir: str) -> str:
    """Convention for where the trainer writes its log."""
    return os.path.join(checkpoint_dir, TRAINING_LOG_FILENAME)
