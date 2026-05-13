"""Visualize the training-time validation log written by ``Trainer.train``."""

import csv
import json
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from euclid_polish.training.trainer import TRAINING_LOG_FILENAME


_NUMERIC_LOG_COLS = {
    "step", "wall_time", "loss",
    "psnr_stretched", "psnr_raw",
    "gnorm_avg", "gnorm_max", "clip_norm", "duration_s",
}


def read_training_log(log_path: str) -> List[dict]:
    """Read the trainer's validation-history log and return records as dicts.

    Auto-detects:
      * the current CSV format (header row + N rows of numeric values), and
      * the legacy ``training_log.jsonl`` format (one JSON object per line)
        so logs from runs prior to the CSV switch still plot.
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Training log not found: {log_path}")
    with open(log_path) as fh:
        first_meaningful = ""
        for ln in fh:
            if ln.strip():
                first_meaningful = ln
                break
        fh.seek(0)
        if first_meaningful.startswith("step,"):
            reader = csv.DictReader(fh)
            records = []
            for raw in reader:
                rec = {}
                for k, v in raw.items():
                    if v is None or v == "":
                        continue
                    if k in _NUMERIC_LOG_COLS:
                        try:
                            rec[k] = float(v)
                            if k == "step":
                                rec[k] = int(rec[k])
                        except (TypeError, ValueError):
                            rec[k] = v
                    else:
                        rec[k] = v
                records.append(rec)
        else:
            # Legacy JSONL — one JSON object per line; skip malformed.
            records = []
            for ln in fh:
                ln = ln.strip()
                if not ln or not ln.startswith("{"):
                    continue
                try:
                    records.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
    if not records:
        raise ValueError(f"Training log is empty: {log_path}")
    return records


def plot_training_records(
    records: List[dict],
    output_path: str,
    smooth_window: int = 0,
    title_suffix: str = "",
) -> Tuple[int, int]:
    """Same plot as :func:`plot_training_log`, but from a pre-loaded list
    of records. Used by the FASRC dashboard which fetches the log over
    SSH and filters by wall-time window."""
    if not records:
        raise ValueError("plot_training_records: records is empty")
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
        f"Training log — {len(records)} evaluations, last step {int(steps[-1])}"
        f"{title_suffix}",
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return len(records), int(steps[-1])


def plot_training_log(
    log_path: str,
    output_path: str,
    smooth_window: int = 0,
) -> Tuple[int, int]:
    """Read the trainer's validation log (CSV or legacy JSONL) and plot.

    Thin wrapper around :func:`plot_training_records` that reads the
    file first; kept for back-compat with existing callers.
    """
    records = read_training_log(log_path)
    return plot_training_records(
        records, output_path, smooth_window=smooth_window,
        title_suffix=f"\n({log_path})",
    )


def default_log_path(checkpoint_dir: str) -> str:
    """Convention for where the trainer writes its log."""
    return os.path.join(checkpoint_dir, TRAINING_LOG_FILENAME)
