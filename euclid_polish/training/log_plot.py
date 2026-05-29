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
    # Multi-source validation columns (empty in synthetic-only / older
    # runs — read_training_log skips empty cells, so those rows simply
    # won't carry the key and the corresponding panel is omitted).
    "psnr_stretched_hst", "psnr_raw_hst", "roundtrip_val_psnr",
    "save_best_score",
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
    """Plot the validation metrics — all on ONE graph.

    Three curves share a single figure (from a pre-loaded record list;
    used by the FASRC dashboard which fetches the log over SSH and
    filters by wall-time window):

      * synthetic PSNR (stretched)   — dB
      * HST PSNR (stretched)         — dB (when logged)
      * round-trip ("cycle-run") PSNR — dB (when logged)

    All three are PSNR in dB and "higher is better", so they share one
    left axis. Synthetic-only / older runs without the HST / round-trip
    columns just show the one PSNR line.

    Returns ``(n_records, last_step)``.
    """
    if not records:
        raise ValueError("plot_training_records: records is empty")
    steps    = np.array([r["step"]            for r in records])
    psnr_syn = np.array([r["psnr_stretched"]  for r in records])

    def _opt_series(col: str) -> Tuple[np.ndarray, np.ndarray]:
        """(steps, values) for the rows that actually carry ``col``.

        Multi-source columns are blank on rows from synthetic-only runs
        (read_training_log drops empty cells → key absent) and ``None``
        from the SSH parser; both are filtered so each curve plots only
        the points genuinely measured.
        """
        xs: List[float] = []
        ys: List[float] = []
        for r in records:
            v = r.get(col)
            if v is None or v == "":
                continue
            try:
                ys.append(float(v))
                xs.append(int(r["step"]))
            except (TypeError, ValueError):
                continue
        return np.array(xs), np.array(ys)

    hst_x, hst_y     = _opt_series("psnr_stretched_hst")
    rt_x,  rt_y      = _opt_series("roundtrip_val_psnr")
    score_x, score_y = _opt_series("save_best_score")
    has_score = score_x.size > 0

    def _smoothed(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if smooth_window and smooth_window > 1 and y.size >= smooth_window:
            k = np.ones(smooth_window) / smooth_window
            return x[smooth_window - 1:], np.convolve(y, k, mode="valid")
        return None, None

    # Two stacked plots when the composite save-best score is logged:
    # the combined validation metrics on top, the score we actually
    # track for checkpointing on its own axes below. Older / score-less
    # runs keep the single metrics graph.
    if has_score:
        fig, (ax_psnr, ax_score) = plt.subplots(
            2, 1, figsize=(11, 9), sharex=True,
            gridspec_kw=dict(height_ratios=[3, 2], hspace=0.12),
        )
    else:
        fig, ax_psnr = plt.subplots(figsize=(11, 6))
        ax_score = None

    # ── Left axis: PSNR (dB), higher is better. ──
    ax_psnr.plot(steps, psnr_syn, color="tab:red", lw=1.6, alpha=0.9,
                 label="Synthetic PSNR")
    if hst_x.size:
        ax_psnr.plot(hst_x, hst_y, color="tab:green", lw=1.6, alpha=0.9,
                     label="HST PSNR")
    # Optional smoothed overlays.
    sx, sy = _smoothed(steps, psnr_syn)
    if sx is not None:
        ax_psnr.plot(sx, sy, color="tab:red", lw=2.6, label="Synthetic PSNR (MA)")
    if hst_x.size:
        sx, sy = _smoothed(hst_x, hst_y)
        if sx is not None:
            ax_psnr.plot(sx, sy, color="tab:green", lw=2.6, label="HST PSNR (MA)")
    ax_psnr.set_ylabel("PSNR (dB)  ·  higher better")

    # ── Round-trip PSNR shares the same dB axis — all three metrics are
    #    now PSNR (higher better), so no twin axis is needed. ──
    if rt_x.size:
        ax_psnr.plot(rt_x, rt_y, color="tab:purple", lw=1.6, ls="--",
                     alpha=0.9, label="Round-trip PSNR")
        sx, sy = _smoothed(rt_x, rt_y)
        if sx is not None:
            ax_psnr.plot(sx, sy, color="tab:purple", lw=2.6, ls="--",
                         label="Round-trip PSNR (MA)")

    ax_psnr.legend(loc="best", framealpha=0.9, fontsize=9)
    ax_psnr.set_title("Per-source validation metrics", fontsize=9, loc="left")

    # ── Composite save-best score (the quantity checkpoint selection
    #    keys on: w_syn·PSNR_syn + w_hst·PSNR_hst + w_rt·PSNR_rt). The
    #    running max is the actual save-best threshold; the model is
    #    checkpointed wherever the raw score touches that envelope. ──
    if ax_score is not None:
        ax_score.plot(score_x, score_y, color="tab:blue", lw=1.5,
                      label="save-best score")
        running_best = np.maximum.accumulate(score_y)
        ax_score.plot(score_x, running_best, color="black", lw=1.2, ls="--",
                      drawstyle="steps-post", label="best so far (save threshold)")
        sx, sy = _smoothed(score_x, score_y)
        if sx is not None:
            ax_score.plot(sx, sy, color="tab:blue", lw=2.6,
                          label="save-best score (MA)")
        ax_score.set_ylabel("Composite score  ·  higher better")
        ax_score.set_title(
            "Overall save-best score (drives checkpoint selection)",
            fontsize=9, loc="left",
        )
        ax_score.legend(loc="best", framealpha=0.9, fontsize=9)

    # X-label on the bottom-most axes only (shared x when stacked).
    (ax_score or ax_psnr).set_xlabel("Step")

    fig.suptitle(
        f"Validation metrics — {len(records)} evals, last step "
        f"{int(steps[-1])}{title_suffix}",
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
