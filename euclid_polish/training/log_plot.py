"""Visualize the training-time validation log written by ``Trainer.train``."""

import csv
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from euclid_polish.training.trainer import (
    PER_BAND_PSNR_COLUMNS,
    TRAINING_LOG_FILENAME,
)

_NUMERIC_LOG_COLS = {
    "step", "wall_time",
    "loss",
    "psnr_stretched", "psnr_raw",
    # Per-band validation PSNRs (psnr_vis / psnr_y_e / psnr_j_e /
    # psnr_h_e) — monitoring only; save-best stays on the joint PSNR.
    *PER_BAND_PSNR_COLUMNS,
    "gnorm_avg", "gnorm_max", "clip_norm", "duration_s",
    "combined_loss",
}

# Wavelength-ordered display colors for the per-band panel: VIS (optical)
# renders blue, the NISP bands shade toward red with wavelength.
_BAND_PLOT_COLORS = {
    "psnr_vis": "tab:blue",
    "psnr_y_e": "tab:olive",
    "psnr_j_e": "tab:orange",
    "psnr_h_e": "tab:red",
}


def read_training_log(log_path: str) -> list[dict]:
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
    records: list[dict],
    output_path: str,
    smooth_window: int = 0,
    title_suffix: str = "",
) -> tuple[int, int]:
    """Plot the validation metrics (from a pre-loaded record list; used by
    the FASRC dashboard which fetches the log over SSH and filters by
    wall-time window).

    Stacked panels (shared x): the validation PSNR (with the running-best
    save threshold), the per-band PSNR when ≥ 2 bands are logged, and the
    training / validation losses when logged.

    Returns ``(n_records, last_step)``.
    """
    if not records:
        raise ValueError("plot_training_records: records is empty")
    steps    = np.array([r["step"]            for r in records])
    psnr_syn = np.array([r["psnr_stretched"]  for r in records])

    def _opt_series(col: str) -> tuple[np.ndarray, np.ndarray]:
        """(steps, values) for the rows that actually carry ``col``.

        Optional columns are blank on some rows (read_training_log drops
        empty cells → key absent) and ``None`` from the SSH parser; both
        are filtered so each curve plots only the points genuinely
        measured.
        """
        xs: list[float] = []
        ys: list[float] = []
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

    # Per-band PSNR series (4-band runs; VIS-only / older logs carry none
    # or just psnr_vis — the panel renders whatever is there).
    band_data = []
    for col in PER_BAND_PSNR_COLUMNS:
        bx, by = _opt_series(col)
        if bx.size:
            label = col.replace("psnr_", "").upper()       # psnr_y_e → Y_E
            band_data.append((bx, by, _BAND_PLOT_COLORS.get(col, "gray"),
                              label, col))
    # One band == the joint metric (VIS-only model) — a separate panel
    # would just duplicate the PSNR line, so require ≥ 2.
    has_bands = len(band_data) >= 2

    # Training loss (the optimised objective, eval-window mean) and the
    # held-out validation loss the LOSS save-best track keys on.
    tl_x, tl_y = _opt_series("loss")
    cl_x, cl_y = _opt_series("combined_loss")
    has_loss = tl_x.size > 0 or cl_x.size > 0

    # Resume baseline: the restored checkpoint's score measured at this
    # run's start (Trainer writes one is_baseline row per resume). The
    # latest one is the current "bar to beat" — drawn as a dashed
    # horizontal line so the run's target is visible from step 0.
    baseline_rows = [r for r in records
                     if str(r.get("is_baseline", "")).strip() in ("1", "1.0")]
    baseline = baseline_rows[-1] if baseline_rows else None

    def _baseline_val(col: str):
        if baseline is None:
            return None
        v = baseline.get(col)
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _smoothed(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        if smooth_window and smooth_window > 1 and y.size >= smooth_window:
            k = np.ones(smooth_window) / smooth_window
            return x[smooth_window - 1:], np.convolve(y, k, mode="valid")
        return None

    # Stacked panels (shared x): PSNR always; per-band PSNR when ≥ 2 bands
    # are logged; the losses when logged. Older / single-metric runs
    # collapse to just the PSNR graph.
    panels = ["psnr"] + (["bands"] if has_bands else []) + (
        ["loss"] if has_loss else [])
    ratios = {"psnr": 3, "bands": 2, "loss": 2}
    if len(panels) == 1:
        fig, ax0 = plt.subplots(figsize=(11, 6))
        axmap = {"psnr": ax0}
    else:
        fig, axs = plt.subplots(
            len(panels), 1, figsize=(11, 3 * len(panels)), sharex=True,
            gridspec_kw={"height_ratios": [ratios[p] for p in panels],
                             "hspace": 0.12},
        )
        axmap = dict(zip(panels, np.atleast_1d(axs), strict=False))
    ax_psnr  = axmap["psnr"]
    ax_bands = axmap.get("bands")
    ax_loss  = axmap.get("loss")

    # ── Validation PSNR (dB), higher is better. The running max is the
    #    actual save-best threshold; the model is checkpointed wherever
    #    the raw curve touches that envelope. ──
    ax_psnr.plot(steps, psnr_syn, color="tab:red", lw=1.6, alpha=0.9,
                 label="Validation PSNR")
    running_best = np.maximum.accumulate(psnr_syn)
    ax_psnr.plot(steps, running_best, color="black", lw=1.2, ls="--",
                 drawstyle="steps-post", label="best so far (save threshold)")
    # Optional smoothed overlay.
    smoothed = _smoothed(steps, psnr_syn)
    if smoothed is not None:
        sx, sy = smoothed
        ax_psnr.plot(sx, sy, color="tab:red", lw=2.6, label="Validation PSNR (MA)")
    ax_psnr.set_ylabel("PSNR (dB)  ·  higher better")

    # Dashed "bar to beat" line at the resume baseline.
    b_syn = _baseline_val("psnr_stretched")
    if b_syn is not None:
        ax_psnr.axhline(b_syn, color="tab:red", lw=1.0, ls=":", alpha=0.7,
                        label="Baseline (prev ckpt)")

    ax_psnr.legend(loc="best", framealpha=0.9, fontsize=9)
    ax_psnr.set_title(
        "Validation PSNR (drives checkpoint selection)", fontsize=9, loc="left")

    # ── Per-band validation PSNR (4-band model). MONITORING ONLY — the
    #    joint PSNR above is what save-best keys on; this panel shows
    #    whether VIS and the noisier NISP channels improve independently
    #    of the NISP-dominated joint number. Colors follow wavelength
    #    (VIS blue → H_E red); per-band resume baselines drawn dotted. ──
    if ax_bands is not None:
        for bx, by, bcolor, blabel, bcol in band_data:
            ax_bands.plot(bx, by, color=bcolor, lw=1.4, alpha=0.9,
                          label=f"{blabel} PSNR")
            smoothed = _smoothed(bx, by)
            if smoothed is not None:
                sx, sy = smoothed
                ax_bands.plot(sx, sy, color=bcolor, lw=2.4,
                              label=f"{blabel} PSNR (MA)")
            b_band = _baseline_val(bcol)
            if b_band is not None:
                ax_bands.axhline(b_band, color=bcolor, lw=1.0, ls=":",
                                 alpha=0.6)
        ax_bands.set_ylabel("PSNR (dB)  ·  per band")
        ax_bands.legend(loc="best", framealpha=0.9, fontsize=8, ncol=2)
        ax_bands.set_title(
            "Per-band validation PSNR (monitoring only — save-best uses "
            "the joint PSNR)", fontsize=9, loc="left",
        )

    # ── Losses (lower is better; log scale). Training loss = the optimised
    #    objective averaged over the eval window; validation loss = the
    #    held-out MAE the LOSS save-best track keys on, with its dotted
    #    running *minimum* as that track's save threshold. ──
    if ax_loss is not None:
        if tl_x.size:
            ax_loss.plot(tl_x, tl_y, color="tab:red", lw=1.6, alpha=0.9,
                         label="Training loss")
            smoothed = _smoothed(tl_x, tl_y)
            if smoothed is not None:
                sx, sy = smoothed
                ax_loss.plot(sx, sy, color="tab:red", lw=2.6,
                             label="Training loss (MA)")
        if cl_x.size:
            ax_loss.plot(cl_x, cl_y, color="tab:orange", lw=1.5,
                         label="Validation loss")
            running_min = np.minimum.accumulate(cl_y)
            ax_loss.plot(cl_x, running_min, color="darkorange", lw=1.2, ls="--",
                         drawstyle="steps-post", label="loss best so far")
            smoothed = _smoothed(cl_x, cl_y)
            if smoothed is not None:
                sx, sy = smoothed
                ax_loss.plot(sx, sy, color="tab:orange", lw=2.6,
                             label="Validation loss (MA)")
            b_loss = _baseline_val("combined_loss")
            if b_loss is not None:
                ax_loss.axhline(b_loss, color="chocolate", lw=1.3, ls=":",
                                alpha=0.85, label="loss baseline (bar to beat)")
        ax_loss.set_yscale("log")
        ax_loss.set_ylabel("Loss  ·  lower better (log)")
        ax_loss.legend(loc="best", framealpha=0.9, fontsize=9)
        ax_loss.set_title("Training / validation loss", fontsize=9, loc="left")

    # X-label on the bottom-most axes only (shared x when stacked).
    axmap[panels[-1]].set_xlabel("Step")

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
) -> tuple[int, int]:
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


def dedupe_latest_per_step(records: list[dict]) -> list[dict]:
    """Keep the LAST-logged record per step, sorted by step.

    Rollbacks rewind the step counter and re-log earlier steps; taking the last
    write per step keeps the value from the *final* pass through it and discards
    the rolled-back ones, so the curve is monotonic in step (no backward jumps).
    """
    by_step: dict[int, dict] = {}
    for rec in records:
        s = rec.get("step")
        if s is None:
            continue
        by_step[int(s)] = rec          # append order is chronological → last wins
    return [by_step[s] for s in sorted(by_step)]


def ensemble_training_series(base_dir: str) -> list[dict]:
    """Per-member training series for a client-side chart.

    Each entry: ``{"name", "psnr": [[step, dB], ...], "loss": [[step, loss], ...]}``,
    rollback-deduped (latest value per step) with resume-baseline rows dropped.
    """
    out: list[dict] = []
    for d in sorted(glob.glob(os.path.join(base_dir, "member_*"))):
        log = os.path.join(d, TRAINING_LOG_FILENAME)
        if not os.path.isfile(log):
            continue
        try:
            recs = read_training_log(log)
        except (FileNotFoundError, ValueError):
            continue
        recs = [r for r in recs
                if str(r.get("is_baseline", "")).strip() not in ("1", "1.0", "true", "True")]
        recs = dedupe_latest_per_step(recs)
        psnr = [[int(r["step"]), float(r["psnr_stretched"])]
                for r in recs if "psnr_stretched" in r]
        loss = [[int(r["step"]), float(r["combined_loss"])]
                for r in recs if "combined_loss" in r]
        if psnr or loss:
            out.append({"name": os.path.basename(d), "psnr": psnr, "loss": loss})
    return out
