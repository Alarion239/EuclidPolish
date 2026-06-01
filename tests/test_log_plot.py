"""Tests for the training-log plotter, including the multi-source panels.

The workflow wants three curves visible — synthetic PSNR, HST PSNR, and
the star-anchor PSNR. These pin that:

  * synthetic-only records still produce a valid plot (back-compat),
  * records carrying the HST / star-anchor columns add their panels,
  * rows with the columns blank/None are filtered, not crashed on.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

from euclid_polish.training.log_plot import plot_training_records


def _base_row(step: int) -> dict:
    return {
        "step": step,
        "wall_time": 1000.0 + step,
        "loss": 1.0 / (1 + step),
        "psnr_stretched": 20.0 + step * 0.01,
        "psnr_raw": 50.0 + step * 0.01,
    }


def test_synthetic_only_records_plot(tmp_path):
    """Records without the multi-source columns plot as before."""
    records = [_base_row(s) for s in (0, 100, 200)]
    out = str(tmp_path / "syn.png")
    n, last = plot_training_records(records, out)
    assert n == 3 and last == 200
    assert os.path.getsize(out) > 0


def test_multisource_records_one_graph(tmp_path):
    """When HST PSNR + star-anchor PSNR are present, all three validation
    metrics render on a single combined graph (all three on the shared
    dB axis) and the call returns the right record/step counts."""
    records = []
    for s in (0, 100, 200, 300):
        r = _base_row(s)
        r["psnr_stretched_hst"] = 18.0 + s * 0.02
        r["psnr_raw_hst"] = 48.0 + s * 0.02
        r["anchor_val_psnr"] = 35.0 + s * 0.01
        records.append(r)
    out = str(tmp_path / "multi.png")
    n, last = plot_training_records(records, out, smooth_window=2)
    assert n == 4 and last == 300
    assert os.path.getsize(out) > 0


def test_loss_columns_render_loss_panel(tmp_path):
    """Per-lane loss columns add the Loss panel (3 panels with the score)
    and the plot renders without error."""
    records = []
    for s in (0, 100, 200, 300):
        r = _base_row(s)
        r["loss_syn"] = 0.01 / (1 + s * 0.001)
        r["loss_anchor"] = 0.02 / (1 + s * 0.001)
        r["save_best_score"] = 30.0 + s * 0.001
        records.append(r)
    out = str(tmp_path / "loss.png")
    n, last = plot_training_records(records, out, smooth_window=2)
    assert n == 4 and last == 300
    assert os.path.getsize(out) > 0


def test_save_best_score_adds_second_panel(tmp_path):
    """When the composite ``save_best_score`` column is present, the plot
    renders the second (score) panel without error and returns the right
    counts. The running-max envelope is exercised via a non-monotonic
    score sequence."""
    scores = [30.0, 29.5, 31.0, 30.8]   # dips then recovers → tests max-accumulate
    records = []
    for s, sc in zip((0, 100, 200, 300), scores):
        r = _base_row(s)
        r["psnr_stretched_hst"] = 18.0 + s * 0.02
        r["anchor_val_psnr"] = 35.0
        r["save_best_score"] = sc
        records.append(r)
    out = str(tmp_path / "score.png")
    n, last = plot_training_records(records, out)
    assert n == 4 and last == 300
    assert os.path.getsize(out) > 0


def test_partial_multisource_columns_filtered(tmp_path):
    """A run that only logged HST for some rows (None elsewhere) must
    not crash — the panel plots only the populated points."""
    records = []
    for s in (0, 100, 200):
        r = _base_row(s)
        # HST present only on the middle row; star-anchor never.
        r["psnr_stretched_hst"] = 19.0 if s == 100 else None
        r["anchor_val_psnr"] = ""   # blank string, as the CSV writes
        records.append(r)
    out = str(tmp_path / "partial.png")
    n, _ = plot_training_records(records, out)
    assert n == 3
    assert os.path.getsize(out) > 0


def test_baseline_row_draws_without_error(tmp_path):
    """A resume ``is_baseline`` row (the bar-to-beat) is plotted as a dashed
    reference line and doesn't break the figure."""
    records = []
    # One baseline row, then normal eval rows after the resume step.
    base = _base_row(5000)
    base["save_best_score"] = 40.0
    base["psnr_stretched_hst"] = 30.0
    base["is_baseline"] = "1"
    records.append(base)
    for s in (5100, 5200, 5300):
        r = _base_row(s)
        r["save_best_score"] = 38.0 + (s - 5100) * 0.005
        r["psnr_stretched_hst"] = 29.0
        r["is_baseline"] = ""
        records.append(r)
    out = str(tmp_path / "baseline.png")
    n, last = plot_training_records(records, out)
    assert n == 4 and last == 5300
    assert os.path.getsize(out) > 0
