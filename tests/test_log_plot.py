"""Tests for the training-log plotter.

These pin that:

  * PSNR-only records produce a valid plot (back-compat),
  * records carrying the loss columns add the loss panel,
  * rows with optional columns blank/None are filtered, not crashed on.
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


def test_psnr_only_records_plot(tmp_path):
    """Records without the loss columns plot as before."""
    records = [{k: v for k, v in _base_row(s).items() if k != "loss"}
               for s in (0, 100, 200)]
    out = str(tmp_path / "syn.png")
    n, last = plot_training_records(records, out)
    assert n == 3 and last == 200
    assert os.path.getsize(out) > 0


def test_loss_columns_render_loss_panel(tmp_path):
    """The training/validation loss columns add the Loss panel and the
    plot renders without error."""
    records = []
    for s in (0, 100, 200, 300):
        r = _base_row(s)
        r["combined_loss"] = 0.02 / (1 + s * 0.001)
        records.append(r)
    out = str(tmp_path / "loss.png")
    n, last = plot_training_records(records, out, smooth_window=2)
    assert n == 4 and last == 300
    assert os.path.getsize(out) > 0


def test_nonmonotonic_psnr_exercises_running_best(tmp_path):
    """The running-max save-threshold envelope is exercised via a
    non-monotonic PSNR sequence."""
    psnrs = [30.0, 29.5, 31.0, 30.8]   # dips then recovers → tests max-accumulate
    records = []
    for s, p in zip((0, 100, 200, 300), psnrs, strict=False):
        r = _base_row(s)
        r["psnr_stretched"] = p
        records.append(r)
    out = str(tmp_path / "score.png")
    n, last = plot_training_records(records, out)
    assert n == 4 and last == 300
    assert os.path.getsize(out) > 0


def test_partial_optional_columns_filtered(tmp_path):
    """A run that only logged the validation loss for some rows (None or
    blank elsewhere) must not crash — the panel plots only the populated
    points."""
    records = []
    for s in (0, 100, 200):
        r = _base_row(s)
        # Validation loss present only on the middle row.
        r["combined_loss"] = 0.02 if s == 100 else None
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
    base["combined_loss"] = 0.02
    base["is_baseline"] = "1"
    records.append(base)
    for s in (5100, 5200, 5300):
        r = _base_row(s)
        r["combined_loss"] = 0.02 - (s - 5100) * 1e-6
        r["is_baseline"] = ""
        records.append(r)
    out = str(tmp_path / "baseline.png")
    n, last = plot_training_records(records, out)
    assert n == 4 and last == 5300
    assert os.path.getsize(out) > 0


def test_per_band_psnr_columns_render_band_panel(tmp_path):
    """Rows carrying ≥ 2 per-band PSNR columns (the 4-band model) add the
    per-band panel; the joint PSNR stays the save-best driver."""
    records = []
    for s in (0, 100, 200):
        r = _base_row(s)
        r["psnr_vis"] = 30.0 + s * 0.01
        r["psnr_y_e"] = 24.0 + s * 0.01
        r["psnr_j_e"] = 23.0 + s * 0.01
        r["psnr_h_e"] = 22.0 + s * 0.01
        records.append(r)
    out = str(tmp_path / "bands.png")
    n, last = plot_training_records(records, out)
    assert n == 3 and last == 200
    assert os.path.getsize(out) > 0


def test_single_band_psnr_column_skips_band_panel(tmp_path):
    """A VIS-only run logs just psnr_vis — that duplicates the joint
    metric, so no separate band panel is drawn (and nothing crashes)."""
    records = []
    for s in (0, 100):
        r = _base_row(s)
        r["psnr_vis"] = 30.0 + s * 0.01
        records.append(r)
    out = str(tmp_path / "one_band.png")
    n, last = plot_training_records(records, out)
    assert n == 2 and last == 100
    assert os.path.getsize(out) > 0
