"""Tests for the TrainingLog metrics-CSV component — the structured,
resume-continuous training history stored next to the checkpoint."""

import csv
import os

from euclid_polish.observability.training_log import TrainingLog

COLS = ("step", "loss", "psnr_stretched", "combined_loss")


def _read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_append_writes_header_then_rows(tmp_path):
    p = str(tmp_path / "training_log.csv")
    log = TrainingLog(p, COLS)
    assert log.rotated_backup is None
    log.append({"step": 1, "loss": 0.5, "psnr_stretched": 0.4, "combined_loss": ""})
    log.append({"step": 2, "loss": 0.3, "psnr_stretched": 0.2, "combined_loss": 30.0})
    rows = _read(p)
    assert [r["step"] for r in rows] == ["1", "2"]
    assert rows[0]["psnr_stretched"] == "0.4"
    assert rows[1]["combined_loss"] == "30.0"


def test_missing_keys_blank_and_extras_ignored(tmp_path):
    p = str(tmp_path / "training_log.csv")
    log = TrainingLog(p, COLS)
    log.append({"step": 1, "loss": 0.5, "extra": 99})   # missing cols blank; extra dropped
    rows = _read(p)
    assert rows[0]["psnr_stretched"] == ""
    assert "extra" not in rows[0]


def test_append_is_continuous_across_instances(tmp_path):
    """A new TrainingLog over the same path + columns appends (a resume),
    it does not truncate — the whole history persists."""
    p = str(tmp_path / "training_log.csv")
    TrainingLog(p, COLS).append({"step": 1, "loss": 0.5})
    log2 = TrainingLog(p, COLS)            # resume
    assert log2.rotated_backup is None     # same header → no rotation
    log2.append({"step": 2, "loss": 0.4})
    assert [r["step"] for r in _read(p)] == ["1", "2"]


def test_stale_header_rotated(tmp_path):
    p = str(tmp_path / "training_log.csv")
    TrainingLog(p, ("step", "loss")).append({"step": 1, "loss": 0.5})
    # Reopen with a DIFFERENT schema → the old file rotates to a .bak and a
    # fresh, internally-consistent file starts.
    log2 = TrainingLog(p, COLS)
    assert log2.rotated_backup is not None and os.path.exists(log2.rotated_backup)
    log2.append({"step": 2, "loss": 0.3, "psnr_stretched": 0.2, "combined_loss": ""})
    rows = _read(p)
    assert [r["step"] for r in rows] == ["2"]   # only the post-rotation row
    assert "psnr_stretched" in rows[0]
