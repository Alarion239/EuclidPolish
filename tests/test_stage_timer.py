"""StageTimer CSV schema + row-per-stage behaviour."""

from __future__ import annotations

import csv
import time

import pytest

from euclid_polish.training.stage_timer import StageTimer


EXPECTED_HEADER = [
    "jobid", "stage", "started_at", "ended_at", "duration_seconds",
    "params_dependent", "n_train", "n_valid", "image_size",
    "batch_size", "steps",
]


def _read_rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _read_header(path):
    with open(path) as fh:
        return next(csv.reader(fh))


def test_header_written_on_construction(tmp_path):
    p = tmp_path / "stages.csv"
    StageTimer(str(p), jobid="99")
    assert _read_header(str(p)) == EXPECTED_HEADER


def test_stage_context_writes_one_row(tmp_path):
    p = tmp_path / "stages.csv"
    t = StageTimer(str(p), jobid="42",
                   params={"n_train": 100, "n_valid": 10, "image_size": 60,
                           "batch_size": 4, "steps": 1000})
    with t.stage("generate", params_dependent=True):
        time.sleep(0.05)
    rows = _read_rows(str(p))
    assert len(rows) == 1
    r = rows[0]
    assert r["jobid"] == "42"
    assert r["stage"] == "generate"
    assert r["params_dependent"] == "1"
    assert r["n_train"] == "100"
    assert r["image_size"] == "60"
    assert r["steps"] == "1000"
    dur = float(r["duration_seconds"])
    assert 0.04 < dur < 1.0


def test_multiple_stages_append_in_order(tmp_path):
    p = tmp_path / "stages.csv"
    t = StageTimer(str(p), jobid="100")
    with t.stage("init",     params_dependent=False): pass
    with t.stage("generate", params_dependent=True):  pass
    with t.stage("convolve", params_dependent=True):  pass
    rows = _read_rows(str(p))
    assert [r["stage"] for r in rows] == ["init", "generate", "convolve"]
    assert [r["params_dependent"] for r in rows] == ["0", "1", "1"]


def test_mark_writes_explicit_bracketed_row(tmp_path):
    p = tmp_path / "stages.csv"
    t = StageTimer(str(p), jobid="7")
    t.mark("init", params_dependent=False,
           started_at=1_000_000.0, ended_at=1_000_012.5)
    row = _read_rows(str(p))[0]
    assert row["stage"] == "init"
    assert row["started_at"] == "1000000.000"
    assert row["ended_at"]   == "1000012.500"
    assert row["duration_seconds"] == "12.500"
    assert row["params_dependent"] == "0"


def test_row_written_even_if_stage_raises(tmp_path):
    """Crashed stages still produce a row so the diagnostic data
    survives — bracketed by the timestamps actually observed."""
    p = tmp_path / "stages.csv"
    t = StageTimer(str(p), jobid="9")
    with pytest.raises(RuntimeError):
        with t.stage("train", params_dependent=True):
            raise RuntimeError("kaboom")
    rows = _read_rows(str(p))
    assert len(rows) == 1
    assert rows[0]["stage"] == "train"
    assert float(rows[0]["duration_seconds"]) >= 0


def test_existing_file_is_appended_not_overwritten(tmp_path):
    p = tmp_path / "stages.csv"
    StageTimer(str(p), jobid="1").mark(
        "init", params_dependent=False,
        started_at=1.0, ended_at=2.0,
    )
    # A fresh StageTimer pointed at the same file should NOT rewrite the
    # header — it should append.
    StageTimer(str(p), jobid="1").mark(
        "generate", params_dependent=True,
        started_at=2.0, ended_at=5.0,
    )
    rows = _read_rows(str(p))
    assert len(rows) == 2
    assert [r["stage"] for r in rows] == ["init", "generate"]
