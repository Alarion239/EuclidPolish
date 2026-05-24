"""Job-history sqlite + ETA heuristic + log parsing."""

from __future__ import annotations

import time

import pytest

from euclid_polish.web import fasrc_jobs


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Fresh JobDB in tmp_path so the tests never touch ~/.euclid_polish."""
    path = tmp_path / "jobs.db"
    fresh = fasrc_jobs.JobDB(path=str(path))
    monkeypatch.setattr(fasrc_jobs, "DB", fresh)
    return fresh


def test_roundtrip_insert_update_get(db):
    db.insert("12345", label="test", params={"steps": 1000},
              script_path="/p/s.sh", log_path="/p/o.out", err_path="/p/e.err")
    row = db.get("12345")
    assert row["state"] == "PENDING"
    assert row["script_path"] == "/p/s.sh"

    t0 = time.time() - 60
    db.update_state("12345", state="RUNNING", started_at=t0)
    db.update_progress("12345", step=500, total=1000)
    row = db.get("12345")
    assert row["state"] == "RUNNING"
    assert row["progress_step"] == 500
    assert row["progress_total"] == 1000
    assert abs(row["started_at"] - t0) < 1.0


def test_reconcile_marks_finished_runs_done(db):
    """A job we'd seen running (started_at set) that has fallen off
    squeue should flip to DONE and pick up an ended_at."""
    db.insert("100", label="t", params={},
              script_path=".", log_path=".", err_path=".")
    db.update_state("100", state="RUNNING",
                    started_at=time.time() - 120)
    # Squeue is empty → the row is no longer in the live queue.
    changes = fasrc_jobs.reconcile_with_squeue([], db=db)
    assert changes == {"100": "DONE"}
    row = db.get("100")
    assert row["state"] == "DONE"
    assert row["ended_at"] is not None


def test_reconcile_marks_never_started_jobs_unknown(db):
    """A job that was inserted PENDING and disappeared from squeue
    without ever being seen as RUNNING (no started_at) is the failure
    mode the user hit: we don't know what happened, so flag UNKNOWN
    instead of leaving the row PENDING forever."""
    db.insert("200", label="lost", params={},
              script_path=".", log_path=".", err_path=".")
    # Still PENDING, no started_at, not in squeue.
    changes = fasrc_jobs.reconcile_with_squeue([], db=db)
    assert changes == {"200": "UNKNOWN"}
    row = db.get("200")
    assert row["state"] == "UNKNOWN"
    assert row["ended_at"] is not None


def test_reconcile_leaves_terminal_rows_alone(db):
    """Once a row is in any TERMINAL_STATES bucket — DONE, FAILED,
    CANCELLED, TIMEOUT, COMPLETED, UNKNOWN — we don't touch it again
    even if a stale squeue snapshot still has the jobid. Belt-and-
    braces against accidental state churn."""
    db.insert("300", label="done", params={},
              script_path=".", log_path=".", err_path=".")
    db.update_state("300", state="COMPLETED", ended_at=time.time() - 10)
    changes = fasrc_jobs.reconcile_with_squeue([], db=db)
    assert "300" not in changes
    assert db.get("300")["state"] == "COMPLETED"


def test_reconcile_promotes_pending_to_running_when_in_squeue(db):
    """The other direction: squeue says RUNNING but our DB still
    thinks the job is PENDING — flip it to RUNNING and record
    started_at so the next reconciliation (if it disappears) marks
    DONE instead of UNKNOWN."""
    db.insert("400", label="up", params={},
              script_path=".", log_path=".", err_path=".")
    squeue_rows = [{"jobid": "400", "state": "RUNNING", "time": "0:30"}]
    changes = fasrc_jobs.reconcile_with_squeue(squeue_rows, db=db)
    assert changes == {"400": "RUNNING"}
    row = db.get("400")
    assert row["state"] == "RUNNING"
    assert row["started_at"] is not None


def test_list_recent_orders_newest_first(db):
    for i in range(5):
        db.insert(f"job{i}", label=f"L{i}", params={"steps": 100 * i},
                  script_path=".", log_path=".", err_path=".")
        time.sleep(0.005)
    recent = db.list_recent(10)
    assert [r["jobid"] for r in recent] == [f"job{i}" for i in range(4, -1, -1)]


def test_eta_returns_none_with_no_history(db):
    assert fasrc_jobs.secs_per_step_history() is None
    assert fasrc_jobs.eta_for_submission(10_000) is None


def test_eta_uses_median_seconds_per_step(db):
    # Synthesize 3 finished jobs at 1ms/step, 2ms/step, 3ms/step.
    samples = [(1_000, 1.0), (2_000, 4.0), (3_000, 9.0)]
    base = time.time() - 1000
    for i, (steps, runtime) in enumerate(samples):
        db.insert(f"job{i}", label="x", params={"steps": steps},
                  script_path=".", log_path=".", err_path=".")
        db.update_state(f"job{i}", state="COMPLETED",
                        started_at=base + i, ended_at=base + i + runtime)
    spt = fasrc_jobs.secs_per_step_history()
    # samples → 1.0/1000, 4.0/2000, 9.0/3000 = 0.001, 0.002, 0.003.
    # Median = 0.002.
    assert abs(spt - 0.002) < 1e-6
    assert abs(fasrc_jobs.eta_for_submission(50_000) - 100.0) < 1e-3


def test_eta_for_running_uses_live_progress(db):
    db.insert("99", label="x", params={"steps": 1000},
              script_path=".", log_path=".", err_path=".")
    db.update_state("99", state="RUNNING", started_at=time.time() - 30)
    db.update_progress("99", step=10, total=1000)
    row = db.get("99")
    eta = fasrc_jobs.eta_for_running(row)
    # 30s for 10 steps → 990 more steps ≈ 2970s. Allow wiggle for clock.
    assert 2800 < eta < 3100


def test_parse_progress_picks_step_total():
    assert fasrc_jobs.parse_progress("Epoch 12345/400000 [12:34<…]") == (12345, 400000)
    assert fasrc_jobs.parse_progress("step 1/250000 loss=4.2") == (1, 250000)


def test_parse_progress_rejects_tiny_totals():
    # The regex would match "shape 4/4", but the guard against total < 50 throws it out.
    assert fasrc_jobs.parse_progress("output shape 4/4") is None


def test_parse_progress_rejects_step_above_total():
    assert fasrc_jobs.parse_progress("counter 9999/100") is None


def test_parse_squeue_pipe_separated():
    """Current format uses ``|`` because modern SLURM doesn't expand
    ``\\t`` inside ``--format`` strings."""
    text = (
        "1001|euclid-1|RUNNING|01:23:45|12:00:00|1|None|2026-05-12T12:00:00\n"
        "1002|euclid-2|PENDING|0:00|12:00:00|1|Resources|N/A\n"
    )
    rows = fasrc_jobs.parse_squeue(text)
    assert len(rows) == 2
    assert rows[0]["jobid"] == "1001"
    assert rows[0]["state"] == "RUNNING"
    assert rows[1]["reason"] == "Resources"


def test_parse_squeue_still_handles_tab_separated_paste():
    """Tab-separated input still parses — handy if someone pastes the
    output of an older squeue call."""
    text = "1001\teuclid-1\tRUNNING\t01:23:45\t12:00:00\t1\tNone\t2026-05-12T12:00:00\n"
    rows = fasrc_jobs.parse_squeue(text)
    assert rows[0]["jobid"] == "1001"
    assert rows[0]["state"] == "RUNNING"


def test_squeue_fmt_uses_pipes():
    """The format string we hand to ``squeue --format`` must use ``|``
    so the bug we just fixed doesn't regress."""
    assert "|" in fasrc_jobs.SQUEUE_FMT
    assert "\\t" not in fasrc_jobs.SQUEUE_FMT


def test_parse_slurm_time_handles_all_three_formats():
    p = fasrc_jobs.parse_slurm_time
    assert p("0:30")        == 30.0
    assert p("1:23")        == 83.0
    assert p("01:23:45")    == 3600 + 23*60 + 45
    assert p("2-00:00:00")  == 2 * 86400
    assert p(None)          == 0.0
    assert p("")            == 0.0
    assert p("garbage")     == 0.0
