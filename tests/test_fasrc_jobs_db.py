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


def test_parse_squeue_handles_tab_separated_fixed_format():
    text = (
        "1001\teuclid-1\tRUNNING\t01:23:45\t12:00:00\t1\tNone\t2026-05-12T12:00:00\n"
        "1002\teuclid-2\tPENDING\t0:00\t12:00:00\t1\tResources\tN/A\n"
    )
    rows = fasrc_jobs.parse_squeue(text)
    assert len(rows) == 2
    assert rows[0]["jobid"] == "1001"
    assert rows[0]["state"] == "RUNNING"
    assert rows[1]["reason"] == "Resources"
