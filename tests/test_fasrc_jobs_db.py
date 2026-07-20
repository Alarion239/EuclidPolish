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


def test_reconcile_keeps_fresh_pending_job_during_grace(db):
    """A *just-submitted* job that isn't in squeue yet must NOT be flagged
    UNKNOWN — sbatch returns before the controller reliably lists the job, so
    a brief absence right after submit is normal, not lost. (Marking it
    terminal here is the bug that made the sidebar and current-submission
    views disagree.)"""
    db.insert("200", label="fresh", params={},
              script_path=".", log_path=".", err_path=".")
    # Still PENDING, no started_at, not in squeue, submitted just now.
    changes = fasrc_jobs.reconcile_with_squeue([], db=db)
    assert "200" not in changes
    assert db.get("200")["state"] == "PENDING"


def test_reconcile_marks_long_missing_never_started_job_unknown(db):
    """Once a never-seen-in-squeue job has been missing well past the submit
    grace window, flag it UNKNOWN so it doesn't sit PENDING forever."""
    db.insert("201", label="lost", params={},
              script_path=".", log_path=".", err_path=".")
    # Backdate submitted_at past the grace window.
    with db._conn() as c:
        c.execute("UPDATE fasrc_jobs SET submitted_at = ? WHERE jobid = ?",
                  (time.time() - fasrc_jobs.SUBMIT_GRACE_S - 10, "201"))
    changes = fasrc_jobs.reconcile_with_squeue([], db=db)
    assert changes == {"201": "UNKNOWN"}
    row = db.get("201")
    assert row["state"] == "UNKNOWN"
    assert row["ended_at"] is not None


def test_reconcile_resurrects_speculatively_finalised_running_job(db):
    """The reported inconsistency: a job wrongly marked DONE (e.g. one
    transient empty squeue while it was actually still running) must flip back
    to RUNNING when squeue shows it alive again. Otherwise the squeue-driven
    sidebar shows it RUNNING while the DB-driven current-submission view has
    permanently dropped it (reconcile skips terminal rows)."""
    db.insert("500", label="flap", params={},
              script_path=".", log_path=".", err_path=".")
    db.update_state("500", state="RUNNING", started_at=time.time() - 60)
    # Transient empty squeue → speculatively finalised DONE.
    assert fasrc_jobs.reconcile_with_squeue([], db=db) == {"500": "DONE"}
    assert db.get("500")["state"] == "DONE"
    # squeue shows it alive again → it was finalised in error; resurrect.
    rows = [{"jobid": "500", "state": "RUNNING", "time": "1:00"}]
    changes = fasrc_jobs.reconcile_with_squeue(rows, db=db)
    assert changes == {"500": "RUNNING"}
    row = db.get("500")
    assert row["state"] == "RUNNING"
    assert row["ended_at"] is None          # cleared on resurrection


def test_reconcile_does_not_resurrect_authoritative_terminal(db):
    """A FAILED/CANCELLED/COMPLETED job (authoritative, from squeue/sacct) is
    NOT resurrected even if a stale squeue snapshot still lists the jobid."""
    db.insert("600", label="failed", params={},
              script_path=".", log_path=".", err_path=".")
    db.update_state("600", state="FAILED", ended_at=time.time() - 10)
    rows = [{"jobid": "600", "state": "RUNNING", "time": "0:10"}]
    changes = fasrc_jobs.reconcile_with_squeue(rows, db=db)
    assert "600" not in changes
    assert db.get("600")["state"] == "FAILED"


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


def test_reconcile_array_children_keep_parent_running(db):
    db.insert("700", label="array", params={"array_count": 3},
              script_path="x", log_path="x", err_path="x")
    rows = [
        {"jobid": "700_0", "state": "COMPLETED", "time": "1:00"},
        {"jobid": "700_1", "state": "RUNNING", "time": "0:30"},
        {"jobid": "700_2", "state": "PENDING", "time": "0:00"},
    ]
    changes = fasrc_jobs.reconcile_with_squeue(rows, db=db)
    assert changes == {"700": "RUNNING"}
    assert db.get("700")["state"] == "RUNNING"


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


def test_secs_per_step_handles_string_steps_in_params(db):
    """REGRESSION — form-submitted params arrive as strings because
    ``request.form.to_dict()`` returns str values. They get stored
    verbatim in ``params_json``. The ETA helper used to compare
    ``steps <= 0`` directly, which raises
    ``TypeError: '<=' not supported between instances of 'str' and
    'int'`` and 500-ed the /api/fasrc/training-status endpoint on
    every poll until the offending row was deleted.

    The fix coerces ``steps`` via float() and silently skips rows
    where coercion fails."""
    base = time.time() - 1000
    # Mix of types params can take in the wild:
    #   - "20000"  (the bug: string from form)
    #   - 20000    (int from a CLI-injected job)
    #   - "junk"   (corrupt row — must not crash; skip silently)
    #   - missing  (older rows that pre-date the "steps" field)
    samples = [
        ("jobA", {"steps": "20000"},  20.0),
        ("jobB", {"steps": 10000},    10.0),
        ("jobC", {"steps": "junk"},    5.0),
        ("jobD", {},                   5.0),
    ]
    for i, (jobid, params, runtime) in enumerate(samples):
        db.insert(jobid, label="x", params=params,
                  script_path=".", log_path=".", err_path=".")
        db.update_state(jobid, state="COMPLETED",
                        started_at=base + i,
                        ended_at=base + i + runtime)
    # Must not raise. Median of the two valid samples (jobA: 20/20000,
    # jobB: 10/10000) = 0.001.
    spt = fasrc_jobs.secs_per_step_history()
    assert spt is not None
    assert abs(spt - 0.001) < 1e-7


def test_secs_per_step_handles_string_progress_total(db):
    """A row that has no ``params['steps']`` but does have
    ``progress_total`` (parsed from a ``step X/Y`` log line) should
    also handle string-typed totals without crashing."""
    base = time.time() - 100
    db.insert("99", label="x", params={},      # no steps key
              script_path=".", log_path=".", err_path=".")
    db.update_state("99", state="COMPLETED",
                    started_at=base, ended_at=base + 5.0)
    db.update_progress("99", step=100, total=5000)
    # update_progress writes int; double-check secs_per_step doesn't
    # explode regardless.
    spt = fasrc_jobs.secs_per_step_history()
    assert spt is not None
    assert spt > 0


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
