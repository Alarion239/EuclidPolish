"""Job-history sqlite, squeue/SLURM-time parsing and row compaction."""

from __future__ import annotations

import json
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
    row = db.get("12345")
    assert row["state"] == "RUNNING"
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


# ---------------------------------------------------------------------------
# compact_params: history/tracking rows without the embedded payload blobs
# ---------------------------------------------------------------------------

def test_compact_params_drops_private_payload_blobs():
    big = "{" + "x" * 5000 + "}"
    params = {"num_stars": "10000", "_star_prior_json": big,
              "_cosmos_vis_transfer_artifact_json": {"a": 1},
              "members": "member_01,member_02", "_tiny": "ok"}
    kept, omitted = fasrc_jobs.compact_params(params)
    assert kept == {"num_stars": "10000", "members": "member_01,member_02", "_tiny": "ok"}
    assert omitted["_star_prior_json"] == len(big)
    assert omitted["_cosmos_vis_transfer_artifact_json"] > 0
    assert "_tiny" not in omitted


def test_compact_params_truncates_any_huge_public_value():
    params = {"extra_flags": "y" * 20_000}
    kept, omitted = fasrc_jobs.compact_params(params)
    assert "extra_flags" not in kept
    assert omitted == {"extra_flags": 20_000}


def test_compact_row_ships_params_once():
    row = {"jobid": "1", "params_json": '{"a":"1","_joint_galaxy_population_json":"' + "z" * 4000 + '"}'}
    out = fasrc_jobs.compact_row(row)
    assert out["params"] == {"a": "1"}
    assert out["params_omitted"] == {"_joint_galaxy_population_json": 4000}
    assert "params_json" not in out                     # not the same data twice
    assert row["params_json"].startswith('{"a"')        # input untouched


@pytest.mark.parametrize(("ledger", "db", "shown", "unresolved"), [
    ("COMPLETED", "DONE", "COMPLETED", False),       # sacct's verdict wins
    ("OUT_OF_MEMORY", "DONE", "OUT_OF_MEMORY", False),
    ("CANCELLED by 123", "", "CANCELLED", False),
    ("RUNNING", "CANCELLED", "CANCELLED", True),     # stale live ledger never wins
    ("RUNNING", "DONE", "DONE", True),
    ("PENDING", "CANCELLED", "CANCELLED", True),
    ("RUNNING", "RUNNING", "RUNNING", False),        # genuinely live
    ("RUNNING", "", "RUNNING", False),               # nothing to compare against
    ("", "RUNNING", "RUNNING", False),
    ("", "UNKNOWN", "UNKNOWN", True),
    ("DONE", "COMPLETED", "COMPLETED", True),
    ("UNKNOWN", "DONE", "UNKNOWN", True),
    ("", "", "PENDING", True),
])
def test_display_state_and_unresolved(ledger, db, shown, unresolved):
    assert fasrc_jobs.display_state(ledger, db) == shown
    assert fasrc_jobs.is_unresolved(ledger, db) is unresolved


def test_compact_row_survives_malformed_params_json():
    out = fasrc_jobs.compact_row({"jobid": "1", "params_json": "{not json"})
    assert out["params"] == {} and out["params_omitted"] == {}
