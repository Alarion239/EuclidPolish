"""End-to-end behaviour of the sacct → JobLog post-mortem path.

Covers:

  * ``reconcile_with_squeue`` triggers sacct + JobLog.record_post_mortem
    on freshly-terminated jobs, and only on those.
  * ``sync_pending_on_connect`` recovers post-mortems for jobs that
    finished while the server was offline.
"""

from __future__ import annotations

from typing import Tuple

import pytest

from euclid_polish.observability import JobLog, JobRecord
from euclid_polish.web import fasrc_jobs


_SACCT_DONE = (
    "12345|COMPLETED|0:0|2026-05-26T14:33:21|2026-05-26T14:35:44|"
    "143|572||8000Mc|4|cpu=4,mem=8000M|4|cpu=4,gres/gpu=1,mem=8000M|2:00:00\n"
    "12345.batch|COMPLETED|0:0|2026-05-26T14:33:21|2026-05-26T14:35:44|"
    "143|572|2048M||||||cpu=4,gres/gpu=1,mem=8000M|"
)


class _SSHStub:
    """SSH stand-in with scripted ``run`` responses keyed by command prefix."""

    def __init__(self, responses=None, *, connected=True):
        self.connected  = connected
        self.responses  = dict(responses or {})
        self.calls: list[str] = []

    def is_connected(self) -> bool:
        return self.connected

    def run(self, cmd: str, *, timeout: float = 10) -> Tuple[int, str, str]:
        self.calls.append(cmd)
        for prefix, resp in self.responses.items():
            if cmd.startswith(prefix):
                return resp
        return 1, "", f"unhandled stub command: {cmd}"


@pytest.fixture
def db(tmp_path):
    """Fresh JobDB so the test never touches ~/.euclid_polish."""
    return fasrc_jobs.JobDB(path=str(tmp_path / "jobs.db"))


@pytest.fixture
def job_log(tmp_path):
    """Fresh JobLog in its own tmp_path."""
    return JobLog(str(tmp_path / "log.csv"))


# ---------------------------------------------------------------------------
# reconcile_with_squeue: sacct hook on terminal transition
# ---------------------------------------------------------------------------

class TestReconcileSacctHook:

    def test_terminated_job_gets_sacct_post_mortem(self, db, job_log):
        """A DB row that was RUNNING but is now absent from squeue → DONE.
        With ssh provided, sacct is queried and the CSV row gets the
        full post-mortem fields."""
        db.insert("12345", label="x", params={},
                  script_path="/s", log_path="/o", err_path="/e")
        db.update_state("12345", state="RUNNING",
                        started_at=1_700_000_000.0)
        job_log.record_submission(JobRecord(jobid="12345"))

        ssh = _SSHStub({"sacct ": (0, _SACCT_DONE, "")})
        changes = fasrc_jobs.reconcile_with_squeue(
            squeue_rows=[], db=db, job_log=job_log, ssh=ssh,
        )

        assert changes == {"12345": "DONE"}
        # sacct was invoked exactly once.
        assert sum(1 for c in ssh.calls if c.startswith("sacct ")) == 1
        # And the CSV row is now populated.
        r = job_log.get("12345")
        assert r["state"] == "COMPLETED"
        assert r["exit_code"] == "0:0"
        assert r["max_rss_mb"] == "2048.0"
        assert r["alloc_gpus"] == "1"

    def test_no_ssh_means_no_sacct_call(self, db, job_log):
        """Reconcile without an ssh handle (e.g. server can't reach
        FASRC right now) must still update DB state but skip sacct."""
        db.insert("99", label="x", params={},
                  script_path="/s", log_path="/o", err_path="/e")
        db.update_state("99", state="RUNNING",
                        started_at=1_700_000_000.0)
        job_log.record_submission(JobRecord(jobid="99"))

        changes = fasrc_jobs.reconcile_with_squeue(
            squeue_rows=[], db=db, job_log=job_log, ssh=None,
        )

        assert changes == {"99": "DONE"}
        # CSV row's post-mortem stays empty.
        assert job_log.get("99")["state"] == ""

    def test_already_terminal_jobs_are_skipped(self, db, job_log):
        """Don't re-query sacct for jobs that were already terminated in
        a previous reconcile pass."""
        db.insert("7", label="x", params={},
                  script_path="/s", log_path="/o", err_path="/e")
        db.update_state("7", state="DONE",
                        ended_at=1_700_000_000.0,
                        started_at=1_700_000_000.0)
        ssh = _SSHStub()  # any sacct call would 1/unhandled and fail

        changes = fasrc_jobs.reconcile_with_squeue(
            squeue_rows=[], db=db, job_log=job_log, ssh=ssh,
        )
        assert changes == {}
        assert sum(1 for c in ssh.calls if c.startswith("sacct ")) == 0

    def test_jobs_still_in_squeue_dont_get_post_mortem(self, db, job_log):
        """Sacct is only fetched on the *transition* to terminal — a
        RUNNING-to-RUNNING tick must not query sacct."""
        db.insert("42", label="x", params={},
                  script_path="/s", log_path="/o", err_path="/e")
        db.update_state("42", state="RUNNING",
                        started_at=1_700_000_000.0)
        ssh = _SSHStub()

        fasrc_jobs.reconcile_with_squeue(
            squeue_rows=[{"jobid": "42", "state": "RUNNING", "time": "10:00"}],
            db=db, job_log=job_log, ssh=ssh,
        )
        assert sum(1 for c in ssh.calls if c.startswith("sacct ")) == 0


# ---------------------------------------------------------------------------
# sync_pending_on_connect
# ---------------------------------------------------------------------------

class TestSyncPendingOnConnect:

    def test_recovers_jobs_that_finished_offline(self, db, job_log):
        """A job that submitted, then finished while the server was off:
        DB still says PENDING/RUNNING; squeue no longer lists it; sacct
        knows what happened. The sync must catch that up."""
        db.insert("12345", label="offline finisher", params={},
                  script_path="/s", log_path="/o", err_path="/e")
        db.update_state("12345", state="RUNNING",
                        started_at=1_700_000_000.0)
        job_log.record_submission(JobRecord(jobid="12345"))

        ssh = _SSHStub({
            # squeue returns no rows for this user → job is gone.
            "squeue ": (0, "", ""),
            "sacct ": (0, _SACCT_DONE, ""),
        })
        changes = fasrc_jobs.sync_pending_on_connect(
            ssh, db=db, job_log=job_log,
        )

        assert changes == {"12345": "DONE"}
        assert db.get("12345")["state"] == "DONE"
        assert job_log.get("12345")["state"] == "COMPLETED"
        assert job_log.get("12345")["max_rss_mb"] == "2048.0"

    def test_no_op_when_ssh_disconnected(self, db, job_log):
        db.insert("1", label="x", params={},
                  script_path="/s", log_path="/o", err_path="/e")
        ssh = _SSHStub({}, connected=False)
        assert fasrc_jobs.sync_pending_on_connect(
            ssh, db=db, job_log=job_log,
        ) == {}
        # Nothing in the DB changed.
        assert db.get("1")["state"] == "PENDING"

    def test_no_op_when_ssh_is_none(self, db, job_log):
        assert fasrc_jobs.sync_pending_on_connect(
            None, db=db, job_log=job_log,
        ) == {}

    def test_squeue_failure_returns_empty(self, db, job_log):
        """A squeue rc≠0 must not crash startup."""
        ssh = _SSHStub({"squeue ": (1, "", "slurmctld unreachable")})
        assert fasrc_jobs.sync_pending_on_connect(
            ssh, db=db, job_log=job_log,
        ) == {}
