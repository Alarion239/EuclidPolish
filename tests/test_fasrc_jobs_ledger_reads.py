"""``reconcile_with_squeue`` consults the CSV job ledger once per call (one
lookup for every recent terminal job), not once per job — it runs on every
SLURM-feed poll, and each ledger read used to be a full parse of the file."""

from __future__ import annotations

import pytest

from euclid_polish.observability import JobLog, JobRecord
from euclid_polish.web import fasrc_jobs

_READS = ("get", "get_many", "list_all", "history_for_step", "history_by_step")


@pytest.fixture
def db(tmp_path):
    return fasrc_jobs.JobDB(path=str(tmp_path / "jobs.db"))


@pytest.fixture
def job_log(tmp_path):
    return JobLog(str(tmp_path / "log.csv"))


def _count_reads(monkeypatch, log: JobLog) -> dict[str, int]:
    """Count calls to the ledger's public read methods on this instance."""
    calls = dict.fromkeys(_READS, 0)
    for name in _READS:
        real = getattr(JobLog, name, None)
        if real is None:
            continue

        def spy(*args, _name=name, _real=real, **kwargs):
            calls[_name] += 1
            return _real(log, *args, **kwargs)

        monkeypatch.setattr(log, name, spy)
    return calls


def _terminal_job(db, log, jobid, *, state="DONE", ledger_state="COMPLETED"):
    db.insert(jobid, label="x", params={}, script_path="/s", log_path="/o", err_path="/e")
    db.update_state(jobid, state=state, ended_at=1_700_000_100.0)
    log.record_submission(JobRecord(jobid=jobid))
    if ledger_state:
        log.record_post_mortem(jobid, {"state": ledger_state})


def test_reconcile_reads_the_ledger_once_for_many_terminal_jobs(db, job_log, monkeypatch):
    for i in range(20):
        _terminal_job(db, job_log, str(1000 + i),
                      state=("DONE", "UNKNOWN", "CANCELLED")[i % 3])
    calls = _count_reads(monkeypatch, job_log)

    changes = fasrc_jobs.reconcile_with_squeue([], db=db, job_log=job_log, ssh=None)

    assert changes == {}
    assert sum(calls.values()) == 1, calls


def test_no_ledger_read_without_terminal_jobs(db, job_log, monkeypatch):
    db.insert("42", label="x", params={}, script_path="/s", log_path="/o", err_path="/e")
    db.update_state("42", state="RUNNING", started_at=1_700_000_000.0)
    calls = _count_reads(monkeypatch, job_log)

    fasrc_jobs.reconcile_with_squeue(
        [{"jobid": "42", "state": "RUNNING", "time": "10:00"}],
        db=db, job_log=job_log, ssh=None)

    assert sum(calls.values()) == 0, calls


class _SacctSSH:
    """Connected SSH stand-in: sacct reports CANCELLED for whatever job id."""

    def __init__(self) -> None:
        self.sacct_jobids: list[str] = []

    def is_connected(self) -> bool:
        return True

    def run(self, cmd: str, *, timeout: float = 10) -> tuple[int, str, str]:
        if cmd.startswith("sacct "):
            jobid = cmd.split("-j ", 1)[1].split()[0].strip("'\"")
            self.sacct_jobids.append(jobid)
            return 0, (
                f"{jobid}|CANCELLED by 1000|0:15|2026-05-26T14:33:21|2026-05-26T14:35:44|"
                "143|572|09:32||8000Mc|4|cpu=4,mem=8000M|4|cpu=4,mem=8000M|2:00:00\n"), ""
        return 1, "", "unhandled"


def test_only_blank_ledger_rows_are_retried_in_db_order(db, job_log, monkeypatch):
    """The single ledger read still selects exactly the terminal jobs whose
    ledger state is blank (a filled, missing or live row is left alone)."""
    monkeypatch.setattr(fasrc_jobs, "fetch_jobstats_stats", lambda _ssh, _jobid: None)
    _terminal_job(db, job_log, "1", ledger_state="")
    _terminal_job(db, job_log, "2", ledger_state="COMPLETED")
    _terminal_job(db, job_log, "3", ledger_state="  ")
    db.insert("4", label="x", params={}, script_path="/s", log_path="/o", err_path="/e")
    db.update_state("4", state="DONE", ended_at=1_700_000_100.0)  # no ledger row
    ssh = _SacctSSH()

    fasrc_jobs.reconcile_with_squeue([], db=db, job_log=job_log, ssh=ssh)

    expected = [r["jobid"] for r in db.list_recent(50) if r["jobid"] in {"1", "3"}]
    assert ssh.sacct_jobids == expected
    assert job_log.get("1")["state"] == "CANCELLED"
    assert job_log.get("2")["state"] == "COMPLETED"
    assert job_log.get("3")["state"] == "CANCELLED"
    assert job_log.get("4") is None
