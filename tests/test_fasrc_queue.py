"""Tests for the local fail-stop submission queue (fasrc_queue)."""

from __future__ import annotations

import time

import pytest

from euclid_polish.web.fasrc_queue import JobQueue, job_outcome


class FakeDB:
    def __init__(self, rows=None):
        self.rows = rows or {}

    def get(self, jid):
        return self.rows.get(jid)

    def list_recent(self, n=10):
        return list(self.rows.values())


class FakeLog:
    def __init__(self, rows=None):
        self.rows = rows or {}

    def get(self, jid):
        return self.rows.get(jid)


class FakeSSH:
    def __init__(self, connected=True):
        self._c = connected

    def is_connected(self):
        return self._c


@pytest.fixture
def q(tmp_path):
    return JobQueue(path=str(tmp_path / "q.json"))


SPEC = {"kind": "hst", "step": "train", "form": {"steps": "1000"}}


# --------------------------------------------------------------------------
# outcome classifier — the success/fail/pending decision
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sacct_state,expect", [
    ("COMPLETED", "success"),
    ("OUT_OF_MEMORY", "failure"),     # OOM must count as failure
    ("FAILED", "failure"),
    ("TIMEOUT", "failure"),
    ("CANCELLED", "failure"),
    ("NODE_FAIL", "failure"),
    ("RUNNING", "pending"),
])
def test_job_outcome_from_sacct(sacct_state, expect):
    log = FakeLog({"1": {"state": sacct_state}})
    db = FakeDB({"1": {"state": "DONE", "ended_at": time.time()}})
    assert job_outcome("1", db, log)[0] == expect


def test_job_outcome_running_db_only():
    db = FakeDB({"1": {"state": "RUNNING"}})
    assert job_outcome("1", db, FakeLog())[0] == "pending"


def test_job_outcome_done_waits_for_sacct():
    # Left squeue (DONE) but sacct hasn't reported yet → wait (don't promote).
    db = FakeDB({"1": {"state": "DONE", "ended_at": time.time()}})
    assert job_outcome("1", db, FakeLog())[0] == "pending"


def test_job_outcome_done_stale_is_failure():
    # sacct silent for too long → treat as failure (never promote on unknown).
    db = FakeDB({"1": {"state": "DONE", "ended_at": time.time() - 5000}})
    assert job_outcome("1", db, FakeLog())[0] == "failure"


# --------------------------------------------------------------------------
# busy detection + enqueue
# --------------------------------------------------------------------------

def test_active_is_running_for_active_job(q):
    db = FakeDB({"1": {"jobid": "1", "state": "RUNNING"}})
    q.on_direct_submit("1")
    assert q.active_is_running(db) is True


def test_active_is_running_adopts_external_job(q):
    # No queue-tracked active, but a job is RUNNING → adopt + report busy.
    db = FakeDB({"9": {"jobid": "9", "state": "RUNNING"}})
    assert q.active_is_running(db) is True
    assert q.active_jobid == "9"


def test_idle_when_active_terminal(q):
    db = FakeDB({"1": {"jobid": "1", "state": "COMPLETED"}})
    q.on_direct_submit("1")
    assert q.active_is_running(db) is False


def test_enqueue_lists_names(q):
    q.enqueue(SPEC, "train A")
    q.enqueue(SPEC, "train B")
    pub = q.public()
    assert pub["names"] == ["train A", "train B"]
    assert pub["count"] == 2


# --------------------------------------------------------------------------
# tick: promote on success, halt on failure
# --------------------------------------------------------------------------

def test_tick_promotes_on_success(q):
    q.on_direct_submit("1")
    q.enqueue(SPEC, "next job")
    db = FakeDB({"1": {"state": "DONE", "ended_at": time.time()}})
    log = FakeLog({"1": {"state": "COMPLETED"}})
    calls = []
    def submit_fn(spec):
        calls.append(spec)
        return ("2", {"ok": True})
    q.tick(db, log, FakeSSH(), submit_fn)
    assert calls == [SPEC]            # the queued job was submitted
    assert q.active_jobid == "2"
    assert q.public()["count"] == 0
    assert not q.halted


def test_tick_halts_on_failure_including_oom(q):
    q.on_direct_submit("1")
    q.enqueue(SPEC, "should not run")
    db = FakeDB({"1": {"state": "DONE", "ended_at": time.time()}})
    log = FakeLog({"1": {"state": "OUT_OF_MEMORY"}})
    calls = []
    q.tick(db, log, FakeSSH(), lambda s: calls.append(s) or ("x", {}))
    assert calls == []                # NOTHING submitted after a failure
    assert q.halted is True
    assert "OUT_OF_MEMORY" in (q.halted_reason or "")
    assert q.public()["count"] == 1   # the queued job is left in place, halted


def test_tick_pending_is_noop(q):
    q.on_direct_submit("1")
    q.enqueue(SPEC, "waiting")
    db = FakeDB({"1": {"state": "RUNNING"}})
    calls = []
    q.tick(db, FakeLog(), FakeSSH(), lambda s: calls.append(s) or ("x", {}))
    assert calls == []
    assert not q.halted
    assert q.active_jobid == "1"


def test_tick_halts_when_submit_fails(q):
    q.on_direct_submit("1")
    q.enqueue(SPEC, "boom")
    db = FakeDB({"1": {"state": "DONE", "ended_at": time.time()}})
    log = FakeLog({"1": {"state": "COMPLETED"}})
    q.tick(db, log, FakeSSH(), lambda s: (None, {"error": "sbatch exploded"}))
    assert q.halted is True
    assert "sbatch exploded" in (q.halted_reason or "")
    assert q.public()["count"] == 1   # job not consumed


def test_tick_noop_when_ssh_down(q):
    q.on_direct_submit("1")
    q.enqueue(SPEC, "x")
    db = FakeDB({"1": {"state": "DONE", "ended_at": time.time()}})
    log = FakeLog({"1": {"state": "COMPLETED"}})
    calls = []
    q.tick(db, log, FakeSSH(connected=False),
           lambda s: calls.append(s) or ("2", {}))
    assert calls == []                # can't sbatch while disconnected


def test_tick_halted_is_noop(q):
    q.on_direct_submit("1")
    q.enqueue(SPEC, "x")
    q._halt("prior failure")
    db = FakeDB({"1": {"state": "DONE", "ended_at": time.time()}})
    log = FakeLog({"1": {"state": "COMPLETED"}})
    calls = []
    q.tick(db, log, FakeSSH(), lambda s: calls.append(s) or ("2", {}))
    assert calls == []


# --------------------------------------------------------------------------
# persistence + management
# --------------------------------------------------------------------------

def test_persistence_across_reload(q, tmp_path):
    q.on_direct_submit("1")
    q.enqueue(SPEC, "persist me")
    q._halt("died")
    q2 = JobQueue(path=q.path)
    assert q2.active_jobid == "1"
    assert q2.halted is True
    assert q2.public()["names"] == ["persist me"]


def test_clear_resets_items_and_halt(q):
    q.enqueue(SPEC, "a")
    q._halt("x")
    q.clear()
    assert q.public()["count"] == 0
    assert not q.halted


def test_resume_clears_halt_only(q):
    q.enqueue(SPEC, "a")
    q._halt("x")
    q.resume()
    assert not q.halted
    assert q.public()["count"] == 1


def test_remove_one_item(q):
    it = q.enqueue(SPEC, "a")
    q.enqueue(SPEC, "b")
    q.remove(it["id"])
    assert q.public()["names"] == ["b"]
