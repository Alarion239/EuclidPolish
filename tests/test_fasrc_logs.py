"""Unit tests for the FASRC ``Logs`` tab endpoints.

Both endpoints (``/api/fasrc/runs`` and ``/api/fasrc/runs/log``) run a
single SSH command and parse its stdout. We stub the SSH session with a
canned-response shim so the tests don't depend on GNU ``find -printf``
(unavailable on macOS) or a real FASRC connection.
"""

from __future__ import annotations

import os
import tempfile
import time

import pytest

from euclid_polish.web import app as web_app
from euclid_polish.web import fasrc_config, fasrc_jobs


class StubSSH:
    """Capture-and-respond SSH stand-in.

    Each ``run`` call pops the next ``(rc, stdout, stderr)`` from
    ``responses`` and records the command in ``calls`` so tests can
    assert what the route asked the shell to do.
    """

    def __init__(self, responses: list[tuple[int, str, str]]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def is_connected(self) -> bool:
        return True

    def run(self, cmd: str, timeout: int = 10):
        self.calls.append(cmd)
        if not self.responses:
            return 0, "", ""
        return self.responses.pop(0)


@pytest.fixture
def tmp_repo(tmp_path, monkeypatch):
    """Point the FASRC config at a tmp dir + give us a fresh JobDB."""
    repo = tmp_path / "repo"
    log_dir = repo / "logs" / "jobs"
    log_dir.mkdir(parents=True)
    # Save a temp config — restored at the end of the test
    cfg = fasrc_config.FasrcConfig(repo_path=str(repo))
    monkeypatch.setattr(fasrc_config, "load", lambda: cfg)

    fresh = fasrc_jobs.JobDB(path=str(tmp_path / "jobs.db"))
    monkeypatch.setattr(fasrc_jobs, "DB", fresh)
    return repo, log_dir, fresh


@pytest.fixture
def client(tmp_repo):
    app = web_app.create_app()
    return app.test_client()


def _find_response(rows: list[tuple[float, int, str]]) -> str:
    """Format rows the way GNU ``find -printf '%T@\\t%s\\t%p\\n'`` would."""
    return "\n".join(f"{m}\t{s}\t{p}" for m, s, p in rows) + "\n"


def test_runs_returns_empty_when_log_dir_missing(client, tmp_repo, monkeypatch):
    """If FASRC has no logs/jobs yet, the route still returns ok with an empty list."""
    # /api/fasrc/runs now does TWO ssh calls: squeue first (for state
    # reconciliation), then find. Both empty here.
    web_app.STATE.ssh = StubSSH([(0, "", ""), (0, "", "")])
    r = client.get("/api/fasrc/runs")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["runs"] == []


def test_runs_lists_from_find_alone(client, tmp_repo, monkeypatch):
    """Files-on-disk show up with mtime + size, no jobid when DB has no row."""
    repo, log_dir, _ = tmp_repo
    rows = [
        (1700000000.0, 12345, f"{log_dir}/euclid-20260101-120000.out"),
        (1700000000.0,    42, f"{log_dir}/euclid-20260101-120000.err"),
        (1700000500.0, 99999, f"{log_dir}/euclid-20260202-150000.out"),
        (1700000500.0,   100, f"{log_dir}/euclid-20260202-150000.err"),
    ]
    # 1st response: empty squeue (reconcile no-ops). 2nd: find output.
    web_app.STATE.ssh = StubSSH([(0, "", ""),
                                  (0, _find_response(rows), "")])
    r = client.get("/api/fasrc/runs")
    body = r.get_json()
    assert body["ok"] is True
    runs = body["runs"]
    assert len(runs) == 2
    # Newest first.
    assert runs[0]["name"] == "euclid-20260202-150000"
    assert runs[0]["out_size"] == 99999
    assert runs[0]["err_size"] == 100
    assert runs[0]["jobid"] is None   # not in DB
    assert runs[1]["name"] == "euclid-20260101-120000"


def test_runs_overlays_db_metadata(client, tmp_repo, monkeypatch):
    """DB-known jobs get their jobid / label / state attached to the file row."""
    repo, log_dir, db = tmp_repo
    db.insert("123456", label="my run",
              params={"steps": 1000},
              script_path=f"{repo}/x.sh",
              log_path=f"{log_dir}/euclid-20260202-150000.out",
              err_path=f"{log_dir}/euclid-20260202-150000.err")
    db.update_state("123456", state="COMPLETED",
                    started_at=time.time() - 100, ended_at=time.time())
    rows = [
        (1700000500.0, 99999, f"{log_dir}/euclid-20260202-150000.out"),
        (1700000500.0,   100, f"{log_dir}/euclid-20260202-150000.err"),
    ]
    web_app.STATE.ssh = StubSSH([(0, "", ""),
                                  (0, _find_response(rows), "")])
    r = client.get("/api/fasrc/runs")
    runs = r.get_json()["runs"]
    assert len(runs) == 1
    assert runs[0]["jobid"] == "123456"
    assert runs[0]["state"] == "COMPLETED"
    assert runs[0]["label"] == "my run"
    assert runs[0]["params"] == {"steps": 1000}


def test_runs_shows_db_jobs_with_missing_files(client, tmp_repo):
    """A DB row whose .out has been purged still appears (marked missing)."""
    repo, log_dir, db = tmp_repo
    db.insert("777", label="gone",
              params={"steps": 1},
              script_path=f"{repo}/x.sh",
              log_path=f"{log_dir}/euclid-purged.out",
              err_path=f"{log_dir}/euclid-purged.err")
    db.update_state("777", state="FAILED",
                    started_at=time.time() - 200, ended_at=time.time() - 100)
    web_app.STATE.ssh = StubSSH([(0, "", ""), (0, "", "")])
    r = client.get("/api/fasrc/runs")
    runs = r.get_json()["runs"]
    assert len(runs) == 1
    assert runs[0]["missing"] is True
    assert runs[0]["jobid"] == "777"
    assert runs[0]["state"] == "FAILED"


def test_runs_log_returns_content(client, tmp_repo):
    repo, log_dir, _ = tmp_repo
    path = f"{log_dir}/euclid-20260202-150000.out"
    web_app.STATE.ssh = StubSSH([(0, "hello\nworld\n", "")])
    r = client.get(f"/api/fasrc/runs/log?path={path}&lines=200")
    body = r.get_json()
    assert r.status_code == 200
    assert body["ok"] is True
    assert "hello" in body["content"]


def test_runs_log_rejects_outside_logs_dir(client, tmp_repo):
    web_app.STATE.ssh = StubSSH([])
    r = client.get("/api/fasrc/runs/log?path=/etc/passwd&lines=100")
    assert r.status_code == 400


def test_runs_log_rejects_wrong_extension(client, tmp_repo):
    repo, log_dir, _ = tmp_repo
    r = client.get(f"/api/fasrc/runs/log?path={log_dir}/x.sh&lines=100")
    assert r.status_code == 400


def test_runs_log_rejects_path_traversal(client, tmp_repo):
    repo, log_dir, _ = tmp_repo
    r = client.get(f"/api/fasrc/runs/log?path={log_dir}/../../etc/passwd.out&lines=100")
    assert r.status_code == 400


def test_runs_log_clamps_lines_param(client, tmp_repo):
    """``lines`` clamps to [50, 10000]; the test inspects the actual tail command."""
    repo, log_dir, _ = tmp_repo
    path = f"{log_dir}/euclid-20260202-150000.out"
    ssh = StubSSH([(0, "hi\n", "")])
    web_app.STATE.ssh = ssh
    client.get(f"/api/fasrc/runs/log?path={path}&lines=999999")
    # Should have requested at most 10000 lines.
    assert " -n 10000 " in ssh.calls[0]

    ssh.responses = [(0, "hi\n", "")]; ssh.calls.clear()
    client.get(f"/api/fasrc/runs/log?path={path}&lines=1")
    # Should bump up to at least 50.
    assert " -n 50 " in ssh.calls[0]


def test_runs_handles_ssh_disconnected(client, tmp_repo):
    web_app.STATE.ssh = None
    r = client.get("/api/fasrc/runs")
    assert r.status_code == 400
    r = client.get("/api/fasrc/runs/log?path=/anything.out&lines=100")
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Log viewer pagination (page counted from the END; page 0 = newest)
# ---------------------------------------------------------------------------

def _lines(a, b):
    return "\n".join(f"L{i}" for i in range(a, b + 1)) + "\n"


def test_log_pagination_newest_page(client, tmp_repo):
    repo, log_dir, _ = tmp_repo
    path = f"{log_dir}/run.out"
    # 120-line file, page_size 50 → page 0 = lines 71–120 (the newest block).
    web_app.STATE.ssh = StubSSH([(0, "120\n", ""), (0, _lines(71, 120), "")])
    r = client.get(f"/api/fasrc/runs/log?path={path}&page=0&page_size=50")
    d = r.get_json()
    assert d["ok"] and d["total_lines"] == 120
    assert (d["start_line"], d["end_line"]) == (71, 120)
    assert d["has_older"] is True and d["has_newer"] is False
    assert "L71" in d["content"] and "L120" in d["content"]


def test_log_pagination_middle_and_first_page(client, tmp_repo):
    repo, log_dir, _ = tmp_repo
    path = f"{log_dir}/run.out"
    # page 1 → lines 21–70 (older block, both neighbours exist).
    web_app.STATE.ssh = StubSSH([(0, "120\n", ""), (0, _lines(21, 70), "")])
    d = client.get(f"/api/fasrc/runs/log?path={path}&page=1&page_size=50").get_json()
    assert (d["start_line"], d["end_line"]) == (21, 70)
    assert d["has_older"] is True and d["has_newer"] is True

    # page 2 → the very first lines 1–20; no older page left.
    web_app.STATE.ssh = StubSSH([(0, "120\n", ""), (0, _lines(1, 20), "")])
    d = client.get(f"/api/fasrc/runs/log?path={path}&page=2&page_size=50").get_json()
    assert (d["start_line"], d["end_line"]) == (1, 20)
    assert d["has_older"] is False and d["has_newer"] is True
    assert "L1" in d["content"]


def test_log_pagination_past_beginning_is_empty(client, tmp_repo):
    repo, log_dir, _ = tmp_repo
    path = f"{log_dir}/run.out"
    # page 3 is before the start → empty content, only wc -l is called (no sed).
    web_app.STATE.ssh = StubSSH([(0, "120\n", "")])
    d = client.get(f"/api/fasrc/runs/log?path={path}&page=3&page_size=50").get_json()
    assert d["content"] == "" and (d["start_line"], d["end_line"]) == (0, 0)
    assert d["has_older"] is False and d["has_newer"] is True


# ---------------------------------------------------------------------------
# Runs LIST pagination (page back through the full job history)
# ---------------------------------------------------------------------------

def _runs_find(n, log_dir):
    """Fake `find -printf '%T@\\t%s\\t%p'` output for n runs (mtime = index)."""
    rows = []
    for i in range(1, n + 1):
        stem = f"euclid-{i:04d}"
        rows.append(f"{float(i)}\t{100+i}\t{log_dir}/{stem}.out")
        rows.append(f"{float(i)}\t{10+i}\t{log_dir}/{stem}.err")
    return "\n".join(rows) + "\n"


def test_runs_list_pagination_newest_page(client, tmp_repo):
    repo, log_dir, _ = tmp_repo
    find_out = _runs_find(250, str(log_dir))
    web_app.STATE.ssh = StubSSH([(0, "", ""), (0, find_out, "")])  # squeue, find
    d = client.get("/api/fasrc/runs?page=0&page_size=100").get_json()
    assert d["ok"] and d["total_runs"] == 250
    assert len(d["runs"]) == 100
    assert (d["start_index"], d["end_index"]) == (1, 100)
    assert d["has_older"] is True and d["has_newer"] is False
    assert d["runs"][0]["name"] == "euclid-0250"     # newest first


def test_runs_list_pagination_oldest_page(client, tmp_repo):
    repo, log_dir, _ = tmp_repo
    find_out = _runs_find(250, str(log_dir))
    web_app.STATE.ssh = StubSSH([(0, "", ""), (0, find_out, "")])
    d = client.get("/api/fasrc/runs?page=2&page_size=100").get_json()
    assert len(d["runs"]) == 50                       # runs 201–250
    assert (d["start_index"], d["end_index"]) == (201, 250)
    assert d["has_older"] is False and d["has_newer"] is True
    assert d["runs"][-1]["name"] == "euclid-0001"     # the very first run


def test_runs_list_pagination_past_end_empty(client, tmp_repo):
    repo, log_dir, _ = tmp_repo
    find_out = _runs_find(250, str(log_dir))
    web_app.STATE.ssh = StubSSH([(0, "", ""), (0, find_out, "")])
    d = client.get("/api/fasrc/runs?page=3&page_size=100").get_json()
    assert d["runs"] == [] and d["total_runs"] == 250
    assert (d["start_index"], d["end_index"]) == (0, 0)
    assert d["has_older"] is False and d["has_newer"] is True
