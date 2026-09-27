"""Queue promotion is server-side, never a side effect of a GET.

``GET /api/fasrc/current-submission`` used to call the queue tick, so any
page (even a cross-site ``<img src>``) polling it could ``sbatch`` the next
queued step. Promotion now runs in a server-side ticker thread
(:class:`fasrc_queue.QueueTicker`, started by ``app.main`` and by the POSTs
that queue or resume) whose step reconciles against ``squeue`` first.
"""

from __future__ import annotations

import threading

import pytest

from euclid_polish.web import app as web_app
from euclid_polish.web import fasrc_queue, remote
from euclid_polish.web.app import create_app
from euclid_polish.web.routes import fasrc as fasrc_routes


class _SqueueSSH:
    def __init__(self):
        self.commands: list[str] = []

    def is_connected(self):
        return True

    def run(self, cmd, timeout=None):
        self.commands.append(cmd)
        return (0, "", "")


@pytest.fixture
def app(monkeypatch):
    application = create_app()
    application.config["TESTING"] = True
    return application


@pytest.fixture
def ticks(monkeypatch):
    calls: list[object] = []
    monkeypatch.setattr(fasrc_queue.QUEUE, "tick", lambda db, joblog, ssh, submit: calls.append(ssh))
    return calls


def test_polling_the_current_submission_never_promotes(app, monkeypatch, ticks):
    ssh = _SqueueSSH()
    monkeypatch.setattr(remote.STATE, "ssh", ssh)
    fasrc_queue.QUEUE.enqueue({"kind": "step", "step": "x", "form": {}}, "queued step")
    response = app.test_client().get("/api/fasrc/current-submission")
    assert response.status_code == 200, response.get_data(as_text=True)[:300]
    assert any("squeue" in cmd for cmd in ssh.commands)    # still reconciles
    assert ticks == []


def test_the_server_side_step_reconciles_then_ticks(app, monkeypatch, ticks):
    ssh = _SqueueSSH()
    monkeypatch.setattr(remote.STATE, "ssh", ssh)
    step = app.extensions[fasrc_routes.QUEUE_STEP_KEY]
    step()                                                  # nothing queued: no SSH at all
    assert ssh.commands == [] and ticks == []
    fasrc_queue.QUEUE.enqueue({"kind": "step", "step": "x", "form": {}}, "queued step")
    step()
    assert any("squeue" in cmd for cmd in ssh.commands)
    assert ticks == [ssh]


def test_the_step_skips_while_offline(app, monkeypatch, ticks):
    monkeypatch.setattr(remote.STATE, "ssh", None)
    fasrc_queue.QUEUE.enqueue({"kind": "step", "step": "x", "form": {}}, "queued step")
    app.extensions[fasrc_routes.QUEUE_STEP_KEY]()
    assert ticks == []


def test_the_ticker_runs_its_step_on_a_poke_and_stops():
    ran = threading.Event()
    ticker = fasrc_queue.QueueTicker(interval_s=60.0)
    assert ticker.start(ran.set) is True
    assert ticker.start(ran.set) is False                   # one thread per ticker
    ticker.poke()
    assert ran.wait(5)
    ticker.stop()
    assert not ticker.running


def test_a_failing_step_does_not_kill_the_ticker():
    calls: list[int] = []
    again = threading.Event()

    def step():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("squeue timed out")
        again.set()

    ticker = fasrc_queue.QueueTicker(interval_s=60.0)
    ticker.start(step)
    ticker.poke()
    deadline = threading.Event()
    for _ in range(100):
        if calls:
            break
        deadline.wait(0.02)
    ticker.poke()
    assert again.wait(5)
    ticker.stop()


def test_resume_is_still_a_post_only_local_action(app):
    client = app.test_client()
    assert client.get("/api/fasrc/queue/resume").status_code == 405
    assert client.post("/api/fasrc/queue/resume").get_json()["ok"] is True


def test_the_real_server_starts_the_ticker_and_warms_the_caches(app, monkeypatch):
    started: list[object] = []
    warmed = threading.Event()
    monkeypatch.setattr(fasrc_queue.TICKER, "start", lambda step: started.append(step) or True)
    monkeypatch.setattr(web_app.sky_atlas, "layers_payload", lambda: None)
    monkeypatch.setattr(web_app.provenance_index, "get_index", lambda: warmed.set())
    web_app.start_background_services(app)
    assert started == [app.extensions[fasrc_routes.QUEUE_STEP_KEY]]
    assert warmed.wait(5)


def test_the_step_still_reconciles_the_last_promoted_job_after_the_queue_empties(
        app, monkeypatch, ticks):
    # With nothing left queued, the final tick must still run so a finished
    # last job clears ``active_jobid`` (or a failed one halts the queue).
    ssh = _SqueueSSH()
    monkeypatch.setattr(remote.STATE, "ssh", ssh)
    monkeypatch.setattr(fasrc_queue.QUEUE, "items", [])
    monkeypatch.setattr(fasrc_queue.QUEUE, "active_jobid", "4242")
    app.extensions[fasrc_routes.QUEUE_STEP_KEY]()
    assert ticks == [ssh]
    monkeypatch.setattr(fasrc_queue.QUEUE, "halted", True)
    app.extensions[fasrc_routes.QUEUE_STEP_KEY]()
    assert ticks == [ssh]                                   # halted: nothing to reconcile
