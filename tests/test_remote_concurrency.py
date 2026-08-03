"""SSHSession runs commands concurrently over the one ControlMaster socket.

Before: a single ``threading.Lock`` serialised every ``run``/``rsync``, so
a long transfer or the training-status poll blocked all small interactive
commands. Now a bounded semaphore lets up to ``max_concurrency`` commands
run at once (ControlMaster multiplexes the channels), while still capping
concurrency so we stay under sshd's MaxSessions and are a good login-node
citizen.

These tests stub ``subprocess.run`` so no real SSH is needed; they assert
on the observed peak concurrency.
"""

from __future__ import annotations

import threading
import time

import pytest

from euclid_polish.web.remote import SSHConfig, SSHSession


class _FakeCompleted:
    def __init__(self) -> None:
        self.returncode = 0
        self.stdout = b"ok"
        self.stderr = b""


def _make_session(max_concurrency: int) -> SSHSession:
    sess = SSHSession(SSHConfig(user="u", host="h", max_concurrency=max_concurrency))
    # Pretend the ControlMaster is live so run() doesn't short-circuit.
    sess.is_connected = lambda: True  # type: ignore[method-assign]
    return sess


def _concurrency_probe(hold_s: float):
    """Return (fake_run, get_peak): fake_run records peak simultaneous calls."""
    state = {"active": 0, "peak": 0}
    lock = threading.Lock()

    def fake_run(args, **kwargs):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(hold_s)
        with lock:
            state["active"] -= 1
        return _FakeCompleted()

    return fake_run, lambda: state["peak"]


def _run_n(sess: SSHSession, n: int) -> None:
    threads = [threading.Thread(target=lambda: sess.run("echo hi")) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_run_is_concurrent_not_serialized(monkeypatch):
    sess = _make_session(max_concurrency=4)
    fake_run, peak = _concurrency_probe(hold_s=0.10)
    monkeypatch.setattr("euclid_polish.web.remote.subprocess.run", fake_run)

    _run_n(sess, 4)

    # Old single-lock behaviour would force peak == 1. With the semaphore,
    # the four commands overlap.
    assert peak() >= 2


def test_run_respects_max_concurrency(monkeypatch):
    sess = _make_session(max_concurrency=2)
    fake_run, peak = _concurrency_probe(hold_s=0.05)
    monkeypatch.setattr("euclid_polish.web.remote.subprocess.run", fake_run)

    _run_n(sess, 8)

    # The bounded semaphore must never let more than max_concurrency run.
    assert peak() <= 2


def test_write_text_streams_large_body_over_stdin(monkeypatch):
    session = _make_session(max_concurrency=1)
    captured = {}
    body = "fitted-population-json:" + "x" * 50_000

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["input"] = kwargs.get("input")
        return _FakeCompleted()

    monkeypatch.setattr("euclid_polish.web.remote.subprocess.run", fake_run)

    rc, _out, _err = session.write_text(
        "/remote/logs/job.sh", body, executable=True,
    )

    assert rc == 0
    assert captured["input"] == body.encode("utf-8")
    remote_command = captured["args"][-1]
    assert body not in remote_command
    assert "job.sh.tmp" in remote_command
    assert "chmod +x" in remote_command
