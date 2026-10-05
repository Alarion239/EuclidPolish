"""The FASRC credential saves never put the secret on a command line.

``/euclid-auth/save`` (``routes/auth.py``) and ``/tng-auth/save``
(``routes/tng.py``) write the Euclid archive password and the IllustrisTNG
API token to owner-only files in the FASRC home. The secret must travel on
the SSH channel's stdin: never in the local ``ssh`` argv (visible to ``ps``
on the laptop) nor in the remote command string (visible to ``ps`` on the
shared login node).

The real :class:`SSHSession` is used with ``subprocess.run`` stubbed, so the
assertions see exactly what would be handed to ``ssh``; the captured remote
command is then run with a local ``bash`` (``HOME`` = ``tmp_path``) to check
the file it leaves behind. No network.
"""

from __future__ import annotations

import os
import stat
import subprocess

import pytest

from euclid_polish.web import remote
from euclid_polish.web.app import create_app
from euclid_polish.web.remote import SSHConfig, SSHSession

# The fixtures patch ``subprocess.run`` (one module object, shared with
# ``remote``); the local replay of a captured remote command needs the real one.
_REAL_RUN = subprocess.run

PASSWORD = "s3cr3t!$x 'q' `id`"
TOKEN = "abc123DEADBEEF"


class _Completed:
    returncode = 0
    stdout = b""
    stderr = b""


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def ssh_calls(monkeypatch):
    """A real SSHSession on ``STATE.ssh`` whose ``ssh`` invocations are recorded."""
    calls: list[dict] = []

    def fake_run(args, **kwargs):
        calls.append({"args": list(args), "input": kwargs.get("input")})
        return _Completed()

    session = SSHSession(SSHConfig(user="u", host="h"))
    session.is_connected = lambda: True  # type: ignore[method-assign]
    monkeypatch.setattr(remote.subprocess, "run", fake_run)
    monkeypatch.setattr(remote.STATE, "ssh", session)
    return calls


def _run_remote_locally(call: dict, home) -> None:
    env = {**os.environ, "HOME": str(home)}
    result = _REAL_RUN(["bash", "-c", call["args"][-1]], input=call["input"] or b"",
                       cwd=str(home), env=env, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr.decode()


def _assert_secret_only_on_stdin(calls: list[dict], secret: str) -> dict:
    assert calls, "nothing was sent over SSH"
    for call in calls:
        for arg in call["args"]:
            assert secret not in arg, f"secret leaked into the ssh argv: {arg!r}"
    carriers = [c for c in calls if c["input"] and secret.encode() in c["input"]]
    assert len(carriers) == 1, "the secret must be streamed once, over stdin"
    return carriers[0]


def test_euclid_password_travels_on_stdin_into_an_owner_only_file(client, ssh_calls, tmp_path):
    r = client.post("/euclid-auth/save",
                    data={"euclid_user": "alice", "euclid_password": PASSWORD})
    assert r.status_code == 200
    assert r.get_json() == {"ok": True, "user": "alice"}
    assert PASSWORD not in r.get_data(as_text=True)

    call = _assert_secret_only_on_stdin(ssh_calls, PASSWORD)
    _run_remote_locally(call, tmp_path)
    creds = tmp_path / ".euclid_credentials"
    assert creds.read_text() == f"alice\n{PASSWORD}\n"
    assert stat.S_IMODE(creds.stat().st_mode) == 0o600


def test_tng_token_travels_on_stdin_into_an_owner_only_file(client, ssh_calls, tmp_path):
    r = client.post("/tng-auth/save", data={"tng_token": TOKEN})
    assert r.status_code == 200
    assert r.get_json() == {"ok": True, "chars": len(TOKEN)}
    assert TOKEN not in r.get_data(as_text=True)

    call = _assert_secret_only_on_stdin(ssh_calls, TOKEN)
    _run_remote_locally(call, tmp_path)
    key = tmp_path / ".tng_api_key"
    assert key.read_text() == f"{TOKEN}\n"
    assert stat.S_IMODE(key.stat().st_mode) == 0o600


@pytest.mark.parametrize("path,data,label", [
    ("/euclid-auth/save", {"euclid_user": "alice", "euclid_password": PASSWORD}, "credentials"),
    ("/tng-auth/save", {"tng_token": TOKEN}, "token"),
])
def test_a_failed_remote_write_reports_stderr_without_the_secret(
        client, monkeypatch, path, data, label):
    class _Failed:
        returncode = 1
        stdout = b""
        stderr = b"cat: .tmp: Disk quota exceeded\n"

    session = SSHSession(SSHConfig(user="u", host="h"))
    session.is_connected = lambda: True  # type: ignore[method-assign]
    monkeypatch.setattr(remote.subprocess, "run", lambda args, **kwargs: _Failed())
    monkeypatch.setattr(remote.STATE, "ssh", session)

    r = client.post(path, data=data)
    assert r.status_code == 500
    body = r.get_json()
    assert body["ok"] is False
    assert body["error"] == f"failed to write {label}: cat: .tmp: Disk quota exceeded"
