"""``SSHSession.write_text`` streams the file body over stdin.

The remote command the ``ssh`` client receives (its argv, which ``ps`` shows
on the laptop and which the login node's shell sees) carries only the path
plumbing; the body travels on stdin. ``private=True`` additionally makes the
remote file owner-only (mode 600), for the archive credentials and the TNG
API token.

``subprocess.run`` is stubbed to capture the ``ssh`` call; the captured
remote command is then executed with a local ``bash`` (``HOME`` pointed at
``tmp_path``) to check what it really does to the file. No network.
"""

from __future__ import annotations

import os
import stat
import subprocess

import pytest

from euclid_polish.web import remote
from euclid_polish.web.remote import SSHConfig, SSHSession

# The fixtures patch ``subprocess.run`` (one module object, shared with
# ``remote``); the local replay of a captured remote command needs the real one.
_REAL_RUN = subprocess.run

SECRET = "s3cr3t!$x 'q' \"dq\" `id`"


class _Completed:
    returncode = 0
    stdout = b""
    stderr = b""


@pytest.fixture
def captured(monkeypatch):
    calls: list[dict] = []

    def fake_run(args, **kwargs):
        calls.append({"args": list(args), **kwargs})
        return _Completed()

    monkeypatch.setattr(remote.subprocess, "run", fake_run)
    return calls


@pytest.fixture
def session() -> SSHSession:
    sess = SSHSession(SSHConfig(user="u", host="h"))
    sess.is_connected = lambda: True  # type: ignore[method-assign]
    return sess


def _run_remote_locally(cmd: str, body: bytes, home) -> None:
    """Execute a captured remote command the way the login node would."""
    env = {**os.environ, "HOME": str(home)}
    result = _REAL_RUN(["bash", "-c", cmd], input=body, cwd=str(home), env=env,
                       capture_output=True, check=False)
    assert result.returncode == 0, result.stderr.decode()


def test_private_write_keeps_the_body_out_of_the_ssh_argv(session, captured):
    rc, _out, _err = session.write_text("~/.euclid_credentials", f"alice\n{SECRET}\n",
                                        private=True, timeout=15)
    assert rc == 0
    (call,) = captured
    assert all(SECRET not in arg for arg in call["args"])
    assert all("s3cr3t" not in arg for arg in call["args"])
    assert call["input"] == f"alice\n{SECRET}\n".encode()
    assert call["timeout"] == 15


def test_private_write_creates_an_owner_only_file_in_the_remote_home(session, captured, tmp_path):
    session.write_text("~/.euclid_credentials", f"alice\n{SECRET}\n", private=True)
    cmd = captured[0]["args"][-1]
    _run_remote_locally(cmd, captured[0]["input"], tmp_path)

    target = tmp_path / ".euclid_credentials"
    assert target.read_text() == f"alice\n{SECRET}\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert not (tmp_path / ".euclid_credentials.tmp").exists()


def test_private_write_tightens_a_stale_temporary_and_replaces_a_loose_target(
        session, captured, tmp_path):
    """A leftover ``.tmp`` (an interrupted write) or an older world-readable
    target never leaves the new secret readable by others."""
    stale = tmp_path / ".tng_api_key.tmp"
    stale.write_text("old")
    stale.chmod(0o644)
    target = tmp_path / ".tng_api_key"
    target.write_text("older-token\n")
    target.chmod(0o644)

    session.write_text("~/.tng_api_key", "new-token\n", private=True)
    _run_remote_locally(captured[0]["args"][-1], captured[0]["input"], tmp_path)

    assert target.read_text() == "new-token\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert not stale.exists()


def test_plain_write_quotes_the_path_and_leaves_permissions_to_the_umask(
        session, captured, tmp_path):
    (tmp_path / "jobs dir").mkdir()
    path = str(tmp_path / "jobs dir" / "run.sh")
    session.write_text(path, "#!/bin/bash\necho hi\n", executable=True)
    cmd = captured[0]["args"][-1]
    assert "umask" not in cmd and "chmod 600" not in cmd
    _run_remote_locally(cmd, captured[0]["input"], tmp_path)

    with open(path, encoding="utf-8") as handle:
        assert handle.read() == "#!/bin/bash\necho hi\n"
    assert os.access(path, os.X_OK)
