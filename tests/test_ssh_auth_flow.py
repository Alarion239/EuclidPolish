"""Cover ``SSHSession.connect()`` against a fake ``ssh`` binary.

Connect is just ``ssh -M -S <socket> -o BatchMode=yes <target> true`` now
— no pexpect, no prompts. The fake ssh either succeeds (touches the
socket file and exits 0) or fails with ``Permission denied
(publickey)``. We assert the success path creates the socket and the
failure path bubbles up a clean :class:`SSHError` that points the user
at ``ssh-copy-id``.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from euclid_polish.web.remote import SSHConfig, SSHError, SSHSession


def _install_fake_ssh(tmp_path: Path, body: str) -> dict:
    """Write a shell script masquerading as `ssh` and prepend its dir to PATH."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text(body)
    ssh.chmod(0o755)
    sock = tmp_path / "ctrl.sock"
    return {"bin": bin_dir, "ssh": ssh, "sock": sock}


def _make_session(tmp: dict, monkeypatch) -> SSHSession:
    monkeypatch.setenv("PATH", f"{tmp['bin']}{os.pathsep}{os.environ['PATH']}")
    return SSHSession(SSHConfig(
        user="tester", host="fake.example", socket=str(tmp["sock"]),
    ))


def test_connect_succeeds_when_key_auth_works(tmp_path, monkeypatch):
    """Happy path: ssh -M ... exits 0, the control socket appears,
    ``SSHSession.connect()`` returns cleanly."""
    sock = tmp_path / "ctrl.sock"
    body = textwrap.dedent(f"""\
        #!/usr/bin/env bash
        # Just simulate the master socket being created by a successful
        # ControlMaster handshake.
        touch {sock}
        exit 0
    """)
    tmp = _install_fake_ssh(tmp_path, body)
    sess = _make_session({"bin": tmp["bin"], "sock": sock}, monkeypatch)
    # Patch out ``ssh -O check`` — our fake control socket isn't a real
    # mux socket, but the file's existence is what is_connected() uses.
    monkeypatch.setattr(sess, "is_connected",
                        lambda: os.path.exists(sock))
    sess.connect(timeout=5)
    assert os.path.exists(sock)


def test_connect_fails_with_helpful_message_on_publickey_denied(tmp_path,
                                                                 monkeypatch):
    """If the user's key isn't installed yet, ssh exits 255 with
    ``Permission denied (publickey).`` — we should surface that and
    explicitly mention ``ssh-copy-id`` so the user knows the one-shot fix."""
    sock = tmp_path / "ctrl.sock"
    body = textwrap.dedent("""\
        #!/usr/bin/env bash
        echo "Permission denied (publickey)." >&2
        exit 255
    """)
    tmp = _install_fake_ssh(tmp_path, body)
    sess = _make_session({"bin": tmp["bin"], "sock": sock}, monkeypatch)
    monkeypatch.setattr(sess, "is_connected", lambda: False)
    with pytest.raises(SSHError) as ei:
        sess.connect(timeout=5)
    msg = str(ei.value)
    assert "Permission denied" in msg
    assert "ssh-copy-id" in msg
    assert "tester@fake.example" in msg


def test_connect_noop_when_already_connected(tmp_path, monkeypatch):
    """Second connect() against a live socket should short-circuit
    without invoking ssh at all (otherwise we'd waste a round-trip on
    every page render that hits /api/fasrc/connect)."""
    sock = tmp_path / "ctrl.sock"
    sock.touch()
    # Fake ssh that explodes if called — proves connect() didn't fire it.
    body = "#!/usr/bin/env bash\necho 'ssh should not have been called' >&2\nexit 99\n"
    tmp = _install_fake_ssh(tmp_path, body)
    sess = _make_session({"bin": tmp["bin"], "sock": sock}, monkeypatch)
    monkeypatch.setattr(sess, "is_connected", lambda: True)
    sess.connect(timeout=5)              # must not raise, must not call ssh
