"""Cover the pexpect-driven SSH auth handshake against a fake ``ssh`` binary.

We can't reach FASRC from a test, but the auth state machine is what
breaks in practice: which prompts arrive, in what order, with what
spelling. Each test plants a different shell script at ``$PATH/ssh``
that mimics one FASRC flavour, then asserts ``SSHSession.connect()``
sends the right secret to the right prompt and writes a control socket.

Three scenarios:

  1. Password + 2FA, prompts in order ``Password:`` → ``VerificationCode:``
  2. SSH key already + 2FA, just one ``VerificationCode:`` prompt
  3. Wrong TOTP — server re-prompts → we must surface a clear error
"""

from __future__ import annotations

import os
import stat
import textwrap
from pathlib import Path

import pytest

from euclid_polish.web.remote import SSHConfig, SSHError, SSHSession


pexpect = pytest.importorskip("pexpect")


def _install_fake_ssh(tmp_path: Path, body: str) -> dict:
    """Write a shell script masquerading as `ssh` and prepend its dir to PATH.

    Returns the env dict to pass to subprocess calls + the artefact paths
    the test can inspect afterwards.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text(body)
    ssh.chmod(0o755)
    # Touch a "control socket" placeholder so ``is_connected()`` can find it.
    sock = tmp_path / "ctrl.sock"
    return {"bin": bin_dir, "ssh": ssh, "sock": sock}


def _make_session(tmp: dict, monkeypatch) -> SSHSession:
    """SSHSession that talks to the fake ssh at tmp['bin']/ssh."""
    monkeypatch.setenv("PATH", f"{tmp['bin']}{os.pathsep}{os.environ['PATH']}")
    return SSHSession(SSHConfig(
        user="tester", host="fake.example", socket=str(tmp["sock"]),
    ))


def test_password_then_verification_code_flow(tmp_path, monkeypatch):
    sock = tmp_path / "ctrl.sock"
    # Fake ssh prompts for Password:, reads password, prompts for
    # VerificationCode:, reads totp, then "succeeds" by writing the
    # success marker and creating the control socket (so is_connected
    # returns true afterwards).
    body = textwrap.dedent(f"""\
        #!/usr/bin/env bash
        printf 'Password: '
        IFS= read -r PWD
        printf 'VerificationCode: '
        IFS= read -r OTP
        if [ "$PWD" != "secret-pw" ]; then echo Permission denied; exit 1; fi
        if [ "$OTP" != "123456" ];   then echo Permission denied; exit 1; fi
        touch {sock}
        echo __EUCLID_POLISH_CONNECTED__
    """)
    tmp = _install_fake_ssh(tmp_path, body)
    # Override our control socket so is_connected() consults the right file.
    sess = _make_session({"bin": tmp["bin"], "sock": sock}, monkeypatch)
    # Patch out `ssh -O check` — our fake control socket isn't a real
    # mux socket, but the file's mere existence is enough for the unit
    # under test. Force is_connected → True only after connect() runs.
    monkeypatch.setattr(sess, "is_connected",
                        lambda: os.path.exists(sock))
    sess.connect(password="secret-pw", totp="123456", timeout=5)
    assert os.path.exists(sock)


def test_verification_code_only_flow(tmp_path, monkeypatch):
    """SSH key auth succeeded silently; just one VerificationCode prompt."""
    sock = tmp_path / "ctrl.sock"
    body = textwrap.dedent(f"""\
        #!/usr/bin/env bash
        # No password prompt — key auth already accepted.
        printf 'VerificationCode: '
        IFS= read -r OTP
        if [ "$OTP" != "654321" ]; then echo Permission denied; exit 1; fi
        touch {sock}
        echo __EUCLID_POLISH_CONNECTED__
    """)
    tmp = _install_fake_ssh(tmp_path, body)
    sess = _make_session({"bin": tmp["bin"], "sock": sock}, monkeypatch)
    monkeypatch.setattr(sess, "is_connected",
                        lambda: os.path.exists(sock))
    # Password value is irrelevant for this flow; we still pass one.
    sess.connect(password="anything", totp="654321", timeout=5)
    assert os.path.exists(sock)


def test_wrong_totp_surfaces_clean_error(tmp_path, monkeypatch):
    """If FASRC re-prompts because the TOTP was wrong, we raise SSHError
    instead of looping forever or burning the full pexpect timeout."""
    sock = tmp_path / "ctrl.sock"
    # Re-prompt forever — exactly what FASRC does for a bad TOTP.
    body = textwrap.dedent("""\
        #!/usr/bin/env bash
        for i in 1 2 3 4 5; do
          printf 'VerificationCode: '
          IFS= read -r OTP
        done
        echo Permission denied
        exit 1
    """)
    tmp = _install_fake_ssh(tmp_path, body)
    sess = _make_session({"bin": tmp["bin"], "sock": sock}, monkeypatch)
    monkeypatch.setattr(sess, "is_connected", lambda: False)
    with pytest.raises(SSHError) as ei:
        sess.connect(password="x", totp="000000", timeout=5)
    assert "verification" in str(ei.value).lower() or "permission" in str(ei.value).lower()
