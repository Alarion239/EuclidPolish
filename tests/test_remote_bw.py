"""BitwardenClient: missing-binary + unlock-failure error surfaces."""

from __future__ import annotations

import subprocess

import pytest

from euclid_polish.web.remote import BitwardenClient, BitwardenError


def test_unlock_without_password_raises():
    bw = BitwardenClient(bw_path="/usr/bin/false")  # path doesn't matter — empty pw
    with pytest.raises(BitwardenError, match="master password"):
        bw.unlock("")


def test_unlock_reports_missing_binary(tmp_path):
    bw = BitwardenClient(bw_path=str(tmp_path / "nonexistent-bw"))
    assert bw.is_installed() is False
    with pytest.raises(BitwardenError) as ei:
        bw.unlock("some-master-password")
    msg = str(ei.value)
    assert "not installed" in msg
    assert "bitwarden.com/help/cli" in msg


def test_unlock_propagates_bw_error_text(tmp_path, monkeypatch):
    """If `bw` is on PATH but exits non-zero, surface its stderr to the user."""
    fake = tmp_path / "bw"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        "if [ \"$1\" = '--version' ]; then echo '2026.1.0'; exit 0; fi\n"
        "echo 'Invalid master password.' >&2\n"
        "exit 1\n"
    )
    fake.chmod(0o755)
    bw = BitwardenClient(bw_path=str(fake))
    assert bw.is_installed() is True
    with pytest.raises(BitwardenError, match="Invalid master password"):
        bw.unlock("wrong")
    assert bw.unlocked is False


def test_unlock_caches_session_token(tmp_path):
    fake = tmp_path / "bw"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        "if [ \"$1\" = '--version' ]; then echo '2026.1.0'; exit 0; fi\n"
        "if [ \"$1\" = 'unlock' ]; then echo 'TOKEN-abc123'; exit 0; fi\n"
        "exit 99\n"
    )
    fake.chmod(0o755)
    bw = BitwardenClient(bw_path=str(fake))
    bw.unlock("master")
    assert bw.unlocked is True
    assert bw._session == "TOKEN-abc123"
    bw.lock()
    assert bw.unlocked is False


def test_get_password_when_locked_raises(tmp_path):
    bw = BitwardenClient(bw_path=str(tmp_path / "bw"))
    with pytest.raises(BitwardenError, match="locked"):
        bw.get_password("any-item")
