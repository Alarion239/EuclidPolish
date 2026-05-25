"""Shared pytest fixtures and config for the EuclidPolish test suite."""

from __future__ import annotations

import os
import sys

# Make ``euclid_polish`` importable even when pytest is run from /tests.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ─── LOAD-BEARING SAFETY: disable the dev server's auto-SSH-connect. ───
#
# Tests routinely call ``create_app()`` to exercise web routes against a
# stubbed ``STATE.ssh``. Without this env var, ``create_app()`` would
# *silently* dial out to the developer's real FASRC via their pre-
# existing ControlMaster socket and overwrite the stub — which means
# every test posting to a submit endpoint would actually submit real
# SLURM jobs to real FASRC through the test author's own credentials.
#
# Setting the env var at module import time (BEFORE any test imports
# ``euclid_polish.web.app``) guarantees that ``_try_startup_ssh_connect``
# is a no-op for the entire pytest run. Tests that want to install
# their own stub via ``monkeypatch.setattr(remote.STATE, "ssh", stub)``
# now actually keep that stub in effect.
os.environ.setdefault("EUCLID_POLISH_DISABLE_AUTO_SSH", "1")


# ─── Session-wide harmless SSH stub on ``STATE.ssh``. ──────────────────
#
# Many existing tests GET pages that go through the ``_enforce_ssh_gate``
# ``before_request`` hook. Pre-fix, the auto-connect populated
# ``STATE.ssh`` with a real session so these tests sailed through. With
# the auto-connect now disabled, ``STATE.ssh`` is ``None`` and every
# such request 302-redirects to /connection-error — breaking ~60 tests
# that never asked for SSH at all.
#
# Solution: install a do-nothing SSH stub as the session default. It
# reports ``is_connected() == True`` and returns ``(0, "", "")`` for
# every ``.run(...)`` so the gate is satisfied without touching FASRC.
# The submit route would still fail (it tries to parse a sbatch jobid
# out of the empty string and 500s), so even if a future test bypasses
# the new arm/nonce guard, no real cluster work happens.
#
# Tests that need richer SSH behaviour (test_fasrc_pipeline, _logs,
# _integration, _fetcher) monkeypatch ``STATE.ssh`` to their own stub
# — those overrides win for the duration of the test, then pytest's
# monkeypatch reverts to this session-default no-op.
import pytest as _pytest


class _SessionNullSSH:
    """Pretends to be connected; .run returns (0, '', '') for everything.

    Never touches a network socket. Never writes a file. Never invokes
    sbatch. If a test's request handler tries to parse a real response
    out of these empty strings it'll fail with a clear ValueError or
    return 500 — both better outcomes than silently reaching FASRC.
    """

    def is_connected(self) -> bool:
        return True

    def run(self, cmd, timeout=None):
        return (0, "", "")

    def stream(self, cmd):
        # Empty iterator — matches the streaming contract from
        # ``LocalSSHSession`` / ``SSHSession`` without producing output.
        return iter(())

    def rsync_pull(self, *_a, **_kw):
        return (0, "", "")


@_pytest.fixture(autouse=True, scope="function")
def _safe_default_ssh_state(monkeypatch):
    """Set ``STATE.ssh`` to the no-op stub before each test, unless the
    test installs its own stub via ``monkeypatch.setattr`` (which wins
    by virtue of running later).

    Scoped to ``function`` so each test gets a fresh stub instance and
    nothing leaks between tests. ``monkeypatch`` automatically reverts
    at end-of-test, so the previous test's stub never bleeds in.
    """
    # Import inside the fixture so the import happens AFTER the env var
    # above has been set — otherwise importing ``app`` would still
    # trigger the auto-connect on the very first test.
    from euclid_polish.web import remote
    monkeypatch.setattr(remote.STATE, "ssh", _SessionNullSSH())
    monkeypatch.setattr(remote.STATE, "connected_at", 0.0)
    yield
