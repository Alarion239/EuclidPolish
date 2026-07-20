"""Shared pytest fixtures and config for the EuclidPolish test suite."""

from __future__ import annotations

import builtins
import io
import os
import shutil
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

    def rsync_push(self, *_a, **_kw):
        return (0, "", "")


@_pytest.fixture(autouse=True, scope="function")
def _forbid_real_data_writes(monkeypatch):
    """Fail before a test mutates anything under the live data directory.

    This is a prevention boundary, not a snapshot: it neither walks nor reads
    ``Config.DATA_DIR``.  Normal reads remain allowed, while Python file and
    filesystem APIs reject writes, deletes, renames, and directory creation
    below the real data root.  Tests must redirect writable paths to
    ``tmp_path`` (the shared writable-path fixture does this for production
    outputs used throughout the suite).
    """
    from euclid_polish.config import Config

    configured_data_dir = os.fspath(Config.DATA_DIR)
    real_data_roots = {
        os.path.realpath(os.path.abspath(configured_data_dir)),
    }
    if not os.path.isabs(configured_data_dir):
        # Also guard the checkout's data directory when pytest is launched
        # from a subdirectory and the relative Config path resolves elsewhere.
        real_data_roots.add(
            os.path.realpath(os.path.join(_PROJECT_ROOT, configured_data_dir))
        )

    def _inside_real_data(path) -> bool:
        if isinstance(path, int):
            return False
        try:
            candidate = os.path.realpath(os.path.abspath(os.fspath(path)))
            return any(
                os.path.commonpath((root, candidate)) == root
                for root in real_data_roots
            )
        except (TypeError, ValueError):
            return False

    def _reject(path, operation: str) -> None:
        if _inside_real_data(path):
            raise AssertionError(
                f"test attempted to {operation} live data path {os.fspath(path)!r}; "
                "redirect the output to tmp_path"
            )

    original_builtin_open = builtins.open
    original_io_open = io.open
    original_os_open = os.open

    def _guarded_builtin_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            _reject(file, f"open for mode {mode!r}")
        return original_builtin_open(file, mode, *args, **kwargs)

    def _guarded_io_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            _reject(file, f"open for mode {mode!r}")
        return original_io_open(file, mode, *args, **kwargs)

    write_flags = (
        os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    )

    def _guarded_os_open(path, flags, mode=0o777, *, dir_fd=None):
        if flags & write_flags:
            _reject(path, "open for writing")
        if dir_fd is None:
            return original_os_open(path, flags, mode)
        return original_os_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(builtins, "open", _guarded_builtin_open)
    monkeypatch.setattr(io, "open", _guarded_io_open)
    monkeypatch.setattr(os, "open", _guarded_os_open)

    def _guard_one(module, name: str, operation: str) -> None:
        original = getattr(module, name)

        def guarded(path, *args, **kwargs):
            # shutil's fd-safe rmtree implementation passes child names such
            # as ``data`` relative to an already-open temporary directory.
            # Interpreting those names against pytest's process cwd produces a
            # false match with the checkout's live ``./data`` tree.  The
            # absolute rmtree root was guarded before its dir_fd traversal.
            if kwargs.get("dir_fd") is None:
                _reject(path, operation)
            return original(path, *args, **kwargs)

        monkeypatch.setattr(module, name, guarded)

    def _guard_two(module, name: str, operation: str) -> None:
        original = getattr(module, name)

        def guarded(src, dst, *args, **kwargs):
            _reject(src, operation)
            _reject(dst, operation)
            return original(src, dst, *args, **kwargs)

        monkeypatch.setattr(module, name, guarded)

    for name in (
        "mkdir", "makedirs", "remove", "unlink", "rmdir", "removedirs",
        "truncate", "utime",
    ):
        _guard_one(os, name, name)
    for name in ("rename", "replace", "link"):
        _guard_two(os, name, name)
    for name in ("rmtree",):
        _guard_one(shutil, name, name)
    for name in ("move",):
        _guard_two(shutil, name, name)
    for name in ("copy", "copy2", "copyfile", "copytree"):
        original = getattr(shutil, name)

        def guarded_copy(src, dst, *args, _original=original, _name=name, **kwargs):
            _reject(dst, _name)
            return _original(src, dst, *args, **kwargs)

        monkeypatch.setattr(shutil, name, guarded_copy)

    yield


@_pytest.fixture(autouse=True, scope="function")
def _redirect_writable_config_paths(
    monkeypatch, tmp_path_factory, _forbid_real_data_writes,
):
    """Redirect every ``Config.*`` path that test runs are known to
    write to → a per-test tmp directory.

    Background: the CLI visualization commands write ``psf_<band>.png``
    to ``Config.VIS_PSF_DIR`` and a star-position plot to
    ``Config.VIS_STAR_POSITIONS``; web routes that spawn background jobs
    (e.g. inference reconstruct) can likewise write under ``data/`` from
    a daemon thread after the HTTP response has returned.

    Forcing these to tmp paths per test means the live ``data/`` tree
    is never mutated by a unit test even when the production code
    hardcodes a real-path output. Tests that genuinely need to inspect
    the produced file should monkeypatch the path back to a known
    location.
    """
    from euclid_polish.config import Config
    pkg_tmp = tmp_path_factory.mktemp("writable_config_paths")
    # Child Python processes construct Config afresh.  Point those processes
    # at the same temporary boundary instead of letting them inherit ./data.
    monkeypatch.setenv("EUCLID_POLISH_DATA_DIR", str(pkg_tmp / "data"))
    vis_psf_dir = str(pkg_tmp / "vis_psf")
    os.makedirs(vis_psf_dir, exist_ok=True)
    monkeypatch.setattr(Config, "VIS_PSF_DIR", vis_psf_dir, raising=False)
    monkeypatch.setattr(
        Config, "VIS_STAR_POSITIONS",
        str(pkg_tmp / "star_positions.png"), raising=False,
    )
    # The fasrc training-plot route renders ``tmp_training_plot.png`` into
    # ``Config.VIS_DIR``; without this redirect a test that exercises it
    # overwrites the live WebUI's copy under ./data/vis. Routes read that
    # Config path at request time, so writing AND serving (/vis/...) stay
    # consistent;
    # tests that need a specific VIS_DIR monkeypatch it themselves (their
    # setattr runs after this autouse fixture and wins).
    vis_dir = str(pkg_tmp / "vis")
    os.makedirs(vis_dir, exist_ok=True)
    monkeypatch.setattr(Config, "VIS_DIR", vis_dir, raising=False)
    # Isolate the experiment-tracking store: submit handlers now log every
    # FASRC job into the active campaign (or tracking/unassigned_*.jsonl),
    # which would otherwise create a real ./tracking folder in the repo.
    monkeypatch.setattr(
        Config, "TRACKING_DIR", str(pkg_tmp / "tracking"), raising=False,
    )
    # Same for time-travel sandboxes — never create a real git worktree
    # under ./.timetravel during a test run.
    monkeypatch.setattr(
        Config, "TIMETRAVEL_DIR", str(pkg_tmp / "timetravel"), raising=False,
    )

    # Isolate the FASRC job stores (sqlite DB + CSV job log) so NO test ever
    # writes into the real ``~/.euclid_polish/{fasrc_jobs.db,
    # fasrc_job_log.csv}``. The integration fixtures only patched ``DB``, so
    # every submit test was appending the fake-sbatch sentinel jobid 99999
    # into the user's real job log — and since ``/api/fasrc/submit`` defaults
    # the step to ``synthetic_generate``, those phantom rows flooded that
    # card's "previous runs" panel with never-finalising "pending" entries.
    # A fresh per-test store also means no run history leaks between tests.
    from euclid_polish.web import fasrc_jobs as _fasrc_jobs
    monkeypatch.setattr(
        _fasrc_jobs, "JOBLOG",
        _fasrc_jobs.JobLog(str(pkg_tmp / "fasrc_job_log.csv")),
        raising=False,
    )
    monkeypatch.setattr(
        _fasrc_jobs, "DB",
        _fasrc_jobs.JobDB(path=str(pkg_tmp / "fasrc_jobs.db")),
        raising=False,
    )
    # Isolate the local submission queue too — submit routes now route
    # through it, so without this a test submit would mutate the real
    # ``~/.euclid_polish/fasrc_queue.json``.
    from euclid_polish.web import fasrc_queue as _fasrc_queue
    monkeypatch.setattr(
        _fasrc_queue, "QUEUE",
        _fasrc_queue.JobQueue(path=str(pkg_tmp / "fasrc_queue.json")),
        raising=False,
    )
    yield
    # Drain background REGISTRY jobs a route spawned during the test (e.g.
    # an inference reconstruct that writes PNGs/FITS under ./data in a
    # daemon thread). They must finish while the redirects above are still
    # active — this fixture's teardown runs BEFORE its ``monkeypatch``
    # dependency reverts the paths — otherwise a late write lands in the
    # real ./data tree (where the prevention fixture would reject it).
    try:
        import time as _time

        from euclid_polish.web.jobs import REGISTRY
        _deadline = _time.monotonic() + 5.0
        while _time.monotonic() < _deadline:
            if not any(j.get("status") == "running" for j in REGISTRY.list()):
                break
            _time.sleep(0.02)
    except Exception:
        pass


@_pytest.fixture
def experimental_lanes_on(monkeypatch):
    """Enable the EXPERIMENTAL supervision lanes (HST / star-anchor /
    round-trip) for one test.

    Their WebUI surfaces are disabled by default (see
    ``euclid_polish.web.experimental``); route registration reads the
    flag at ``create_app()`` time, so request this fixture BEFORE
    building the app in tests that exercise the lane pages/steps."""
    from euclid_polish.web import experimental
    monkeypatch.setattr(experimental, "EXPERIMENTAL_LANES_ENABLED", True)
    yield


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
