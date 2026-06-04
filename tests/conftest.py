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

    def rsync_push(self, *_a, **_kw):
        return (0, "", "")


@_pytest.fixture(autouse=True, scope="function")
def _redirect_writable_config_paths(monkeypatch, tmp_path_factory):
    """Redirect every ``Config.*`` path that test runs are known to
    write to → a per-test tmp directory.

    Background: ``_job_viz_psf`` writes ``psf_<band>.png`` to
    ``Config.VIS_PSF_DIR``; ``_job_viz_star_positions`` writes to
    ``Config.VIS_STAR_POSITIONS``. Tests like
    ``test_post_psfs_visualize_returns_job_id`` POST to the endpoint
    just to check that the HTTP response contains a ``job_id`` — but
    the background thread the endpoint spawns goes ahead and writes
    the PNG anyway, overwriting whatever the live pipeline last wrote.

    Forcing these to tmp paths per test means the live ``data/`` tree
    is never mutated by a unit test even when the production code
    hardcodes a real-path output. Tests that genuinely need to inspect
    the produced file should monkeypatch the path back to a known
    location.
    """
    from euclid_polish.config import Config
    pkg_tmp = tmp_path_factory.mktemp("writable_config_paths")
    vis_psf_dir = str(pkg_tmp / "vis_psf")
    os.makedirs(vis_psf_dir, exist_ok=True)
    monkeypatch.setattr(Config, "VIS_PSF_DIR", vis_psf_dir, raising=False)
    monkeypatch.setattr(
        Config, "VIS_STAR_POSITIONS",
        str(pkg_tmp / "star_positions.png"), raising=False,
    )
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
    yield
    # Drain background REGISTRY jobs a route spawned during the test (e.g.
    # /psfs/visualize writes psf_<band>.png to Config.VIS_PSF_DIR in a
    # daemon thread). They must finish while the redirects above are still
    # active — this fixture's teardown runs BEFORE its ``monkeypatch``
    # dependency reverts the paths — otherwise a late write lands in the
    # real ./data tree and trips the session-teardown immutability guard.
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


# ─── LOAD-BEARING SAFETY: real-data-dir files are immutable during tests ─
#
# Background: a unit test that wrote a 31×31 Gaussian stub to
# ``Config.EUCLID_PSF_DIR/euclid_psf_VIS.fits`` quietly replaced the
# user's real 1023² Euclid VIS ePSF on every ``pytest`` run. The user
# only noticed when the live pipeline started producing wrong results.
#
# This fixture takes a snapshot (path → SHA-256) of every FITS file
# under ``Config.DATA_DIR`` at session start, then on session teardown
# re-checks every snapshot path. If any file's contents changed, it
# (a) restores the original bytes from the snapshot when possible
# (kept in memory only for files < 16 MB so we don't blow RAM on big
# tile mosaics — we'll just fail loudly on those), and (b) raises so
# the test author sees the violation immediately. Side benefit: the
# restored bytes mean a failed run doesn't leave the working tree
# corrupted.
#
# A test that *legitimately* needs to write into a real-data path
# should ``tmp_path`` + monkeypatch the resolver instead. There's no
# valid reason for a test to mutate ``data/`` in-place.
@_pytest.fixture(scope="session", autouse=True)
def _protect_real_data_dir():
    import hashlib
    from euclid_polish.config import Config

    DATA_DIR = Config.DATA_DIR
    INLINE_RESTORE_LIMIT = 16 * 1024 * 1024   # 16 MB
    # Files at or above this size skip the SHA-256 round trip on both
    # snapshot and verify — they use ``(size, mtime_ns)`` as a tamper
    # check instead. Hashing the 10 GB COSMOS2025 catalog twice per
    # session was ~50 s of the suite runtime; mtime+size is good
    # enough for "did a test accidentally rewrite this file?" without
    # the I/O cost. Small load-bearing files (PSFs, kernels, summary
    # JSONs) — the ones the original incident actually touched —
    # stay under this threshold and keep the full SHA-256 check.
    HASH_SIZE_LIMIT = 32 * 1024 * 1024        # 32 MB

    def _snapshot(p: str, size: int) -> dict:
        if size >= HASH_SIZE_LIMIT:
            try:
                mtime_ns = os.stat(p).st_mtime_ns
            except FileNotFoundError:
                mtime_ns = 0
            return {"sha": None, "size": size, "mtime_ns": mtime_ns,
                    "blob": None}
        with open(p, "rb") as fh:
            blob = fh.read()
        return {
            "sha":      hashlib.sha256(blob).hexdigest(),
            "size":     size,
            "mtime_ns": None,
            # Keep contents in memory only for files small enough to
            # restore from RAM. Larger-than-inline files (16-32 MB)
            # still get hashed but can't be auto-restored.
            "blob":     blob if size <= INLINE_RESTORE_LIMIT else None,
        }

    snapshots: dict = {}
    if os.path.isdir(DATA_DIR):
        for root, dirs, files in os.walk(DATA_DIR):
            # Skip the FASRC rsync cache — it's transient by design.
            if "_fasrc_cache" in root.split(os.sep):
                continue
            for f in files:
                p = os.path.join(root, f)
                try:
                    st = os.stat(p)
                except FileNotFoundError:
                    continue
                snapshots[p] = _snapshot(p, st.st_size)

    yield

    violations = []
    for p, snap in snapshots.items():
        if not os.path.exists(p):
            violations.append(("deleted", p))
            continue
        try:
            st = os.stat(p)
        except FileNotFoundError:
            violations.append(("deleted", p))
            continue
        if snap["sha"] is None:
            # Big-file shortcut: only compare size + mtime. Cheap.
            if st.st_size != snap["size"] or st.st_mtime_ns != snap["mtime_ns"]:
                violations.append(
                    ("modified (large file, mtime+size diverged, cannot "
                     "restore)", p),
                )
            continue
        with open(p, "rb") as fh:
            cur = fh.read()
        if hashlib.sha256(cur).hexdigest() != snap["sha"]:
            if snap["blob"] is not None:
                with open(p, "wb") as fh:
                    fh.write(snap["blob"])
                violations.append(("modified (restored)", p))
            else:
                violations.append(("modified (cannot restore, > 16MB)", p))

    if violations:
        msg = [
            "",
            "─" * 72,
            " conftest session-teardown guard fired",
            "─" * 72,
            "",
            " IMPORTANT: this is NOT a failure of the test pytest blames",
            " it on. The check runs at SESSION teardown and pytest",
            " attributes session-fixture teardown errors to whichever",
            " test happened to run last. The real culprit is whatever",
            " production code path modified the file(s) listed below.",
            "",
            " Files mutated under Config.DATA_DIR during this run:",
            "",
        ]
        for kind, p in violations:
            msg.append(f"   [{kind}] {p}")
        msg.append("")
        msg.append(" Tests must NEVER write to real data paths. Use")
        msg.append(" ``tmp_path`` + monkeypatch the path resolver")
        msg.append(" (e.g. psf_path_for_band, Config.VIS_PSF_DIR)")
        msg.append(" so the production write lands under tmp_path.")
        msg.append("─" * 72)
        raise AssertionError("\n".join(msg))
