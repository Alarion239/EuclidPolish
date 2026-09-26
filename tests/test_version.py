"""``GET /api/version`` (contract C3): boot commit vs HEAD, dirty, dist build."""

from __future__ import annotations

import datetime as dt
import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from euclid_polish.web import version
from euclid_polish.web.app import create_app
from euclid_polish.web.version import VersionTracker

KEYS = {"boot_commit", "boot_short", "head_commit", "head_short", "behind",
        "dirty", "started_at", "pid", "dist"}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
         "-c", "commit.gpgsign=false", *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "tracked.txt").write_text("one\n")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-q", "-m", "first")
    return root


@pytest.fixture
def dist_index(tmp_path):
    index = tmp_path / "dist" / "index.html"
    index.parent.mkdir()
    index.write_text('<div id="root"></div>')
    os.utime(index, (1_700_000_000, 1_700_000_000))
    return index


def test_boot_commit_is_captured_once_and_head_is_live(repo, dist_index):
    tracker = VersionTracker(repo=repo, dist_index=dist_index)
    first = _git(repo, "rev-parse", "HEAD")

    payload = tracker.payload()
    assert set(payload) == KEYS
    assert payload["boot_commit"] == payload["head_commit"] == first
    assert payload["boot_short"] == first[:7]
    assert payload["behind"] is False

    (repo / "tracked.txt").write_text("two\n")
    _git(repo, "commit", "-q", "-am", "second")
    second = _git(repo, "rev-parse", "HEAD")

    payload = tracker.payload()
    assert payload["boot_commit"] == first
    assert payload["head_commit"] == second
    assert payload["head_short"] == second[:7]
    assert payload["behind"] is True


def test_dirty_tracks_modified_tracked_files_only(repo, dist_index):
    tracker = VersionTracker(repo=repo, dist_index=dist_index)
    assert tracker.payload()["dirty"] is False

    (repo / "untracked.bin").write_bytes(b"data")
    assert tracker.payload()["dirty"] is False

    (repo / "tracked.txt").write_text("edited\n")
    assert tracker.payload()["dirty"] is True


def test_polling_never_rewrites_the_git_index(repo, dist_index):
    """``GET /api/version`` is polled by the SPA; a plain ``git status``
    refreshes a stale index in place (taking ``.git/index.lock``), which
    would make a concurrent ``git commit``/``git add`` fail with "index.lock:
    File exists". The probe must be a read-only ``--no-optional-locks``
    status."""
    tracker = VersionTracker(repo=repo, dist_index=dist_index)
    # Stale stat info for a clean file: exactly the case ``git status``
    # would "fix" by rewriting the index.
    os.utime(repo / "tracked.txt", (1_600_000_000, 1_600_000_000))
    index = repo / ".git" / "index"
    before = (index.read_bytes(), index.stat().st_mtime_ns)

    assert tracker.payload()["dirty"] is False

    assert (index.read_bytes(), index.stat().st_mtime_ns) == before


def test_every_git_probe_skips_optional_locks(monkeypatch, repo, dist_index):
    calls = []
    real_run = subprocess.run

    def spy(cmd, *args, **kwargs):
        calls.append((list(cmd), kwargs.get("env")))
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(version.subprocess, "run", spy)
    VersionTracker(repo=repo, dist_index=dist_index).payload()

    assert calls
    for cmd, env in calls:
        assert cmd[0] == "git"
        assert "--no-optional-locks" in cmd[:cmd.index("-C") + 3], cmd
        assert env is not None and env.get("GIT_OPTIONAL_LOCKS") == "0", cmd


def test_dist_build_is_described_by_mtime_and_hash(repo, dist_index):
    payload = VersionTracker(repo=repo, dist_index=dist_index).payload()["dist"]
    built = dt.datetime.fromisoformat(payload["built_at"])
    assert built.tzinfo is not None
    assert built.timestamp() == pytest.approx(1_700_000_000)
    expected = hashlib.sha256(dist_index.read_bytes()).hexdigest()[:16]
    assert payload["index_hash"] == expected


def test_missing_dist_and_non_git_dir_degrade_to_nulls(tmp_path):
    tracker = VersionTracker(repo=tmp_path, dist_index=tmp_path / "nope.html")
    payload = tracker.payload()
    assert payload["boot_commit"] is None and payload["head_commit"] is None
    assert payload["boot_short"] is None and payload["head_short"] is None
    assert payload["behind"] is False
    assert payload["dirty"] is False
    assert payload["dist"] == {"built_at": None, "index_hash": None}


def test_started_at_and_pid_describe_this_process(repo, dist_index):
    before = dt.datetime.now(dt.UTC)
    payload = VersionTracker(repo=repo, dist_index=dist_index).payload()
    started = dt.datetime.fromisoformat(payload["started_at"])
    assert abs((started - before).total_seconds()) < 5
    assert payload["pid"] == os.getpid()


def test_process_tracker_is_shared_and_points_at_this_checkout():
    tracker = version.process_tracker()
    assert tracker is version.process_tracker()
    assert tracker.repo == Path(version.__file__).resolve().parents[2]
    assert tracker.dist_index == (
        Path(version.__file__).resolve().parent / "static" / "dist" / "index.html"
    )


def test_version_route_serves_the_contract():
    client = create_app().test_client()
    response = client.get("/api/version")
    assert response.status_code == 200
    payload = response.get_json()
    assert set(payload) == KEYS
    assert set(payload["dist"]) == {"built_at", "index_hash"}
    assert isinstance(payload["behind"], bool)
    assert isinstance(payload["dirty"], bool)
    assert payload["pid"] == os.getpid()
