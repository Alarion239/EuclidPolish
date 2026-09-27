"""``GET /api/version`` (contract C3): boot commit vs HEAD, dirty, dist build,
and ``behind`` = a backend ``.py`` file the server loaded changed on disk."""

from __future__ import annotations

import datetime as dt
import hashlib
import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

from euclid_polish.web import version
from euclid_polish.web.app import create_app
from euclid_polish.web.version import BackendSources, VersionTracker

KEYS = {"boot_commit", "boot_short", "head_commit", "head_short", "behind",
        "changed_files", "changed_count", "changed_digest", "dirty", "started_at", "pid", "dist"}


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


class Clock:
    """A settable monotonic clock for the ~10 s cache of the file check."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def pkg(repo):
    """A tiny backend package inside the repo: two loaded modules, one the
    server never imported, and the frontend/static trees."""
    root = repo / "euclid_polish"
    (root / "web" / "routes").mkdir(parents=True)
    (root / "web" / "frontend").mkdir()
    (root / "web" / "static").mkdir()
    files = {
        "app": root / "web" / "app.py",
        "route": root / "web" / "routes" / "real.py",
        "unused": root / "training.py",
        "frontend": root / "web" / "frontend" / "tool.py",
        "static": root / "web" / "static" / "gen.py",
    }
    for name, path in files.items():
        path.write_text(f"# {name}\nVALUE = 1\n")
        os.utime(path, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "package")
    return files


def _sources(repo, files, clock, *names):
    loaded = [files[n] for n in names]
    return BackendSources(repo / "euclid_polish", rel_to=repo,
                          loaded=lambda: list(loaded), clock=clock)


def _edit(path: Path, text: str) -> None:
    path.write_text(text)
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))


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


def test_committing_the_code_the_server_already_runs_is_not_behind(repo, dist_index, pkg):
    """The false positive of the commit-id check: the server booted with the
    edited files already on disk and the commit landed afterwards."""
    clock = Clock()
    _edit(pkg["route"], "# route\nVALUE = 2\n")          # edited before boot
    tracker = VersionTracker(repo=repo, dist_index=dist_index,
                             sources=_sources(repo, pkg, clock, "app", "route"))
    _git(repo, "commit", "-q", "-am", "commit what the server runs")
    clock.now += 60

    payload = tracker.payload()
    assert payload["head_commit"] != payload["boot_commit"]
    assert payload["behind"] is False
    assert payload["changed_files"] == []
    assert payload["changed_count"] == 0


def test_a_loaded_backend_file_edited_after_boot_is_behind(repo, dist_index, pkg):
    clock = Clock()
    tracker = VersionTracker(repo=repo, dist_index=dist_index,
                             sources=_sources(repo, pkg, clock, "app", "route"))
    assert tracker.payload()["behind"] is False

    _edit(pkg["route"], "# route\nVALUE = 3\n")
    clock.now += 60
    payload = tracker.payload()
    assert payload["behind"] is True
    assert payload["changed_files"] == ["euclid_polish/web/routes/real.py"]
    assert payload["changed_count"] == 1
    # Uncommitted or committed makes no difference: the running code is older.
    _git(repo, "commit", "-q", "-am", "route edit")
    clock.now += 60
    assert tracker.payload()["behind"] is True


def test_changed_files_are_newest_first_and_capped(repo, pkg):
    clock = Clock()
    root = repo / "euclid_polish"
    many = [root / f"mod{i:02d}.py" for i in range(12)]
    for path in many:
        path.write_text("X = 0\n")
    sources = BackendSources(root, rel_to=repo, loaded=lambda: list(many), clock=clock)
    for i, path in enumerate(many):
        path.write_text(f"X = {i + 1}\n")
        os.utime(path, ns=(0, 1_800_000_000_000_000_000 + i * 1_000_000_000))
    clock.now += 60
    changed = sources.changed()
    assert len(changed) == 12
    assert changed[0] == "euclid_polish/mod11.py"
    assert changed[-1] == "euclid_polish/mod00.py"

    tracker = VersionTracker(repo=repo, dist_index=repo / "none.html", sources=sources)
    payload = tracker.payload()
    assert payload["changed_count"] == 12
    assert payload["changed_files"] == changed[:version.MAX_CHANGED_LISTED]
    assert len(payload["changed_files"]) == version.MAX_CHANGED_LISTED < 12


def test_files_the_server_never_loaded_and_frontend_static_are_ignored(repo, dist_index, pkg):
    clock = Clock()
    sources = _sources(repo, pkg, clock, "app", "frontend", "static")
    for name in ("unused", "frontend", "static"):
        _edit(pkg[name], f"# {name}\nVALUE = 9\n")
    clock.now += 60
    assert sources.changed() == []


def test_same_content_with_a_new_mtime_is_not_a_change(repo, pkg):
    """A checkout, stash pop or `touch` rewrites the mtime but not the code."""
    clock = Clock()
    sources = _sources(repo, pkg, clock, "app")
    original = pkg["app"].read_text()
    _edit(pkg["app"], original)
    clock.now += 60
    assert sources.changed() == []
    # Edited, then edited back: the loaded code is what is on disk again.
    _edit(pkg["app"], "# app\nVALUE = 5\n")
    clock.now += 60
    assert sources.changed() == ["euclid_polish/web/app.py"]
    _edit(pkg["app"], original)
    clock.now += 60
    assert sources.changed() == []


def test_a_deleted_loaded_file_is_a_change(repo, pkg):
    clock = Clock()
    sources = _sources(repo, pkg, clock, "app", "route")
    pkg["route"].unlink()
    clock.now += 60
    assert sources.changed() == ["euclid_polish/web/routes/real.py"]


def test_a_module_first_seen_after_boot_is_baselined_then(repo, pkg):
    """A module imported lazily after boot was loaded from the file on disk at
    that time; only later edits count."""
    clock = Clock()
    loaded = [pkg["app"]]
    sources = BackendSources(repo / "euclid_polish", rel_to=repo, loaded=lambda: list(loaded), clock=clock)
    loaded.append(pkg["route"])
    clock.now += 60
    assert sources.changed() == []
    _edit(pkg["route"], "# route\nVALUE = 7\n")
    clock.now += 60
    assert sources.changed() == ["euclid_polish/web/routes/real.py"]


def test_the_file_check_is_cached_for_a_few_seconds(repo, pkg):
    clock = Clock()
    calls = []
    loaded = [pkg["app"]]

    def listing():
        calls.append(clock.now)
        return list(loaded)

    sources = BackendSources(repo / "euclid_polish", rel_to=repo, loaded=listing, clock=clock)
    boot_calls = len(calls)
    _edit(pkg["app"], "# app\nVALUE = 8\n")
    assert sources.changed() == ["euclid_polish/web/app.py"]
    first = len(calls)
    clock.now += version.CHANGED_TTL_S / 2
    assert sources.changed() == ["euclid_polish/web/app.py"]
    assert len(calls) == first                  # served from the cache
    clock.now += version.CHANGED_TTL_S
    sources.changed()
    assert len(calls) == first + 1
    assert first == boot_calls + 1


def test_default_sources_are_the_backend_modules_in_sys_modules():
    files = version.loaded_backend_files(version.PACKAGE_ROOT)
    assert files, "the web package itself is imported"
    as_text = {str(p) for p in files}
    assert str(Path(version.__file__).resolve()) in as_text
    for path in files:
        assert path.suffix == ".py"
        assert path.is_relative_to(version.PACKAGE_ROOT)
        assert not path.is_relative_to(version.PACKAGE_ROOT / "web" / "frontend")
        assert not path.is_relative_to(version.PACKAGE_ROOT / "web" / "static")
    # Nothing outside the package (the standard library, site-packages).
    assert str(Path(sys.modules["json"].__file__).resolve()) not in as_text


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
    assert payload["changed_files"] == [] and payload["changed_count"] == 0
    assert payload["dirty"] is False
    assert payload["dist"] == {"built_at": None, "index_hash": None, "entry": None}


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
    assert tracker.sources.root == Path(version.__file__).resolve().parents[1]


def test_version_route_serves_the_contract():
    client = create_app().test_client()
    response = client.get("/api/version")
    assert response.status_code == 200
    payload = response.get_json()
    assert set(payload) == KEYS
    assert set(payload["dist"]) == {"built_at", "index_hash", "entry"}
    assert isinstance(payload["behind"], bool)
    assert isinstance(payload["changed_files"], list)
    assert payload["changed_count"] >= len(payload["changed_files"])
    assert payload["behind"] is (payload["changed_count"] > 0)
    assert isinstance(payload["dirty"], bool)
    assert payload["pid"] == os.getpid()


def test_changed_digest_names_the_whole_changed_set(repo, pkg):
    """The banner's dismissal key: stable while the SET of changed files is
    the same, whatever subset the (newest-first, capped) list shows."""
    clock = Clock()
    root = repo / "euclid_polish"
    many = [root / f"mod{i:02d}.py" for i in range(12)]
    for path in many:
        path.write_text("X = 0\n")
    sources = BackendSources(root, rel_to=repo, loaded=lambda: list(many), clock=clock)
    tracker = VersionTracker(repo=repo, dist_index=repo / "none.html", sources=sources)
    assert tracker.payload()["changed_digest"] is None
    for i, path in enumerate(many):
        path.write_text(f"X = {i + 1}\n")
        os.utime(path, ns=(0, 1_800_000_000_000_000_000 + i * 1_000_000_000))
    clock.now += 60
    first = tracker.payload()
    assert isinstance(first["changed_digest"], str) and first["changed_digest"]
    # Re-save the oldest changed file: it moves into the listed newest 8, the
    # set of changed files is the same.
    many[0].write_text("X = 100\n")
    os.utime(many[0], ns=(0, 1_900_000_000_000_000_000))
    clock.now += 60
    second = tracker.payload()
    assert second["changed_files"] != first["changed_files"]
    assert second["changed_digest"] == first["changed_digest"]
    # One more changed file is a new set.
    extra = root / "extra.py"
    extra.write_text("X = 0\n")
    many.append(extra)
    clock.now += 60
    tracker.payload()                       # baselined
    _edit(extra, "X = 1\n")
    clock.now += 60
    assert tracker.payload()["changed_digest"] != first["changed_digest"]


def test_a_module_edited_between_its_import_and_first_scan_is_changed(repo, pkg, monkeypatch):
    """A lazily imported module is checked against the source stamp its
    bytecode was compiled from (the .pyc header), so an edit made after the
    import but before the tracker first saw the module is not missed."""
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    clock = Clock()
    loaded = [pkg["app"]]
    sources = BackendSources(repo / "euclid_polish", rel_to=repo, loaded=lambda: list(loaded), clock=clock)
    py_compile.compile(str(pkg["route"]), doraise=True)      # "imported" now
    _edit(pkg["route"], "# route\nVALUE = 70\n")             # edited before the next scan
    loaded.append(pkg["route"])
    clock.now += 60
    assert sources.changed() == ["euclid_polish/web/routes/real.py"]


def test_a_first_seen_module_whose_pyc_matches_is_baselined(repo, pkg, monkeypatch):
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    clock = Clock()
    loaded = [pkg["app"]]
    sources = BackendSources(repo / "euclid_polish", rel_to=repo, loaded=lambda: list(loaded), clock=clock)
    py_compile.compile(str(pkg["route"]), doraise=True)
    loaded.append(pkg["route"])
    clock.now += 60
    assert sources.changed() == []


def test_the_pyc_stamp_is_ignored_when_python_writes_no_bytecode(repo, pkg, monkeypatch):
    """Without bytecode writes a stale .pyc says nothing about what was loaded."""
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    clock = Clock()
    loaded = [pkg["app"]]
    sources = BackendSources(repo / "euclid_polish", rel_to=repo, loaded=lambda: list(loaded), clock=clock)
    py_compile.compile(str(pkg["route"]), doraise=True)
    _edit(pkg["route"], "# route\nVALUE = 71\n")
    loaded.append(pkg["route"])
    clock.now += 60
    assert sources.changed() == []


def test_dist_names_the_entry_script_of_the_build(repo, tmp_path):
    index = tmp_path / "built" / "index.html"
    index.parent.mkdir()
    assets = "/static/dist" + "/assets"   # split: pytest never names bundle assets
    index.write_text(
        '<script>inline()</script>\n'
        f'<script type="module" crossorigin src="{assets}/index-AbC_12.js"></script>\n'
        f'<link rel="modulepreload" crossorigin href="{assets}/react-x.js">\n')
    dist = VersionTracker(repo=repo, dist_index=index).payload()["dist"]
    assert dist["entry"] == f"{assets}/index-AbC_12.js"
