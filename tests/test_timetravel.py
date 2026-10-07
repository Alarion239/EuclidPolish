"""Tests for time-travel sandboxes (euclid_polish.tracking.timetravel)."""

from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import time

import pytest

from euclid_polish.config import Config
from euclid_polish.tracking import dirty_warning
from euclid_polish.tracking import timetravel as tt
from euclid_polish.web import app as web_app

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def _git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args],
                          capture_output=True, text=True, check=True)


@pytest.fixture
def repo(tmp_path):
    """A throwaway git repo with one commit; returns (repo_root, commit)."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(str(root), "init", "-q")
    _git(str(root), "config", "user.email", "t@t.t")
    _git(str(root), "config", "user.name", "t")
    (root / "code.py").write_text("VERSION = 1\n")
    _git(str(root), "add", "-A")
    _git(str(root), "commit", "-q", "-m", "v1")
    commit = _git(str(root), "rev-parse", "HEAD").stdout.strip()
    return str(root), commit


@pytest.fixture
def live_data(tmp_path):
    """A live data dir with an input subtree + a checkpoint to seed from."""
    d = tmp_path / "live_data"
    (d / "images" / "records_v2").mkdir(parents=True)
    (d / "images" / "records_v2" / "shard0.tfrecord").write_bytes(b"rec")
    (d / "euclid_psf").mkdir(parents=True)
    (d / "euclid_psf" / "euclid_psf_VIS.fits").write_bytes(b"psf")
    (d / "vis").mkdir()                       # an output dir — must NOT symlink
    ck = tmp_path / "ckpt_src"
    ck.mkdir()
    (ck / "checkpoint").write_text('model_checkpoint_path: "ckpt-5"\n')
    (ck / "ckpt-5.index").write_bytes(b"i")
    (ck / "meta.json").write_text("{}")       # tracking sidecar — must be skipped
    return str(d), str(ck)


@pytest.fixture(autouse=True)
def _tt_root(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "TIMETRAVEL_DIR", str(tmp_path / "tt"))


# The sandbox's "old code": a stand-in WebUI with the real server's shape (it
# listens on the port and keeps a child in its process group) that boots in
# milliseconds instead of importing TensorFlow. Pids land in the worktree, the
# server's cwd.
_FAKE_WEB_APP = '''
import os, socket, subprocess, sys, time

class _App:
    def run(self, host, port, **_kw):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
        with open("grandchild.pid", "w") as fh:
            fh.write(str(child.pid))
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen()
        while True:
            time.sleep(1)

def create_app():
    with open("server.pid", "w") as fh:
        fh.write(str(os.getpid()))
    return _App()
'''


@pytest.fixture
def fake_server_sandbox(repo, live_data):
    """A real sandbox whose worktree serves ``_FAKE_WEB_APP``."""
    root, commit = repo
    data_dir, _ = live_data
    meta = tt.prepare_local_sandbox(commit, repo_root=root, live_data_dir=data_dir)
    web = os.path.join(meta["worktree"], "euclid_polish", "web")
    os.makedirs(web)
    for package in (os.path.dirname(web), web):
        open(os.path.join(package, "__init__.py"), "w").close()
    with open(os.path.join(web, "app.py"), "w") as fh:
        fh.write(_FAKE_WEB_APP)
    yield meta
    # Whatever the code under test did, nothing it started outlives the test.
    for name in ("server.pid", "grandchild.pid"):
        with contextlib.suppress(OSError, ValueError):
            os.kill(int(open(os.path.join(meta["worktree"], name)).read()), signal.SIGKILL)


def _unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _pid_in(meta, name) -> int:
    return int(open(os.path.join(meta["worktree"], name)).read())


def _gone(pid: int, timeout: float = 5.0) -> bool:
    """True once ``pid`` no longer exists (an orphan waits on init's reaping)."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.kill(pid, 0)
        except OSError:
            return True
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)


# --------------------------------------------------------------------------
# dirty warning
# --------------------------------------------------------------------------

def test_dirty_warning_states():
    assert dirty_warning({"short": "abc", "dirty": False}) is None
    assert "uncommitted" in dirty_warning({"short": "abc", "dirty": True})
    assert "not a git repo" in dirty_warning(None).lower()


# --------------------------------------------------------------------------
# commit guard
# --------------------------------------------------------------------------

def test_commit_exists(repo):
    root, commit = repo
    assert tt.commit_exists(commit, root)
    assert not tt.commit_exists("0" * 40, root)


def test_prepare_rejects_unknown_commit(repo):
    root, _ = repo
    with pytest.raises(tt.TimeTravelError):
        tt.prepare_local_sandbox("0" * 40, repo_root=root)


# --------------------------------------------------------------------------
# local sandbox lifecycle
# --------------------------------------------------------------------------

def test_prepare_local_sandbox(repo, live_data):
    root, commit = repo
    data_dir, ckpt_src = live_data
    meta = tt.prepare_local_sandbox(
        commit, repo_root=root, live_data_dir=data_dir,
        seed_ckpt_dir=ckpt_src, source={"campaign": "x"})

    # worktree holds the old code at the commit
    assert os.path.isfile(os.path.join(meta["worktree"], "code.py"))

    # inputs symlinked, outputs NOT
    recs = os.path.join(meta["data_dir"], "images", "records_v2")
    assert os.path.islink(recs)
    assert os.path.isfile(os.path.join(recs, "shard0.tfrecord"))
    assert os.path.islink(os.path.join(meta["data_dir"], "euclid_psf"))
    assert not os.path.exists(os.path.join(meta["data_dir"], "vis"))
    assert "images/records_v2" in meta["linked_inputs"]

    # checkpoint seeded, meta.json sidecar skipped
    assert os.path.isfile(os.path.join(meta["ckpt_dir"], "checkpoint"))
    assert os.path.isfile(os.path.join(meta["ckpt_dir"], "ckpt-5.index"))
    assert not os.path.exists(os.path.join(meta["ckpt_dir"], "meta.json"))

    # appears in the listing
    shorts = [s["short"] for s in tt.list_sandboxes()]
    assert meta["short"] in shorts


def test_prepare_is_idempotent(repo, live_data):
    root, commit = repo
    data_dir, ckpt_src = live_data
    m1 = tt.prepare_local_sandbox(commit, repo_root=root, live_data_dir=data_dir)
    m2 = tt.prepare_local_sandbox(commit, repo_root=root, live_data_dir=data_dir)
    assert m2["reused"] is True
    assert m1["short"] == m2["short"]


def test_remove_sandbox(repo, live_data):
    root, commit = repo
    data_dir, _ = live_data
    meta = tt.prepare_local_sandbox(commit, repo_root=root, live_data_dir=data_dir)
    short = meta["short"]
    worktree = meta["worktree"]
    assert worktree in _git(root, "worktree", "list").stdout   # registered now
    tt.remove_sandbox(short, repo_root=root)
    assert not os.path.isdir(tt.sandbox_dir(short))
    # the git worktree registration (by path) is gone too
    assert worktree not in _git(root, "worktree", "list").stdout


def test_write_home_fasrc_config(repo, live_data):
    root, commit = repo
    data_dir, _ = live_data
    meta = tt.prepare_local_sandbox(commit, repo_root=root, live_data_dir=data_dir)
    path = tt.write_home_fasrc_config(meta["short"],
                                      {"ssh_user": "u", "data_dir": "/sandbox"})
    assert path.endswith(".euclid_polish/fasrc.json")
    assert path.startswith(meta["home"])
    import json
    assert json.load(open(path))["ssh_user"] == "u"


# --------------------------------------------------------------------------
# pure helpers
# --------------------------------------------------------------------------

def test_remote_path_helpers():
    assert tt.remote_worktree_path("/n/holy/EuclidPolish", "abc1234") == \
        "/n/holy/EuclidPolish-tt/abc1234"
    assert tt.remote_sandbox_base(
        "/n/netscratch/Lab/u/EuclidPolish/data", "abc1234") == \
        "/n/netscratch/Lab/u/EuclidPolish/sandbox/abc1234"


def test_pid_alive():
    assert tt._pid_alive(None) is False
    assert tt._pid_alive(os.getpid()) is True


# --------------------------------------------------------------------------
# remote sandbox over a stub ssh
# --------------------------------------------------------------------------

class _RecordingSSH:
    def __init__(self, missing_commit=False):
        self.cmds = []
        self.missing_commit = missing_commit

    def is_connected(self):
        return True

    def run(self, cmd, timeout=None):
        self.cmds.append(cmd)
        # Simulate "commit not present" on the first cat-file check.
        if "cat-file -e" in cmd and self.missing_commit:
            return (1, "", "missing")
        return (0, "", "")


def test_prepare_remote_sandbox_happy():
    ssh = _RecordingSSH()
    res = tt.prepare_remote_sandbox(
        ssh, repo_path="/n/holy/EuclidPolish",
        data_dir="/n/scratch/EuclidPolish/data", commit="abc123", short="abc123")
    assert res["ok"]
    assert res["worktree"] == "/n/holy/EuclidPolish-tt/abc123"
    assert res["data_dir"] == "/n/scratch/EuclidPolish/sandbox/abc123/data"
    joined = " ;; ".join(ssh.cmds)
    assert "worktree add --detach" in joined
    assert "ln -s" in joined          # input symlinks issued


def test_prepare_remote_sandbox_not_connected():
    class Down:
        def is_connected(self): return False
    res = tt.prepare_remote_sandbox(
        Down(), repo_path="/r", data_dir="/d", commit="c", short="s")
    assert not res["ok"] and "connected" in res["error"]


def test_prepare_remote_sandbox_unreachable_commit_no_push():
    ssh = _RecordingSSH(missing_commit=True)
    res = tt.prepare_remote_sandbox(
        ssh, repo_path="/r", data_dir="/d", commit="deadbeef", short="dead")
    assert not res["ok"]
    assert "not reachable on FASRC" in res["error"]


def test_the_sandbox_launcher_starts_the_background_services(monkeypatch):
    # The queue ticker only runs from start_background_services: a sandbox
    # served by a bare create_app().run() would never promote a persisted
    # queue after a restart.
    events: list[str] = []

    class _App:
        def run(self, **kwargs):
            events.append(f"run:{kwargs['port']}")

    monkeypatch.setattr(web_app, "create_app", lambda: _App())
    monkeypatch.setattr(web_app, "start_background_services",
                        lambda app: events.append("services"))
    exec(tt._SHIM.format(port=8799), {})
    assert events == ["services", "run:8799"]


def test_the_sandbox_launcher_serves_old_code_without_background_services(monkeypatch):
    events: list[str] = []

    class _App:
        def run(self, **kwargs):
            events.append("run")

    monkeypatch.setattr(web_app, "create_app", lambda: _App())
    monkeypatch.delattr(web_app, "start_background_services")
    exec(tt._SHIM.format(port=8799), {})
    assert events == ["run"]


# --------------------------------------------------------------------------
# spawn / stop the second WebUI: nothing it starts may outlive its stop
# --------------------------------------------------------------------------

def test_stop_server_takes_down_the_sandbox_server_and_its_children(fake_server_sandbox):
    short = fake_server_sandbox["short"]
    res = tt.spawn_server(short, port=_unused_port(), wait_s=20)
    assert res["ok"] and res["responding"], res
    server, child = res["pid"], _pid_in(fake_server_sandbox, "grandchild.pid")
    assert tt.list_sandboxes()[0]["running"] is True

    assert tt.stop_server(short) == {"ok": True}
    # Reaped, not a zombie that _pid_alive still counts as running.
    assert not tt._pid_alive(server)
    assert _gone(server, timeout=0)
    assert _gone(child)
    assert tt.read_sandbox(short)["pid"] is None
    assert tt.stop_spawned_servers() == []


def test_stop_spawned_servers_stops_a_server_nothing_else_tracks(fake_server_sandbox):
    """The suite's leak guard (tests/conftest.py) relies on this: it needs no
    sandbox.json, so it works after a test's tmp dir is gone."""
    short = fake_server_sandbox["short"]
    res = tt.spawn_server(short, port=_unused_port(), wait_s=20)
    assert res["ok"] and res["responding"], res
    child = _pid_in(fake_server_sandbox, "grandchild.pid")

    assert tt.stop_spawned_servers() == [res["pid"]]
    assert _gone(res["pid"], timeout=0)
    assert _gone(child)
    assert tt.stop_spawned_servers() == []


def test_a_spawn_that_fails_after_launch_stops_the_server(fake_server_sandbox, monkeypatch):
    """No pid would be recorded, so a server left running here would never be
    stopped by anything (Ctrl-C during the boot wait takes this path too)."""
    def disk_full(_path, _data):
        raise OSError("disk full")

    monkeypatch.setattr(tt, "_write_json", disk_full)
    with pytest.raises(OSError, match="disk full"):
        tt.spawn_server(fake_server_sandbox["short"], port=_unused_port(), wait_s=20)
    assert _gone(_pid_in(fake_server_sandbox, "server.pid"), timeout=0)
    assert _gone(_pid_in(fake_server_sandbox, "grandchild.pid"))
    assert tt.stop_spawned_servers() == []
