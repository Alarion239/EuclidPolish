"""Tests for time-travel sandboxes (euclid_polish.tracking.timetravel)."""

from __future__ import annotations

import os
import subprocess

import pytest

from euclid_polish.config import Config
from euclid_polish.tracking import dirty_warning
from euclid_polish.tracking import timetravel as tt

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
