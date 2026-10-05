"""``scripts/timetravel.py restore``: model / retired-zip / --remote paths.

The sandbox, the second server and SSH are all stubbed, so nothing touches
git worktrees, FASRC or the network.
"""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import pytest

from euclid_polish.tracking.store import TrackingStore
from scripts import timetravel as cli

_COMMIT = "0123456789abcdef0123456789abcdef01234567"
_SHORT = "0123456"
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A tracking store with one dir model backup and one retired-model zip."""
    s = TrackingStore(str(tmp_path / "tracking"), repo_root=str(tmp_path))
    s.create_campaign("tt script")
    models = os.path.join(s.current_dir, "models")
    commit = {"hash": _COMMIT, "short": _SHORT, "dirty": False}
    os.makedirs(os.path.join(models, "live-member"))
    with open(os.path.join(models, "live-member", "meta.json"), "w") as fh:
        json.dump({"commit": commit}, fh)
    with open(os.path.join(models, "old-member.zip"), "wb") as fh:
        fh.write(b"PK")
    with open(os.path.join(models, "old-member.zip.meta.json"), "w") as fh:
        json.dump({"commit": commit, "kind": "model-zip"}, fh)
    monkeypatch.setattr(cli, "default_store", lambda: s)
    return s


@pytest.fixture
def calls(monkeypatch):
    """Stub every side effect of ``_restore``; record what it was called with."""
    seen: dict = {}
    cfg = SimpleNamespace(
        data_dir="/remote/data", repo_path="/remote/repo",
        ssh_user="u", ssh_host="h", control_socket="/tmp/s.sock",
        control_persist="10m", to_dict=lambda: {"repo_path": "/remote/repo"})
    monkeypatch.setattr(cli.fasrc_config, "load", lambda: cfg)

    def fake_prepare_local(commit, **kw):
        seen["local"] = {"commit": commit, **kw}
        return {"short": _SHORT}

    def fake_prepare_remote(ssh, **kw):
        seen["remote"] = kw
        return {"ok": True}

    monkeypatch.setattr(cli.tt, "prepare_local_sandbox", fake_prepare_local)
    monkeypatch.setattr(cli.tt, "prepare_remote_sandbox", fake_prepare_remote)
    monkeypatch.setattr(cli.tt, "write_home_fasrc_config", lambda short, d: None)
    monkeypatch.setattr(cli.tt, "set_sandbox_remote",
                        lambda short, remote: seen.setdefault("set_remote", remote))
    monkeypatch.setattr(cli.tt, "read_sandbox", lambda short: {"short": short})

    def no_spawn(short):
        raise AssertionError("--no-open must not spawn a server")

    monkeypatch.setattr(cli.tt, "spawn_server", no_spawn)

    class FakeSSH:
        def __init__(self, cfg):
            seen["ssh_cfg"] = cfg

        def is_connected(self):
            return False

        def connect(self):
            seen["connected"] = True

    monkeypatch.setattr(cli, "SSHSession", FakeSSH)
    return seen


def _args(**kw) -> argparse.Namespace:
    base = {"campaign": "current", "model": "", "remote": False, "no_open": True}
    base.update(kw)
    return argparse.Namespace(**base)


def test_restore_model_dir_seeds_its_checkpoint(store, calls):
    assert cli._restore(_args(model="live-member")) == 0
    assert calls["local"]["commit"] == _COMMIT
    assert calls["local"]["seed_ckpt_dir"] == store.model_backup_dir("current", "live-member")


def test_restore_retired_model_zip_restores_commit_without_seed(store, calls, capsys):
    assert cli._restore(_args(model="old-member.zip")) == 0, capsys.readouterr().err
    assert calls["local"]["commit"] == _COMMIT
    assert calls["local"]["seed_ckpt_dir"] is None
    assert calls["local"]["source"] == {"campaign": "current", "model": "old-member.zip"}


def test_restore_remote_pushes_commit_from_this_repo(store, calls, capsys):
    assert cli._restore(_args(model="live-member", remote=True)) == 0, capsys.readouterr().err
    assert calls.get("connected") is True
    assert calls["remote"]["push_origin_cmd"] == [
        "git", "-C", _REPO_ROOT, "push", "origin",
        f"{_COMMIT}:refs/heads/timetravel/{_SHORT}"]
    assert calls["remote"]["commit"] == _COMMIT
    assert calls["remote"]["short"] == _SHORT
    assert calls["set_remote"] == {"ok": True}
