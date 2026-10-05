"""The manual checkpoint mirror (:mod:`euclid_polish.web.fasrc_mirror`) pulls
the remote ensemble dir — and only it — into the local ensemble dir."""

from __future__ import annotations

import os

from euclid_polish.web import fasrc_config, fasrc_mirror


class _RecordingSSH:
    """Connected SSH stand-in that records every ``rsync_pull``."""

    def __init__(self) -> None:
        self.pulls: list[tuple[str, str, tuple[str, ...]]] = []

    def is_connected(self) -> bool:
        return True

    def rsync_pull(self, remote_path, local, extra_args=None, timeout=None):
        self.pulls.append((remote_path, local, tuple(extra_args or ())))
        return 0, "member_01/checkpoint", ""


class _State:
    def __init__(self, ssh) -> None:
        self.ssh = ssh


def _config(local_mirror: str) -> fasrc_config.FasrcConfig:
    return fasrc_config.FasrcConfig(
        ssh_user="t", repo_path="/n/repo", data_dir="/n/scratch/data",
        ckpt_dir="/n/scratch/ckpt/wdsr/", local_ckpt_mirror=local_mirror)


def test_remote_ensemble_dir_is_the_sibling_of_the_ckpt_dir():
    assert fasrc_mirror.remote_ensemble_dir(_config("")) == "/n/scratch/ckpt/ensemble"


def test_trigger_pulls_only_the_ensemble_dir(tmp_path, monkeypatch):
    local = tmp_path / "ensemble"
    monkeypatch.setattr(fasrc_config, "load", lambda: _config(str(local)))
    ssh = _RecordingSSH()
    monkeypatch.setattr(fasrc_mirror, "STATE", _State(ssh))

    status = fasrc_mirror.Mirror().trigger()

    assert ssh.pulls == [("/n/scratch/ckpt/ensemble/", str(local), ("--delete-after",))]
    assert status.last_rc == 0 and status.last_error == ""
    assert status.remote_dir == "/n/scratch/ckpt/ensemble/"
    assert status.local_dir == str(local)
    assert status.last_stdout == "member_01/checkpoint"
    # The local ensemble dir is created; no empty ``ensemble-vis`` sibling is.
    assert sorted(os.listdir(tmp_path)) == ["ensemble"]


def test_trigger_without_ssh_records_the_error_and_pulls_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(fasrc_config, "load", lambda: _config(str(tmp_path / "ensemble")))
    monkeypatch.setattr(fasrc_mirror, "STATE", _State(None))

    status = fasrc_mirror.Mirror().trigger()

    assert status.last_error == "ssh not connected"
    assert os.listdir(tmp_path) == []
