"""``GET /api/fasrc/steps/status`` probes each step's remote outputs; the
ensemble_train output is the ensemble's member checkpoints
(``<ckpt parent>/ensemble/member_NN/``), not the legacy single-model dir."""

from __future__ import annotations

import subprocess

import pytest

from euclid_polish.web import fasrc_config, remote
from euclid_polish.web.app import create_app


class _LocalBashSSH:
    """Connected SSH stand-in that runs every command in a local bash."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def is_connected(self) -> bool:
        return True

    def run(self, cmd, timeout=None, binary=False):
        self.calls.append(cmd)
        done = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                              timeout=10, check=False)
        return done.returncode, done.stdout, done.stderr


@pytest.fixture
def ckpt_root(tmp_path, monkeypatch):
    """A local stand-in for the FASRC tree: ``<root>/ckpt/wdsr`` is ckpt_dir."""
    root = tmp_path / "remote"
    (root / "ckpt" / "wdsr").mkdir(parents=True)
    (root / "data").mkdir()
    monkeypatch.setattr(fasrc_config, "CONFIG_PATH", str(tmp_path / "fasrc.json"))
    monkeypatch.setattr(fasrc_config, "CONFIG_DIR", str(tmp_path))
    fasrc_config.save(fasrc_config.FasrcConfig(
        ssh_user="t", repo_path=str(root / "repo"), data_dir=str(root / "data"),
        ckpt_dir=str(root / "ckpt" / "wdsr")))
    monkeypatch.setattr(remote.STATE, "ssh", _LocalBashSSH())
    return root


def _ensemble_output(root):
    app = create_app()
    app.config["TESTING"] = True
    body = app.test_client().get("/api/fasrc/steps/status").get_json()
    steps = {s["step_id"]: s for s in body["steps"]}
    return steps["ensemble_train"]["outputs"], body["artifacts"]


def test_ensemble_train_output_is_found_in_a_member_dir(ckpt_root):
    member = ckpt_root / "ckpt" / "ensemble" / "member_03"
    member.mkdir(parents=True)
    (member / "checkpoint").write_text("model_checkpoint_path: \"ckpt-7\"\n")

    outputs, artifacts = _ensemble_output(ckpt_root)

    assert outputs == [{"key": "ckpt", "path": str(ckpt_root / "ckpt" / "ensemble"), "exists": True}]
    assert artifacts["ckpt"] is True


def test_a_legacy_single_model_checkpoint_does_not_count(ckpt_root):
    (ckpt_root / "ckpt" / "wdsr" / "checkpoint").write_text("legacy\n")
    (ckpt_root / "ckpt" / "ensemble" / "member_01").mkdir(parents=True)   # no checkpoint yet

    outputs, artifacts = _ensemble_output(ckpt_root)

    assert outputs == [{"key": "ckpt", "path": str(ckpt_root / "ckpt" / "ensemble"), "exists": False}]
    assert artifacts["ckpt"] is False
    # The other probes still run in the same round-trip.
    assert artifacts["euclid_cutouts"] is False and artifacts["synthetic_records"] is False


def test_the_probe_quotes_a_ckpt_dir_with_shell_metacharacters(tmp_path, monkeypatch):
    root = tmp_path / "remote dir;$(touch pwned)"
    member = root / "ckpt" / "ensemble" / "member_01"
    member.mkdir(parents=True)
    (member / "checkpoint").write_text("ok\n")
    monkeypatch.setattr(fasrc_config, "CONFIG_PATH", str(tmp_path / "fasrc.json"))
    monkeypatch.setattr(fasrc_config, "CONFIG_DIR", str(tmp_path))
    fasrc_config.save(fasrc_config.FasrcConfig(
        ssh_user="t", repo_path=str(root / "repo"), data_dir=str(root / "data"),
        ckpt_dir=str(root / "ckpt" / "wdsr")))
    monkeypatch.setattr(remote.STATE, "ssh", _LocalBashSSH())
    monkeypatch.chdir(tmp_path)

    _outputs, artifacts = _ensemble_output(root)

    assert artifacts["ckpt"] is True
    assert not (tmp_path / "pwned").exists()
