"""The manual checkpoint mirror (:mod:`euclid_polish.web.fasrc_mirror`) pulls
the remote ensemble dir — and only it — into the local ensemble dir."""

from __future__ import annotations

import json
import os

from euclid_polish import ensemble_registry
from euclid_polish.config import Config
from euclid_polish.observability import JobLog, JobRecord
from euclid_polish.web import fasrc_config, fasrc_jobs, fasrc_mirror


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


def test_trigger_excludes_universal_models_the_registry_does_not_know(tmp_path, monkeypatch):
    """The mirror would bootstrap every unknown member dir into the ENSEMBLE:
    universal models the job log created on FASRC that are neither active
    nor registered universal (nor archived, nor read by a local gate) are
    excluded; registered ones, ordinary members and ensemble members a gate
    reads (195, its dir gone) are mirrored."""
    local = tmp_path / "ensemble"
    monkeypatch.setattr(Config, "VIS_DIR", str(tmp_path / "vis"))
    gate = tmp_path / "vis" / "ensemble" / "starfull" / "spatial_gate_combiner"
    gate.mkdir(parents=True)
    (gate / "combiner.json").write_text(json.dumps({"member_labels": ["195·psnr"]}))
    ensemble_registry.add_universal_entry(str(local), "211")
    monkeypatch.setattr(fasrc_jobs, "JOBLOG", JobLog(str(tmp_path / "joblog.csv")))
    spec = {"asinh_knees": [0.1, 1, 10], "output_knee": 10, "learn_output_knee": True}
    for jobid, name, member in (("1", "member_211", spec), ("2", "member_218", spec),
                                ("3", "member_219", {"asinh_knee": 3.0}),
                                ("4", "member_195", spec)):
        fasrc_jobs.JOBLOG.record_submission(JobRecord(
            jobid=jobid, step_id="ensemble_train", submitted_at=f"2026-10-1{jobid}T00:00:00Z",
            params_json=json.dumps({"mode": "add", "member_names": name,
                                    "member_spec": json.dumps([member])})))
    monkeypatch.setattr(fasrc_config, "load", lambda: _config(str(local)))
    ssh = _RecordingSSH()
    monkeypatch.setattr(fasrc_mirror, "STATE", _State(ssh))

    fasrc_mirror.Mirror().trigger()

    assert ssh.pulls == [("/n/scratch/ckpt/ensemble/", str(local),
                          ("--delete-after", "--exclude=/member_218/"))]
