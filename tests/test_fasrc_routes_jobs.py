"""Live FASRC jobs in :mod:`euclid_polish.web.routes.fasrc`: ``POST
/api/fasrc/cancel`` takes a job or one array task, and the ``live`` rows of
``GET /api/fasrc/current-submission`` carry the step progress folded from each
running job's ``.events`` stream."""

from __future__ import annotations

import json
import subprocess
import time

import pytest

from euclid_polish.web import fasrc_config, fasrc_jobs, remote
from euclid_polish.web.app import create_app


class _LocalEventsSSH:
    """Connected SSH stand-in: canned ``squeue`` output; ``scancel`` is
    recorded; the events ``cat`` commands run in a local bash, so the event
    files are real files under ``tmp_path``."""

    def __init__(self, squeue: str = "", squeue_error: Exception | None = None) -> None:
        self.squeue = squeue
        self.squeue_error = squeue_error
        self.calls: list[str] = []

    def is_connected(self) -> bool:
        return True

    def run(self, cmd, timeout=None, binary=False):
        self.calls.append(cmd)
        if cmd.startswith("squeue"):
            if self.squeue_error is not None:
                raise self.squeue_error
            return 0, self.squeue, ""
        if cmd.startswith("scancel"):
            return 0, "", ""
        if "cat " in cmd:
            done = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                                  timeout=10, check=False)
            return done.returncode, done.stdout, done.stderr
        return 1, "", f"unexpected command: {cmd}"

    def event_reads(self) -> list[str]:
        return [c for c in self.calls if "cat " in c]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(fasrc_config, "CONFIG_PATH", str(tmp_path / "fasrc.json"))
    monkeypatch.setattr(fasrc_config, "CONFIG_DIR", str(tmp_path))
    fasrc_config.save(fasrc_config.FasrcConfig(
        ssh_user="t", repo_path="/n/repo", data_dir="/n/scratch/data",
        ckpt_dir="/n/scratch/ckpt/wdsr"))
    # The reconcile and the Jobstats poll are not under test here.
    monkeypatch.setattr(fasrc_jobs, "reconcile_with_squeue", lambda *_a, **_k: {})
    monkeypatch.setattr(fasrc_jobs, "fetch_live_jobstats", lambda *_a, **_k: None)
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _step(current: int, total: int) -> str:
    return json.dumps({"ts": 1.0, "kind": "step",
                       "value": {"current": current, "total": total, "label": f"step {current}"}}) + "\n"


def _job(jobid: str, state: str, *, events_path: str | None, params: dict | None = None) -> None:
    fasrc_jobs.DB.insert(jobid, label=f"job {jobid}", params=params or {}, script_path=".",
                         log_path=".", err_path=".", events_path=events_path)
    fasrc_jobs.DB.update_state(jobid, state=state)
    time.sleep(0.01)                     # distinct submitted_at → stable newest-first order


# --------------------------------------------------------------------------- cancel

@pytest.mark.parametrize("jobid", ["48107719", "48107719_3"])
def test_cancel_accepts_a_job_or_one_array_task(client, monkeypatch, jobid):
    ssh = _LocalEventsSSH()
    monkeypatch.setattr(remote.STATE, "ssh", ssh)
    response = client.post("/api/fasrc/cancel", data={"jobid": jobid})
    assert response.status_code == 200, response.get_json()
    assert ssh.calls == [f"scancel {jobid}"]


def test_cancelling_one_array_task_leaves_the_parent_job_live(client, monkeypatch):
    monkeypatch.setattr(remote.STATE, "ssh", _LocalEventsSSH())
    _job("48107719", "RUNNING", events_path=None, params={"array_count": 4})
    assert client.post("/api/fasrc/cancel", data={"jobid": "48107719_3"}).status_code == 200
    row = fasrc_jobs.DB.get("48107719")
    assert row["state"] == "RUNNING" and row["ended_at"] is None


def test_cancelling_the_whole_job_marks_it_cancelled(client, monkeypatch):
    monkeypatch.setattr(remote.STATE, "ssh", _LocalEventsSSH())
    _job("48107719", "RUNNING", events_path=None)
    assert client.post("/api/fasrc/cancel", data={"jobid": "48107719"}).status_code == 200
    row = fasrc_jobs.DB.get("48107719")
    assert row["state"] == "CANCELLED" and row["ended_at"] is not None


@pytest.mark.parametrize("jobid", [
    "", "abc", "123;id", "123 456", "$(id)", "`id`", "123_", "_4", "123_4_5",
    "123_[0-3]", "123_4;id", "123_4\nid", "12-3", "١٢٣", "²",
])
def test_cancel_refuses_anything_but_a_job_or_task_id(client, monkeypatch, jobid):
    ssh = _LocalEventsSSH()
    monkeypatch.setattr(remote.STATE, "ssh", ssh)
    response = client.post("/api/fasrc/cancel", data={"jobid": jobid})
    assert response.status_code == 400
    assert response.get_json()["error"] == "bad job id"
    assert ssh.calls == []


# --------------------------------------------------------------------------- live progress

def test_live_rows_carry_the_progress_of_their_event_streams(client, monkeypatch, tmp_path):
    (tmp_path / "701.events").write_text(_step(5000, 60000) + _step(10650, 70000))
    (tmp_path / "ens-702_0.events").write_text(_step(30000, 70000))
    (tmp_path / "ens-702_1.events").write_text(_step(12000, 70000))
    # task 702_2 has not started: no events file yet.
    _job("701", "RUNNING", events_path=str(tmp_path / "701.events"))
    _job("702", "RUNNING", events_path=str(tmp_path / "ens-%A_%a.events"),
         params={"array_count": 3, "member_names": "member_01,member_02,member_03"})
    _job("703", "PENDING", events_path=str(tmp_path / "703.events"))
    ssh = _LocalEventsSSH(
        "701|a|RUNNING|0:10|3:00:00|1|holy1|N/A\n"
        "702_0|b|RUNNING|0:10|3:00:00|1|holy2|N/A\n"
        "702_1|b|RUNNING|0:10|3:00:00|1|holy3|N/A\n"
        "702_2|b|PENDING|0:00|3:00:00|1|JobArrayTaskLimit|N/A\n"
        "703|c|PENDING|0:00|3:00:00|1|Priority|N/A\n")
    monkeypatch.setattr(remote.STATE, "ssh", ssh)

    body = client.get("/api/fasrc/current-submission").get_json()

    live = {row["jobid"]: row for row in body["live"]}
    # A single job: its latest step, as reported.
    assert (live["701"]["progress_step"], live["701"]["progress_total"]) == (10650, 70000)
    # An array: the mean task step (the unstarted task counts as 0).
    assert (live["702"]["progress_step"], live["702"]["progress_total"]) == (14000, 70000)
    # A queued job has no progress yet.
    assert (live["703"]["progress_step"], live["703"]["progress_total"]) == (None, None)
    assert body["current"]["job"]["jobid"] == "703"
    assert body["current"]["status"]["has_events"] is False
    # Every event stream is read in one SSH round-trip.
    assert len(ssh.event_reads()) == 1


def test_the_current_array_job_keeps_its_per_task_statuses(client, monkeypatch, tmp_path):
    (tmp_path / "ens-702_0.events").write_text(_step(30000, 70000))
    (tmp_path / "ens-702_1.events").write_text(_step(40000, 70000))
    _job("702", "RUNNING", events_path=str(tmp_path / "ens-%A_%a.events"),
         params={"array_count": 2, "member_names": "member_01,member_02"})
    ssh = _LocalEventsSSH("702_0|b|RUNNING|0:10|3:00:00|1|holy2|N/A\n"
                          "702_1|b|RUNNING|0:10|3:00:00|1|holy3|N/A\n")
    monkeypatch.setattr(remote.STATE, "ssh", ssh)

    body = client.get("/api/fasrc/current-submission").get_json()

    (row,) = body["live"]
    assert (row["progress_step"], row["progress_total"]) == (35000, 70000)
    tasks = body["current"]["array"]["tasks"]
    assert [t["member"] for t in tasks] == ["member_01", "member_02"]
    assert [t["status"]["step"]["current"] for t in tasks] == [30000, 40000]
    assert [t["state"] for t in tasks] == ["RUNNING", "RUNNING"]
    assert body["current"]["status"] is None
    assert len(ssh.event_reads()) == 1


def test_a_continue_batch_counts_each_task_against_its_own_total(client, monkeypatch, tmp_path):
    # Continued to one target, members run different step counts: member_01
    # finished its 20,000 extra steps, member_02 is halfway through its 10,000.
    (tmp_path / "ens-705_0.events").write_text(_step(20000, 20000))
    (tmp_path / "ens-705_1.events").write_text(_step(5000, 10000))
    _job("705", "RUNNING", events_path=str(tmp_path / "ens-%A_%a.events"),
         params={"array_count": 2, "mode": "continue", "members": "member_01,member_02"})
    monkeypatch.setattr(remote.STATE, "ssh", _LocalEventsSSH(
        "705_1|b|RUNNING|0:10|3:00:00|1|holy3|N/A\n"))

    body = client.get("/api/fasrc/current-submission").get_json()

    (row,) = body["live"]
    # 25,000 of 30,000 steps done (5/6), in member_01's 20,000 steps.
    assert (row["progress_step"], row["progress_total"]) == (16667, 20000)
    assert [t["member"] for t in body["current"]["array"]["tasks"]] == ["member_01", "member_02"]


def test_the_current_single_job_status_and_progress_agree(client, monkeypatch, tmp_path):
    (tmp_path / "801.events").write_text(_step(250, 1000))
    _job("801", "RUNNING", events_path=str(tmp_path / "801.events"))
    ssh = _LocalEventsSSH("801|a|RUNNING|0:10|1:00:00|1|holy1|N/A\n")
    monkeypatch.setattr(remote.STATE, "ssh", ssh)

    body = client.get("/api/fasrc/current-submission").get_json()

    (row,) = body["live"]
    assert (row["progress_step"], row["progress_total"]) == (250, 1000)
    assert body["current"]["status"]["step"] == {"current": 250, "total": 1000, "label": "step 250"}
    assert body["current"]["array"] is None
    assert len(ssh.event_reads()) == 1


def test_a_stale_tick_reports_no_progress(client, monkeypatch, tmp_path):
    (tmp_path / "901.events").write_text(_step(250, 1000))
    _job("901", "RUNNING", events_path=str(tmp_path / "901.events"))
    ssh = _LocalEventsSSH(squeue_error=subprocess.TimeoutExpired("squeue", 15))
    monkeypatch.setattr(remote.STATE, "ssh", ssh)

    body = client.get("/api/fasrc/current-submission").get_json()

    assert body["stale"] is True
    (row,) = body["live"]
    assert (row["progress_step"], row["progress_total"]) == (None, None)
    assert ssh.event_reads() == []
