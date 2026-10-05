"""FASRC routes added by W-Ops (the former Ops › FASRC tab): history across
steps, queue detail and resume, the accounting-reconcile and checkpoint-mirror
jobs, the remote file browser, log search, the FASRC-vs-local HEAD comparison
and step outputs."""

from __future__ import annotations

import json
import time

import pytest

from euclid_polish.observability import JobRecord
from euclid_polish.web import fasrc_config, fasrc_jobs, fasrc_mirror, fasrc_queue, git_ops, remote
from euclid_polish.web.app import create_app
from euclid_polish.web.jobs import REGISTRY


class ScriptedSSH:
    """Connected SSH stand-in answering ``run`` by command substring."""

    def __init__(self, responses=None):
        self.responses = dict(responses or {})
        self.calls: list[str] = []
        self.pulls: list[tuple] = []

    def is_connected(self):
        return True

    def run(self, cmd, timeout=None):
        self.calls.append(cmd)
        for needle, resp in self.responses.items():
            if needle in cmd:
                return resp
        return (0, "", "")

    def rsync_pull(self, remote_path, local, extra_args=None, timeout=None):
        self.pulls.append((remote_path, local, tuple(extra_args or ())))
        return (0, "sent 1 file", "")


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(fasrc_config, "CONFIG_PATH", str(tmp_path / "fasrc.json"))
    monkeypatch.setattr(fasrc_config, "CONFIG_DIR", str(tmp_path))
    c = fasrc_config.FasrcConfig(ssh_user="t", repo_path="/n/repo", data_dir="/n/scratch/data",
                                 ckpt_dir="/n/scratch/ckpt/wdsr",
                                 local_ckpt_mirror=str(tmp_path / "mirror"))
    fasrc_config.save(c)
    return c


@pytest.fixture
def client(cfg):
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _wait(job_id, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = REGISTRY.get(job_id)
        if job is not None and job.to_dict()["status"] != "running":
            return job.to_dict()
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish")


def _record(jobid, *, step, state="", db_state=None, params=None, submitted="2026-09-01T00:00:00Z"):
    fasrc_jobs.JOBLOG.record_submission(JobRecord(
        jobid=jobid, step_id=step, label=f"{step} run", submitted_at=submitted,
        params_json=json.dumps(params or {}), req_cpus=4, req_memory="8G",
        req_time_limit="1:00:00", partition="shared"))
    if state:
        fasrc_jobs.JOBLOG.record_post_mortem(jobid, {"state": state})
    if db_state:
        fasrc_jobs.DB.insert(jobid, label=f"{step} run", params=params or {},
                             script_path="/s", log_path="/o", err_path="/e")
        fasrc_jobs.DB.set_step_id(jobid, step)
        fasrc_jobs.DB.update_state(jobid, state=db_state)


# --------------------------------------------------------------------------- history

def test_history_lists_every_step_newest_first_with_compact_params(client):
    blob = "x" * 5000
    _record("1", step="euclid_query", state="COMPLETED", submitted="2026-09-01T00:00:00Z",
            params={"num_stars": "10000"})
    _record("2", step="synthetic_generate", submitted="2026-09-02T00:00:00Z", db_state="UNKNOWN",
            params={"force": "0", "_star_prior_json": blob})
    _record("3", step="euclid_query", submitted="2026-09-03T00:00:00Z", db_state="RUNNING")
    r = client.get("/api/fasrc/history")
    assert r.status_code == 200
    body = r.get_json()
    assert [row["jobid"] for row in body["rows"]] == ["3", "2", "1"]
    row2 = body["rows"][1]
    assert row2["params"] == {"force": "0"}
    assert row2["params_omitted"] == {"_star_prior_json": 5000}
    assert blob not in r.get_data(as_text=True)
    # The display state falls back to the live DB state while sacct is silent.
    assert [row["state_display"] for row in body["rows"]] == ["RUNNING", "UNKNOWN", "COMPLETED"]
    assert body["facets"]["steps"] == {"euclid_query": 2, "synthetic_generate": 1}
    assert body["unresolved"] == 1


def test_history_filters_by_step_state_and_text(client):
    _record("1", step="euclid_query", state="COMPLETED", submitted="2026-09-01T00:00:00Z")
    _record("2", step="synthetic_generate", state="FAILED", submitted="2026-09-02T00:00:00Z")
    _record("3", step="euclid_query", db_state="UNKNOWN", submitted="2026-09-03T00:00:00Z")
    body = client.get("/api/fasrc/history?step=euclid_query").get_json()
    assert [r["jobid"] for r in body["rows"]] == ["3", "1"]
    body = client.get("/api/fasrc/history?state=unresolved").get_json()
    assert [r["jobid"] for r in body["rows"]] == ["3"]
    body = client.get("/api/fasrc/history?state=FAILED,COMPLETED").get_json()
    assert {r["jobid"] for r in body["rows"]} == {"1", "2"}
    body = client.get("/api/fasrc/history?q=synthetic").get_json()
    assert [r["jobid"] for r in body["rows"]] == ["2"]
    body = client.get("/api/fasrc/history?limit=1&offset=1").get_json()
    assert body["total"] == 3 and len(body["rows"]) == 1 and body["offset"] == 1


def test_history_works_offline(client, monkeypatch):
    monkeypatch.setattr(remote.STATE, "ssh", None)
    _record("1", step="euclid_query", state="COMPLETED")
    assert client.get("/api/fasrc/history").status_code == 200


def test_step_history_rows_drop_the_payload_blobs(client):
    blob = "y" * 4000
    _record("9", step="synthetic_generate", state="COMPLETED", params={"n_train": "5", "_joint_galaxy_population_json": blob})
    r = client.get("/api/fasrc/steps/synthetic_generate/history")
    body = r.get_json()
    assert body["history"][0]["params"] == {"n_train": "5"}
    assert "params_json" not in body["history"][0]
    assert body["history"][0]["params_omitted"] == {"_joint_galaxy_population_json": 4000}
    assert blob not in r.get_data(as_text=True)


# --------------------------------------------------------------------------- queue

def test_queue_resume_route_clears_a_halt(client):
    fasrc_queue.QUEUE.enqueue({"kind": "step", "step": "euclid_query", "form": {}}, "next")
    fasrc_queue.QUEUE._halt("job 1 ended FAILED")
    r = client.post("/api/fasrc/queue/resume")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True and body["queue"]["halted"] is False
    assert body["queue"]["items"][0]["step"] == "euclid_query"


def test_queue_state_is_local_and_detailed(client, monkeypatch):
    monkeypatch.setattr(remote.STATE, "ssh", None)
    fasrc_queue.QUEUE.enqueue({"kind": "step", "step": "tng_grid", "form": {"_blob": "z" * 9000}}, "grid")
    r = client.get("/api/fasrc/queue/state")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["queue"]["count"] == 1 and body["queue"]["items"][0]["step"] == "tng_grid"
    assert "zzzz" not in r.get_data(as_text=True)


def test_queue_resume_works_offline(client, monkeypatch):
    monkeypatch.setattr(remote.STATE, "ssh", None)
    assert client.post("/api/fasrc/queue/resume").status_code == 200


# --------------------------------------------------------------------------- accounting job

def test_refresh_accounting_runs_as_a_cancellable_job(client, monkeypatch):
    calls = {}

    def fake_refresh(ssh, *, scope="all", progress=None, **kw):
        calls["scope"] = scope
        if progress:
            progress(1, 1, "42")
        return {"ok": True, "updated": 1, "total": 1, "scope": scope, "resolved": {"42": "COMPLETED"}}

    monkeypatch.setattr(fasrc_jobs, "refresh_all_post_mortems", fake_refresh)
    r = client.post("/api/fasrc/refresh-accounting", data={"scope": "unresolved"})
    assert r.status_code == 200
    job = _wait(r.get_json()["job_id"])
    assert job["status"] == "done", job
    assert job["kind"] == "fasrc-accounting"
    assert job["result"]["resolved"] == {"42": "COMPLETED"}
    assert calls["scope"] == "unresolved"


def test_refresh_accounting_rejects_an_unknown_scope(client):
    r = client.post("/api/fasrc/refresh-accounting", data={"scope": "everything"})
    assert r.status_code == 400


# --------------------------------------------------------------------------- mirror job

def test_mirror_trigger_requires_confirmation(client, monkeypatch):
    ssh = ScriptedSSH()
    monkeypatch.setattr(remote.STATE, "ssh", ssh)
    r = client.post("/api/fasrc/mirror/trigger")
    assert r.status_code == 400 and r.get_json()["code"] == "confirm_required"
    assert ssh.pulls == []


def test_mirror_trigger_pulls_in_a_job(client, monkeypatch, cfg):
    ssh = ScriptedSSH()
    monkeypatch.setattr(remote.STATE, "ssh", ssh)
    monkeypatch.setattr(fasrc_mirror, "STATE", remote.STATE)
    r = client.post("/api/fasrc/mirror/trigger", data={"confirm": "1"})
    assert r.status_code == 200
    job = _wait(r.get_json()["job_id"])
    assert job["status"] == "done", job
    assert job["kind"] == "fasrc-mirror"
    assert ssh.pulls and ssh.pulls[0][2] == ("--delete-after",)
    status = client.get("/api/fasrc/mirror/status").get_json()
    assert status["last_rc"] == 0 and status["job_id"] is None
    assert "enabled" not in status          # no periodic mirror any more


# --------------------------------------------------------------------------- files

def test_files_lists_the_roots_without_a_dir(client):
    body = client.get("/api/fasrc/files").get_json()
    assert body["ok"] is True and body["dir"] is None
    assert [e["path"] for e in body["entries"]] == ["/n/scratch/data", "/n/scratch/ckpt/wdsr", "/n/repo/logs"]


def test_files_lists_one_remote_dir(client, monkeypatch):
    listing = ("d|0|1790000000.0|images\n"
               "f|2880|1790000100.5|psf.fits\n"
               "l|9|1790000000.0|link\n")
    ssh = ScriptedSSH({"find ": (0, listing, "")})
    monkeypatch.setattr(remote.STATE, "ssh", ssh)
    body = client.get("/api/fasrc/files?dir=/n/scratch/data").get_json()
    assert body["ok"] is True
    names = [(e["name"], e["type"]) for e in body["entries"]]
    assert names == [("images", "dir"), ("link", "link"), ("psf.fits", "file")]
    fits_entry = next(e for e in body["entries"] if e["name"] == "psf.fits")
    assert fits_entry["path"] == "/n/scratch/data/psf.fits" and fits_entry["size"] == 2880
    assert fits_entry["inspectable"] is True
    assert body["crumbs"][0]["path"] == "/n/scratch/data"


def test_files_refuses_paths_outside_the_roots(client):
    r = client.get("/api/fasrc/files?dir=/etc")
    assert r.status_code == 403
    r = client.get("/api/fasrc/files?dir=/n/scratch/data/../../etc")
    assert r.status_code == 403


# --------------------------------------------------------------------------- log search

def test_run_log_grep_returns_numbered_matches(client, monkeypatch):
    path = "/n/repo/logs/pipeline/run-1.out"
    ssh = ScriptedSSH({"grep ": (0, "12:loss nan at step 5\n40:NaN again\n", "")})
    monkeypatch.setattr(remote.STATE, "ssh", ssh)
    body = client.get(f"/api/fasrc/runs/log?path={path}&grep=nan").get_json()
    assert body["ok"] is True
    assert body["matches"] == [{"line": 12, "text": "loss nan at step 5"}, {"line": 40, "text": "NaN again"}]
    cmd = next(c for c in ssh.calls if "grep " in c)
    assert "-F" in cmd and "-i" in cmd and "-e nan " in cmd
    ssh.calls.clear()
    client.get(f"/api/fasrc/runs/log?path={path}&grep=a%27b%3Brm%20-rf")
    cmd = next(c for c in ssh.calls if "grep " in c)
    assert "-e 'a'\"'\"'b;rm -rf' " in cmd            # shell-quoted, never interpreted
    assert client.get(f"/api/fasrc/runs/log?path={path}&grep=%20").status_code == 400


# --------------------------------------------------------------------------- git status

def test_git_status_compares_remote_and_local_head(client, monkeypatch):
    remote_head = "a" * 40
    out = f"main\n0\t2\n{remote_head}\nabc1234\tsubject\t2 days ago\n?? stray.txt\n"
    ssh = ScriptedSSH({"rev-parse --abbrev-ref": (0, out, "")})
    monkeypatch.setattr(remote.STATE, "ssh", ssh)
    monkeypatch.setattr(git_ops, "head", lambda: "b" * 40)
    monkeypatch.setattr(git_ops, "relation", lambda local, rem: {"relation": "remote_behind", "ahead": 3, "behind": 0})
    body = client.get("/api/fasrc/git-status").get_json()
    assert body["ok"] is True
    assert body["head"] == remote_head and body["local_head"] == "b" * 40
    assert body["relation"] == {"relation": "remote_behind", "ahead": 3, "behind": 0}
    assert body["behind"] == 2 and body["ahead"] == 0
    assert body["dirty_files"] == ["?? stray.txt"]
    assert body["last"]["subject"] == "subject"


# --------------------------------------------------------------------------- step outputs

def test_steps_status_lists_known_outputs(client, monkeypatch):
    ssh = ScriptedSSH({"test -e": (0, "ckpt=1\neuclid_cutouts=0\neuclid_psf=1\nsynthetic_records=0\n", "")})
    monkeypatch.setattr(remote.STATE, "ssh", ssh)
    body = client.get("/api/fasrc/steps/status").get_json()
    steps = {s["step_id"]: s for s in body["steps"]}
    outs = steps["extract_euclid_psf"]["outputs"]
    assert outs == [{"key": "euclid_psf", "path": "/n/scratch/data/euclid_psf/euclid_psf_VIS.fits",
                     "exists": True}]
    assert steps["euclid_query"]["outputs"] == []


def test_an_authoritative_ledger_state_is_not_unresolved(client):
    # sacct said OUT_OF_MEMORY while the DB still reads the speculative DONE.
    _record("5", step="ensemble_train", state="OUT_OF_MEMORY", db_state="DONE")
    body = client.get("/api/fasrc/history?state=unresolved").get_json()
    assert body["rows"] == [] and body["unresolved"] == 0
    assert client.get("/api/fasrc/history").get_json()["rows"][0]["state_display"] == "OUT_OF_MEMORY"


def test_a_stale_live_ledger_state_is_unresolved_and_never_shown(client):
    # The ledger caught the job RUNNING / PENDING; the DB has since finalised it.
    _record("7", step="eval_catalog", state="RUNNING", db_state="CANCELLED")
    _record("8", step="eval_catalog", state="PENDING", db_state="DONE")
    _record("9", step="eval_catalog", state="RUNNING", db_state="RUNNING")   # live
    body = client.get("/api/fasrc/history").get_json()
    shown = {r["jobid"]: r["state_display"] for r in body["rows"]}
    assert shown == {"7": "CANCELLED", "8": "DONE", "9": "RUNNING"}
    assert body["unresolved"] == 2
    assert body["facets"]["states"].get("RUNNING") == 1
    rows = client.get("/api/fasrc/history?state=unresolved").get_json()["rows"]
    assert {r["jobid"] for r in rows} == {"7", "8"}
