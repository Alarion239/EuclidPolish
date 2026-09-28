"""The ``/api/studies`` endpoints: candidates, freeze job, study view,
sidecars, figures + CSV, fetch (FASRC), delete. Fake SSH, tmp dirs only."""
from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace

import pytest

from euclid_polish.web import app as web_app
from euclid_polish.web import remote
from euclid_polish.web.helpers import experiments
from euclid_polish.web.jobs import REGISTRY
from euclid_polish.web.routes import studies as routes
from tests._studies_fixtures import LABELS, FakeRemote, make_env

REPO = "/n/holylabs/lab/me/EuclidPolish"


@pytest.fixture
def world(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    monkeypatch.setattr(experiments, "free_bytes", lambda _path: 10 ** 15)
    ssh = FakeRemote(tmp_path / "remote")
    monkeypatch.setattr(remote.STATE, "ssh", ssh)
    monkeypatch.setattr(routes.fasrc_config, "load", lambda: SimpleNamespace(
        repo_path=REPO, tracking_remote_dir=""))
    return {**env, "ssh": ssh, "tmp": tmp_path}


@pytest.fixture
def client(world):
    app = web_app.create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _wait(job_id: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = REGISTRY.get(job_id)
        if job is not None and job.status != "running":
            return job.to_dict()
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish")


def _snapshot(root):
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            path = os.path.join(dirpath, name)
            out[path] = os.stat(path).st_mtime_ns
    return out


def _freeze(client, **form):
    response = client.post("/api/studies", data={"name": "Loss study", "note": "l1 vs l2",
                                                 **form})
    assert response.status_code == 200, response.get_json()
    payload = response.get_json()
    job = _wait(payload["job_id"])
    assert job["status"] == "done", job["log"]
    return payload["study_id"], job


def test_candidates_route_is_read_only(client, world):
    before = _snapshot(world["tmp"])
    response = client.get("/api/studies/candidates?mode=starfull")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] and payload["can_freeze"] and payload["ensemble"]["n_members"] == 3
    assert client.get("/api/studies").get_json()["studies"] == []
    assert _snapshot(world["tmp"]) == before
    assert client.get("/api/studies/candidates?mode=weird").status_code == 400


def test_candidate_thumbnail(client):
    response = client.get("/api/studies/candidates/thumb/test-00000.jpg?size=64")
    assert response.status_code == 200 and response.mimetype == "image/jpeg"
    assert client.get("/api/studies/candidates/thumb/test-00099.jpg").status_code == 404
    assert client.get("/api/studies/candidates/thumb/bogus.jpg").status_code == 400


def test_freeze_view_export_fetch_and_delete(client, world):
    sid, job = _freeze(client, fields="test-00001,blackout-00002")
    assert job["result"]["complete"] and job["result"]["mirror"]["ok"]
    listing = client.get("/api/studies").get_json()["studies"]
    assert [s["id"] for s in listing] == [sid] and listing[0]["state"] == "complete"
    view = client.get(f"/api/studies/{sid}").get_json()
    assert view["manifest"]["complete"] and view["note"] == "l1 vs l2"
    assert [m["id"] for m in view["numbers"]["knee_psnr"]["models"]][:3] == LABELS
    assert [f["fid"] for f in view["fields"]] == ["test-00001", "blackout-00002"]
    assert view["fields"][0]["fetched"] is False and view["fields"][0]["thumb_url"]
    assert sid in view["citation"] and len(view["manifest_sha256"]) == 64
    assert client.get(view["fields"][0]["thumb_url"]).mimetype == "image/jpeg"

    png = client.get(f"/api/studies/{sid}/figure/knee?format=png&dpi=150")
    assert png.status_code == 200 and png.data[:4] == b"\x89PNG"
    pdf = client.get(f"/api/studies/{sid}/figure/paired?format=pdf&group=loss&download=1")
    assert pdf.data[:4] == b"%PDF" and "attachment" in pdf.headers["Content-Disposition"]
    csv = client.get(f"/api/studies/{sid}/figure/integrated.csv")
    assert csv.mimetype == "text/csv" and csv.data.startswith(b"series,kind,loss")
    missing = client.get(f"/api/studies/{sid}/figure/real.csv")
    assert missing.status_code == 404 and "Sky › Compare" in missing.get_json()["error"]
    assert client.get(f"/api/studies/{sid}/figure/knee?format=gif").status_code == 400
    numbers = client.get(f"/api/studies/{sid}/numbers/members.csv")
    assert numbers.status_code == 200 and numbers.data.startswith(b"label,")

    note = client.post(f"/api/studies/{sid}/note", data={"note": "revised"}).get_json()
    assert note["note"] == "revised"
    bad = client.post(f"/api/studies/{sid}/selections", json={"selections": [
        {"name": "x", "members": ["99·psnr"]}]})
    assert bad.status_code == 400 and "99·psnr" in bad.get_json()["error"]
    assert client.post(f"/api/studies/{sid}/selections", json={"selections": [
        {"name": "x", "group": "colour"}]}).status_code == 400
    saved = client.post(f"/api/studies/{sid}/selections", json={"selections": [
        {"name": "l1 only", "members": [LABELS[0], LABELS[2]]}]}).get_json()
    assert saved["selections"][0]["members"] == [LABELS[0], LABELS[2]]
    only = client.get(f"/api/studies/{sid}/figure/integrated.csv?selection=l1%20only").data
    assert LABELS[1].encode() not in only.split(b"\nmean")[0]
    assert client.get(f"/api/studies/{sid}/figure/knee.csv?selection=nope").status_code == 404

    meta = client.get(f"/viewer/meta/study?study={sid}").get_json()
    assert meta["objects"][0]["fetched"] is False
    assert client.get(f"/viewer/cube/study/0?study={sid}&tier=hr").status_code == 404
    fetch = client.post(f"/api/studies/{sid}/fields/test-00001/fetch").get_json()
    assert sorted(fetch["products"]) == sorted(["field.json", "truth.json", "lr", "hr", "mean",
                                                 "gate"])
    job = _wait(fetch["job_id"])
    assert job["status"] == "done" and "member_01" not in job["result"]["fetched"]
    field = client.get(f"/api/studies/{sid}").get_json()["fields"][0]
    assert field["fetched"] is True and field["members_fetched"] == 0
    assert field["core_bytes"] < field["bytes"] and set(field["member_bytes"]) == {
        "member_01", "member_02", "member_03"}
    cube = client.get(f"/viewer/cube/study/0?study={sid}&tier=hr")
    assert cube.status_code == 200 and cube.headers["X-Cube-Shape"]
    member = client.get(f"/viewer/cube/study/0?study={sid}&tier=member2")
    assert member.status_code == 404 and "fetch member 03 first" in member.get_json()["error"]
    assert client.post(f"/api/studies/{sid}/fields/test-00001/fetch",
                       data={"products": "member_99"}).status_code == 400
    one = client.post(f"/api/studies/{sid}/fields/test-00001/fetch",
                      data={"products": "member_03"}).get_json()
    assert _wait(one["job_id"])["result"]["fetched"] == ["member_03"]
    assert client.get(f"/viewer/cube/study/0?study={sid}&tier=member2").status_code == 200
    every = client.post(f"/api/studies/{sid}/fields/test-00001/fetch",
                        data={"products": "members"}).get_json()
    assert sorted(_wait(every["job_id"])["result"]["fetched"]) == ["member_01", "member_02"]

    assert client.post(f"/api/studies/{sid}/delete").status_code == 400
    deleted = client.post(f"/api/studies/{sid}/delete", data={"confirm": "1"}).get_json()
    assert deleted["ok"] and client.get("/api/studies").get_json()["studies"] == []
    assert not world["ssh"].local(f"{REPO}/study_fields/{sid}").exists()
    assert not world["ssh"].local(f"{REPO}/tracking/studies/{sid}").exists()


def test_freeze_refusals(client, world, monkeypatch):
    too_many = ",".join(f"test-{i:05d}" for i in range(11))
    response = client.post("/api/studies", data={"name": "x", "fields": too_many})
    assert response.status_code == 400 and "10" in response.get_json()["error"]
    assert client.post("/api/studies", data={"fields": ""}).status_code == 400
    monkeypatch.setattr(remote.STATE, "ssh", None)
    offline = client.post("/api/studies", data={"name": "x", "fields": "test-00001"})
    assert offline.status_code == 503 and offline.get_json()["code"] == "fasrc_offline"
    monkeypatch.setattr(experiments, "free_bytes", lambda _path: experiments.MIN_FREE_BYTES)
    disk = client.post("/api/studies", data={"name": "x"})
    assert disk.status_code == 507 and disk.get_json()["code"] == "insufficient_storage"
    assert client.get("/api/studies").get_json()["studies"] == []


def test_freeze_without_fields_works_offline(client, monkeypatch):
    monkeypatch.setattr(remote.STATE, "ssh", None)
    sid, job = _freeze(client)
    assert job["result"]["complete"] and job["result"]["mirror"]["ok"] is False
    assert client.get(f"/api/studies/{sid}").get_json()["manifest"]["complete"]


def test_fetch_is_gated_offline(client, monkeypatch):
    sid, _job = _freeze(client, fields="test-00001")
    monkeypatch.setattr(remote.STATE, "ssh", None)
    response = client.post(f"/api/studies/{sid}/fields/test-00001/fetch")
    assert response.status_code == 503 and response.get_json()["code"] == "fasrc_offline"


def test_a_failed_upload_resumes(client, world, monkeypatch):
    bad = FakeRemote(world["tmp"] / "remote", corrupt={"mean.npz"})
    monkeypatch.setattr(remote.STATE, "ssh", bad)
    response = client.post("/api/studies", data={"name": "Broken", "fields": "test-00000"})
    sid = response.get_json()["study_id"]
    assert _wait(response.get_json()["job_id"])["status"] == "failed"
    row = client.get("/api/studies").get_json()["studies"][0]
    assert row["state"] == "incomplete" and "mean" in row["reason"]
    assert client.get(f"/api/studies/{sid}/figure/knee").status_code == 409
    monkeypatch.setattr(remote.STATE, "ssh", world["ssh"])
    resumed = client.post(f"/api/studies/{sid}/resume").get_json()
    assert _wait(resumed["job_id"])["status"] == "done"
    assert client.get("/api/studies").get_json()["studies"][0]["state"] == "complete"
    assert client.post(f"/api/studies/{sid}/resume").status_code == 409


def test_unknown_study_is_404(client):
    assert client.get("/api/studies/20260101-000000-nope").status_code == 404
    assert client.get("/api/studies/..%2Fetc").status_code == 404


def test_reading_a_study_writes_nothing(client, world):
    sid, _job = _freeze(client, fields="test-00001")
    before = _snapshot(world["tmp"])
    assert client.get(f"/api/studies/{sid}").status_code == 200
    assert client.get("/api/studies").status_code == 200
    assert client.get(f"/api/studies/{sid}/figure/knee?format=svg").status_code == 200
    assert client.get(f"/api/studies/{sid}/figure/paired.csv").status_code == 200
    assert client.get(f"/api/studies/{sid}/numbers/knee_psnr.json").status_code == 200
    assert client.get(f"/viewer/meta/study?study={sid}").status_code == 200
    assert client.get(f"/viewer/cube/study/0?study={sid}&tier=lr").status_code == 404
    assert _snapshot(world["tmp"]) == before
    assert not world["ssh"].pulled                       # no page ever fetched a field


def test_fetches_run_one_at_a_time_and_block_delete(client, world):
    sid, _job = _freeze(client, fields="test-00000,test-00001")
    gate = threading.Event()
    world["ssh"].pull_gate = gate
    try:
        first = client.post(f"/api/studies/{sid}/fields/test-00000/fetch").get_json()
        again = client.post(f"/api/studies/{sid}/fields/test-00000/fetch").get_json()
        assert again["already_running"] and again["job_id"] == first["job_id"]
        other = client.post(f"/api/studies/{sid}/fields/test-00001/fetch")
        assert other.status_code == 409 and other.get_json()["code"] == "busy"
        deleting = client.post(f"/api/studies/{sid}/delete", data={"confirm": "1"})
        assert deleting.status_code == 409 and "being fetched" in deleting.get_json()["error"]
    finally:
        gate.set()
    assert _wait(first["job_id"])["status"] == "done"
    assert client.post(f"/api/studies/{sid}/delete", data={"confirm": "1"}).get_json()["ok"]
    assert not (world["tmp"] / "vis" / "study_fields" / sid).exists()


def test_a_freeze_that_loses_the_race_leaves_no_study(client, monkeypatch):
    real_start = routes.start_exclusive

    def busy(*_a, **_k):
        return {"ok": False, "code": "busy", "job_id": "x", "error": "busy: another"}, 409

    monkeypatch.setattr(routes, "start_exclusive", busy)
    response = client.post("/api/studies", data={"name": "Racer"})
    assert response.status_code == 409 and response.get_json()["study_id"] is None
    assert client.get("/api/studies").get_json()["studies"] == []
    monkeypatch.setattr(routes, "start_exclusive", real_start)
