"""HTTP side of the local jobs contract (C2): list, summary, get, cancel."""

from __future__ import annotations

import threading
import time

import pytest

from euclid_polish.web.app import create_app
from euclid_polish.web.jobs import REGISTRY


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _wait(job_id: str, status: str = "running", timeout: float = 3.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = REGISTRY.get(job_id)
        if job is not None and job.status != status:
            return job.to_dict()
        time.sleep(0.005)
    raise AssertionError(f"job {job_id} stayed {status}")


def test_jobs_list_carries_kind_cancellable_result(client):
    job_id = REGISTRY.spawn("api result", lambda _cap: {"answer": 42}, kind="api-test")
    _wait(job_id)

    jobs = client.get("/api/jobs").get_json()
    job = next(j for j in jobs if j["job_id"] == job_id)
    assert job["kind"] == "api-test"
    assert job["result"] == {"answer": 42}
    assert job["cancellable"] is False
    assert job["status"] == "done"


def test_summary_listing_has_no_logs(client):
    def target(cap):
        cap.write("secret log line\n")

    job_id = REGISTRY.spawn("api summary", target)
    _wait(job_id)

    summary = client.get("/api/jobs?summary=1").get_json()
    job = next(j for j in summary if j["job_id"] == job_id)
    assert job["log"] is None
    assert "secret log line" not in client.get("/api/jobs?summary=1").get_data(as_text=True)
    full = next(j for j in client.get("/api/jobs").get_json() if j["job_id"] == job_id)
    assert full["log"] == "secret log line\n"


def test_single_job_keeps_its_log_tail(client):
    job_id = REGISTRY.spawn("api single", lambda cap: cap.write("tail\n"))
    _wait(job_id)
    payload = client.get(f"/api/jobs/{job_id}").get_json()
    assert payload["job_id"] == job_id
    assert payload["log"] == "tail\n"


def test_unknown_job_is_a_json_404(client):
    response = client.get("/api/jobs/doesnotexist")
    assert response.status_code == 404
    assert response.get_json()["ok"] is False


def test_cancel_endpoint_stops_a_ticking_job(client):
    ticking = threading.Event()

    def target(cap):
        for index in range(100_000):
            cap.tick(index, 100_000, "spin")
            ticking.set()
            time.sleep(0.001)

    job_id = REGISTRY.spawn("api cancel", target, kind="spin")
    assert ticking.wait(1)

    response = client.post(f"/api/jobs/{job_id}/cancel")
    assert response.status_code == 200
    assert response.get_json() == {"ok": True}
    payload = _wait(job_id)
    assert payload["status"] == "cancelled"
    assert client.get(f"/api/jobs/{job_id}").get_json()["status"] == "cancelled"


def test_cancel_of_a_finished_job_conflicts(client):
    job_id = REGISTRY.spawn("api done", lambda _cap: None)
    _wait(job_id)
    response = client.post(f"/api/jobs/{job_id}/cancel")
    assert response.status_code == 409
    body = response.get_json()
    assert body["ok"] is False and body["error"]


def test_cancel_of_an_unknown_job_is_404(client):
    response = client.post("/api/jobs/nope/cancel")
    assert response.status_code == 404
    assert response.get_json()["ok"] is False


def test_cancel_is_post_only(client):
    assert client.get("/api/jobs/whatever/cancel").status_code == 405
