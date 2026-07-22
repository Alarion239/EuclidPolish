from __future__ import annotations

import threading
import time

from euclid_polish.web.jobs import Job, JobRegistry


def _wait_for_done(registry: JobRegistry, *job_ids: str) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if all(registry.get(job_id).status != "running" for job_id in job_ids):
            return
        time.sleep(0.005)
    raise AssertionError("jobs did not finish")


def test_concurrent_jobs_keep_print_output_in_the_originating_log():
    registry = JobRegistry()
    a_started = threading.Event()
    b_printed = threading.Event()
    a_finished_print = threading.Event()

    def target_a(_capture):
        print("A1")
        a_started.set()
        assert b_printed.wait(1)
        print("A2")
        a_finished_print.set()

    def target_b(_capture):
        assert a_started.wait(1)
        print("B1")
        b_printed.set()
        assert a_finished_print.wait(1)

    job_a = registry.spawn("A", target_a)
    job_b = registry.spawn("B", target_b)
    _wait_for_done(registry, job_a, job_b)

    assert registry.get(job_a).log == "A1\nA2\n"
    assert registry.get(job_b).log == "B1\n"


def test_job_can_be_serialized_while_log_and_progress_are_updated():
    registry = JobRegistry()
    release = threading.Event()

    def target(capture):
        for index in range(100):
            capture.write(f"line {index}\n")
            capture.tick(index + 1, 100, "working")
        release.wait(1)

    job_id = registry.spawn("pollable", target)
    job = registry.get(job_id)
    deadline = time.monotonic() + 1
    while job.progress_current < 100 and time.monotonic() < deadline:
        payload = job.to_dict()
        assert payload["progress"]["current"] <= payload["progress"]["total"]
    release.set()
    _wait_for_done(registry, job_id)

    payload = job.to_dict()
    assert payload["status"] == "done"
    assert payload["progress"]["current"] == 100
    assert "line 99" in payload["log"]


def test_job_progress_exposes_stage_rate_and_eta(monkeypatch):
    from euclid_polish.web import jobs

    now = [100.0]
    monkeypatch.setattr(jobs.time, "time", lambda: now[0])
    job = Job("eta", "tracked", started=now[0])

    job.set_progress(1, 10, "field 1")
    now[0] = 102.0
    job.set_progress(2, 10, "field 2")
    payload = job.to_dict()["progress"]

    assert payload["stage_elapsed"] == 2.0
    assert payload["rate_per_second"] == 0.5
    assert payload["eta_seconds"] == 16.0
    assert payload["updated_ago_seconds"] == 0.0

    now[0] = 103.0
    job.set_progress(1, 5, "next stage")
    payload = job.to_dict()["progress"]
    assert payload["stage_elapsed"] == 0.0
    assert payload["rate_per_second"] is None
    assert payload["eta_seconds"] is None
