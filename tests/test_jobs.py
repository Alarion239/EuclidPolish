from __future__ import annotations

import threading
import time

import tqdm as tqdm_module

from euclid_polish.web import jobs
from euclid_polish.web.jobs import Job, JobRegistry, start_exclusive


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


# ---------------------------------------------------------------------------
# Contract C2: kind, cancellable, result, cancel, eviction, summary
# ---------------------------------------------------------------------------

def test_spawn_records_kind_and_serialises_the_contract_keys():
    registry = JobRegistry()
    job_id = registry.spawn("typed", lambda _cap: {"n": 3}, kind="unit-test")
    _wait_for_done(registry, job_id)

    payload = registry.get(job_id).to_dict()

    assert payload["kind"] == "unit-test"
    assert payload["status"] == "done"
    assert payload["result"] == {"n": 3}
    assert payload["cancellable"] is False
    for key in ("job_id", "label", "started", "finished", "duration", "error",
                "log", "log_truncated", "progress"):
        assert key in payload
    assert set(payload["progress"]) == {
        "current", "total", "pct", "label", "stage_elapsed",
        "rate_per_second", "eta_seconds", "updated_ago_seconds",
    }


def test_kind_defaults_to_none_and_label_target_stay_positional():
    registry = JobRegistry()
    job_id = registry.spawn("untyped", lambda _cap: None)
    _wait_for_done(registry, job_id)
    assert registry.get(job_id).to_dict()["kind"] is None


def test_running_job_is_cancellable_until_cancel_is_requested():
    registry = JobRegistry()
    release = threading.Event()
    job_id = registry.spawn("waits", lambda _cap: release.wait(1))
    job = registry.get(job_id)

    assert job.to_dict()["cancellable"] is True
    assert job.cancel() is True
    assert job.to_dict()["cancellable"] is False
    assert job.to_dict()["cancel_requested"] is True
    release.set()
    _wait_for_done(registry, job_id)


def test_cancel_mid_run_stops_at_the_next_tick():
    registry = JobRegistry()
    ticking = threading.Event()
    reached = []

    def target(cap):
        for index in range(10_000):
            cap.tick(index, 10_000, "looping")
            reached.append(index)
            ticking.set()
            time.sleep(0.001)
        return "finished"

    job_id = registry.spawn("long", target, kind="loop")
    assert ticking.wait(1)
    assert registry.cancel(job_id) is True
    _wait_for_done(registry, job_id)

    payload = registry.get(job_id).to_dict()
    assert payload["status"] == "cancelled"
    assert payload["finished"] is not None
    assert payload["result"] is None
    assert payload["cancellable"] is False
    assert "cancelled" in payload["log"].lower()
    assert len(reached) < 10_000


def test_cancel_escapes_broad_exception_handlers_in_targets():
    registry = JobRegistry()
    ticking = threading.Event()

    def target(cap):
        while True:
            try:
                ticking.set()
                cap.tick(0, 0, "busy")
                time.sleep(0.001)
            except Exception:  # noqa: BLE001 - the target swallows errors
                continue

    job_id = registry.spawn("stubborn", target)
    assert ticking.wait(1)
    registry.cancel(job_id)
    _wait_for_done(registry, job_id)
    assert registry.get(job_id).status == "cancelled"


def test_cancel_of_finished_or_unknown_job_is_refused():
    registry = JobRegistry()
    job_id = registry.spawn("quick", lambda _cap: 1)
    _wait_for_done(registry, job_id)

    assert registry.cancel(job_id) is False
    assert registry.cancel("nope") is None
    assert registry.get(job_id).status == "done"


def test_tqdm_progress_also_honours_cancel():
    registry = JobRegistry()
    started = threading.Event()

    def target(cap):
        with cap.tqdm_hook("bar"):
            # Attribute lookup at call time sees the hook's patched class.
            for _ in tqdm_module.tqdm(range(100_000)):
                started.set()
                time.sleep(0.0005)

    job_id = registry.spawn("tqdm", target)
    assert started.wait(1)
    registry.cancel(job_id)
    _wait_for_done(registry, job_id)
    assert registry.get(job_id).status == "cancelled"


def test_result_passthrough_only_for_small_json_values():
    registry = JobRegistry()
    small = registry.spawn("small", lambda _cap: {"path": "/tmp/x", "n": [1, 2]})
    big = registry.spawn("big", lambda _cap: {"blob": "x" * 70_000})
    odd = registry.spawn("odd", lambda _cap: {"set": {1, 2}})
    nan = registry.spawn("nan", lambda _cap: {"value": float("nan")})
    _wait_for_done(registry, small, big, odd, nan)

    assert registry.get(small).to_dict()["result"] == {"path": "/tmp/x", "n": [1, 2]}
    assert registry.get(big).to_dict()["result"] is None
    assert registry.get(odd).to_dict()["result"] is None
    assert registry.get(nan).to_dict()["result"] is None
    # The raw return value is still available to in-process callers.
    assert registry.get(odd).result == {"set": {1, 2}}


def test_failed_job_has_no_result_and_is_not_cancellable():
    registry = JobRegistry()

    def boom(_cap):
        raise RuntimeError("boom")

    job_id = registry.spawn("boom", boom)
    _wait_for_done(registry, job_id)
    payload = registry.get(job_id).to_dict()
    assert payload["status"] == "failed"
    assert payload["result"] is None
    assert payload["cancellable"] is False


def test_registry_keeps_at_most_the_newest_finished_jobs():
    registry = JobRegistry(max_finished=5)
    release = threading.Event()
    running = registry.spawn("still running", lambda _cap: release.wait(2))
    finished = []
    for index in range(8):
        job_id = registry.spawn(f"done {index}", lambda _cap: None)
        _wait_for_done(registry, job_id)
        finished.append(job_id)

    listed = {job["job_id"] for job in registry.list()}
    assert running in listed                       # running jobs never evicted
    assert set(finished[-5:]) <= listed            # newest five kept
    assert not (set(finished[:3]) & listed)        # oldest three evicted
    assert registry.get(finished[0]) is None
    release.set()
    _wait_for_done(registry, running)


def test_default_registry_keeps_two_hundred_finished_jobs():
    assert JobRegistry().max_finished == 200


def test_summary_listing_strips_logs_but_keeps_progress():
    registry = JobRegistry()

    def target(cap):
        cap.write("hello\n")
        cap.tick(1, 2, "half")

    job_id = registry.spawn("logged", target)
    _wait_for_done(registry, job_id)

    full = registry.list()[0]
    summary = registry.list(summary=True)[0]
    assert full["log"] == "hello\n"
    assert summary["log"] is None
    stable = ("current", "total", "pct", "label")
    assert {k: summary["progress"][k] for k in stable} == {
        k: full["progress"][k] for k in stable
    }
    assert summary["job_id"] == job_id


def test_listing_is_newest_first():
    registry = JobRegistry()
    first = registry.spawn("first", lambda _cap: None)
    _wait_for_done(registry, first)
    time.sleep(0.01)
    second = registry.spawn("second", lambda _cap: None)
    _wait_for_done(registry, second)
    assert [job["job_id"] for job in registry.list()] == [second, first]


# ---------------------------------------------------------------------------
# single-run guard (spawn_exclusive / running)
# ---------------------------------------------------------------------------

def test_spawn_exclusive_returns_the_running_job_of_the_same_kind():
    registry = JobRegistry()
    release = threading.Event()
    first, started = registry.spawn_exclusive(
        "first", lambda _cap: release.wait(2), kind="nexus-pair", key="a")
    assert started is True and first.key == "a"
    again, started_again = registry.spawn_exclusive(
        "second", lambda _cap: None, kind="nexus-pair", key="b")
    assert started_again is False
    assert again.job_id == first.job_id and again.key == "a"
    assert registry.running("nexus-pair").job_id == first.job_id
    assert registry.running("other-kind") is None
    release.set()
    _wait_for_done(registry, first.job_id)
    assert registry.running("nexus-pair") is None
    later, started_later = registry.spawn_exclusive(
        "later", lambda _cap: None, kind="nexus-pair", key="b")
    assert started_later is True and later.job_id != first.job_id
    _wait_for_done(registry, later.job_id)


def test_spawn_exclusive_per_key_lets_other_keys_run_concurrently():
    registry = JobRegistry()
    release = threading.Event()
    a, started_a = registry.spawn_exclusive(
        "tile a", lambda _cap: release.wait(2), kind="real-tile", key="a", per_key=True)
    b, started_b = registry.spawn_exclusive(
        "tile b", lambda _cap: release.wait(2), kind="real-tile", key="b", per_key=True)
    a2, started_a2 = registry.spawn_exclusive(
        "tile a again", lambda _cap: None, kind="real-tile", key="a", per_key=True)
    assert started_a and started_b and not started_a2
    assert a2.job_id == a.job_id and b.job_id != a.job_id
    release.set()
    _wait_for_done(registry, a.job_id, b.job_id)


def test_spawn_exclusive_is_atomic_under_concurrent_callers():
    registry = JobRegistry()
    release = threading.Event()
    results: list[tuple[str, bool]] = []
    lock = threading.Lock()
    barrier = threading.Barrier(8)

    def call():
        barrier.wait(1)
        job, started = registry.spawn_exclusive(
            "race", lambda _cap: release.wait(2), kind="race")
        with lock:
            results.append((job.job_id, started))

    threads = [threading.Thread(target=call) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2)
    assert sum(started for _id, started in results) == 1
    assert len({job_id for job_id, _started in results}) == 1
    release.set()
    _wait_for_done(registry, results[0][0])


def test_start_exclusive_payloads():
    registry = JobRegistry()
    release = threading.Event()
    first, status = start_exclusive("crop a", lambda _cap: release.wait(2), kind="mosaic",
                                    key="a", busy="a mosaic job is running", registry=registry)
    assert status == 200 and first == {"ok": True, "job_id": first["job_id"]}
    same, same_status = start_exclusive("crop a again", lambda _cap: None, kind="mosaic",
                                        key="a", registry=registry)
    assert same_status == 200
    assert same == {"ok": True, "job_id": first["job_id"], "already_running": True}
    other, other_status = start_exclusive("crop b", lambda _cap: None, kind="mosaic",
                                          key="b", busy="a mosaic job is running",
                                          registry=registry)
    assert other_status == 409
    assert other["ok"] is False and other["code"] == "busy"
    assert other["job_id"] == first["job_id"]
    assert other["error"].startswith("busy: a mosaic job is running (crop a")
    release.set()
    _wait_for_done(registry, first["job_id"])
