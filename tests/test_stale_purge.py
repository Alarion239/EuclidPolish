"""The automatic stale-cube purge: the durable request flag, the idle-time
job the registry hook starts, and the end-to-end effect on the member-cube
buckets (an Evaluate after a purge infers only what the purge removed)."""
from __future__ import annotations

import shutil
import threading
import time

import numpy as np
import pytest

from euclid_polish import ensemble_registry as er
from euclid_polish.config import Config
from euclid_polish.web.helpers import ensemble_viz as ev
from euclid_polish.web.helpers import purge_requests, stale_purge
from euclid_polish.web.jobs import JobRegistry
from tests._ensemble_cube_cache_fixtures import (
    N_FIELDS,
    Cap,
    FakeEnsemble,
    make_env,
    member_dir,
    regenerate_records,
    run_counts,
    set_checkpoint,
)


def test_requests_accumulate_and_clear_only_with_their_token():
    assert purge_requests.read_pending() is None
    purge_requests.request_stale_purge("archived member_03")
    first = purge_requests.read_pending()
    purge_requests.request_stale_purge("pulled member_04")
    second = purge_requests.read_pending()

    assert second["reasons"] == ["archived member_03", "pulled member_04"]
    assert purge_requests.clear_pending(first["requested_at"]) is False   # newer request
    assert purge_requests.clear_pending(second["requested_at"]) is True
    assert purge_requests.read_pending() is None


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "EUCLID_INFERENCE_DIR", str(tmp_path / "inference"))
    return make_env(tmp_path, monkeypatch)


def _evaluate():
    return ev.job_ensemble_evaluate(Cap(), num_images=N_FIELDS, starless=False)


def _member_files(directory, key):
    return sorted(p.name for p in directory.glob(f"member_{key}_*.npy"))


def test_after_a_continued_member_the_purge_leaves_only_that_member_to_infer(env):
    _evaluate()
    set_checkpoint(env["base"], 2, step=2)

    report = stale_purge.purge_stale_caches()

    assert report["bytes_freed"] > 0
    assert [b["dropped"] for b in report["buckets"]] == [["02·psnr"]]
    assert _member_files(env["cubes"], "02") == []
    FakeEnsemble.reset()
    _evaluate()
    assert run_counts() == {"02·psnr": N_FIELDS}
    for rec in range(N_FIELDS):
        stack = np.stack([np.load(env["cubes"] / f"member_{k}_{rec:05d}.npy")
                          for k in ("01", "02", "03")])
        np.testing.assert_allclose(np.load(env["cubes"] / f"sr_{rec:05d}.npy"),
                                   stack.mean(0), rtol=1e-6)


def test_after_an_archive_the_next_evaluate_runs_no_member(env):
    _evaluate()
    er.archive_member_entry(str(env["base"]), "member_03", zip_path="models/x.zip",
                            commit=None)
    ev._mark_archive_stale(False, "member_03")
    shutil.rmtree(member_dir(env["base"], 3))

    stale_purge.purge_stale_caches()

    assert _member_files(env["cubes"], "03") == []
    FakeEnsemble.reset()
    summary = _evaluate()
    assert run_counts() == {}
    assert summary["member_labels"] == ["01·psnr", "02·psnr"]


def test_regenerated_records_wipe_the_bucket(env):
    _evaluate()
    regenerate_records(env["records"], "test")

    report = stale_purge.purge_stale_caches()

    assert report["buckets"][0]["wiped"] == "made from other records"
    assert not env["cubes"].exists()


def test_an_unreadable_ensemble_purges_nothing(env):
    _evaluate()
    shutil.rmtree(env["base"])

    report = stale_purge.purge_stale_caches()

    assert report["bytes_freed"] == 0 and env["cubes"].is_dir()


def test_the_hook_starts_the_purge_only_when_the_registry_is_idle(monkeypatch):
    registry = JobRegistry()
    ran = threading.Event()
    monkeypatch.setattr(stale_purge, "purge_stale_caches",
                        lambda progress=None: (ran.set(), _empty_report())[1])
    registry.add_finish_hook(lambda job: stale_purge.on_job_finished(job, registry))
    release = threading.Event()
    registry.spawn("blocker", lambda _cap: release.wait(2))
    purge_requests.request_stale_purge("archived member_03")
    registry.spawn("archive", lambda _cap: None)
    time.sleep(0.1)
    assert not ran.is_set()                          # the blocker still runs

    release.set()
    assert ran.wait(2)
    _wait_idle(registry)
    assert purge_requests.read_pending() is None
    assert [j["label"] for j in registry.list() if j["kind"] == stale_purge.JOB_KIND] \
        == [stale_purge.JOB_LABEL]


def test_a_failing_purge_does_not_restart_itself(monkeypatch):
    registry = JobRegistry()
    calls = []

    def failing(progress=None):
        calls.append(1)
        raise RuntimeError("disk error")

    monkeypatch.setattr(stale_purge, "purge_stale_caches", failing)
    registry.add_finish_hook(lambda job: stale_purge.on_job_finished(job, registry))
    purge_requests.request_stale_purge("pulled member_04")
    registry.spawn("pull", lambda _cap: None)
    _wait_idle(registry)
    time.sleep(0.1)

    assert calls == [1]
    assert purge_requests.read_pending() is not None   # retried after the next job


def _empty_report():
    return {"buckets": [], "real_fields": [], "bytes_freed": 0, "files_deleted": 0,
            "member_sr_cache": {"bytes_freed": 0, "files_deleted": 0, "members": []}}


def _wait_idle(registry):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        time.sleep(0.01)
        if not any(j["status"] == "running" for j in registry.list(summary=True)):
            return
    raise AssertionError("registry did not go idle")
