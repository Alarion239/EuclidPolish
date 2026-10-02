"""``routes/system.py``: the console's system facts (Settings › About) and
the Home health checks (``/api/system/alerts``).

Everything here is local and must work offline; nothing touches FASRC.
"""

from __future__ import annotations

import json
import os
import platform
import time
from collections import namedtuple
from types import SimpleNamespace

import pytest

from euclid_polish.config import Config
from euclid_polish.web.app import create_app
from euclid_polish.web.jobs import REGISTRY
from euclid_polish.web.routes import system

GIB = 1024 ** 3
Usage = namedtuple("Usage", "total used free")


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def _fresh_state(tmp_path, monkeypatch):
    """A private disk-usage cache file and no memoised checks."""
    monkeypatch.setattr(system, "DISK_USAGE_CACHE_PATH", str(tmp_path / "cache" / "disk.json"))
    system.reset_caches()
    yield
    system.reset_caches()


def _wait(job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = REGISTRY.get(job_id).to_dict()
        if job["status"] != "running":
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never finished")


# ---------------------------------------------------------------------------
# free space
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("free", "total", "level"), [
    (200 * GIB, 460 * GIB, "ok"),
    (19 * GIB, 460 * GIB, "warn"),       # the laptop today: 96 % used, 19 GiB free
    (30 * GIB, 600 * GIB, "warn"),       # plenty of GiB but ≥ 95 % used
    (9 * GIB, 460 * GIB, "bad"),         # below the experiments' floor
    (0, 0, "unknown"),
])
def test_disk_level_thresholds(free, total, level):
    assert system.disk_level(free, total) == level


def test_system_reports_runtime_and_free_space(client, monkeypatch):
    monkeypatch.setattr(system.shutil, "disk_usage", lambda _p: Usage(460 * GIB, 441 * GIB, 19 * GIB))
    body = client.get("/api/system").get_json()
    assert body["python"]["version"] == platform.python_version()
    assert body["platform"]["system"] == platform.system()
    assert "flask" in body["packages"]
    disk = body["disk"]
    assert disk["free_bytes"] == 19 * GIB and disk["total_bytes"] == 460 * GIB
    assert disk["level"] == "warn"
    assert disk["warn_below_bytes"] == system.DISK_WARN_FREE_BYTES
    assert disk["bad_below_bytes"] == system.DISK_BAD_FREE_BYTES
    assert body["experiments"]["cache_budget_bytes"] > 0
    assert body["experiments"]["min_free_bytes"] > 0


def test_system_get_never_starts_a_job(client, monkeypatch):
    spawned = []
    monkeypatch.setattr(system, "spawn_disk_usage_refresh", lambda: spawned.append(1) or "x")
    roots = client.get("/api/system").get_json()["roots"]
    assert spawned == []
    assert roots["items"] == [] and roots["computed_at"] is None
    assert roots["stale"] is True and roots["refresh_job"] is None


# ---------------------------------------------------------------------------
# disk usage per data root (a job; cached)
# ---------------------------------------------------------------------------

def _tree(tmp_path):
    data = tmp_path / "data"
    (data / "vis" / "ensemble").mkdir(parents=True)
    (data / "vis" / "ensemble" / "a.npy").write_bytes(b"x" * 1000)
    (data / "vis" / "b.png").write_bytes(b"x" * 24)
    (data / "euclid_psf").mkdir()
    (data / "euclid_psf" / "p.fits").write_bytes(b"x" * 500)
    (data / "loose.txt").write_bytes(b"x" * 7)
    ckpt = tmp_path / "ckpt" / "wdsr"
    ckpt.mkdir(parents=True)
    (tmp_path / "ckpt" / "ensemble").mkdir()
    (tmp_path / "ckpt" / "ensemble" / "m.index").write_bytes(b"x" * 300)
    return data


def test_measure_tree_counts_bytes_and_files_without_following_links(tmp_path):
    data = _tree(tmp_path)
    (data / "vis" / "link").symlink_to(data / "euclid_psf")
    out = system.measure_tree(str(data / "vis"))
    assert out["bytes"] == 1024 and out["files"] == 2
    assert system.measure_tree(str(tmp_path / "missing")) == {"bytes": 0, "files": 0, "exists": False}


def test_refresh_job_measures_every_root_and_caches_it(client, tmp_path, monkeypatch):
    data = _tree(tmp_path)
    monkeypatch.setattr(Config, "DATA_DIR", str(data))
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR", str(tmp_path / "ckpt" / "wdsr"))
    monkeypatch.setattr(Config, "TRACKING_DIR", str(tmp_path / "tracking"))
    monkeypatch.setattr(system, "REPO_ROOT", str(tmp_path))
    started = client.post("/api/system/disk-usage/refresh").get_json()
    assert started["ok"] is True
    job = _wait(started["job_id"])
    assert job["status"] == "done", job["error"]
    assert job["kind"] == system.DISK_USAGE_JOB_KIND
    roots = client.get("/api/system").get_json()["roots"]
    assert roots["stale"] is False and roots["computed_at"]
    by_id = {item["id"]: item for item in roots["items"]}
    assert by_id["data/vis"]["bytes"] == 1024
    assert by_id["data/euclid_psf"]["bytes"] == 500
    assert by_id["ckpt"]["bytes"] == 300
    assert by_id["tracking"]["exists"] is False
    # biggest first
    sizes = [item["bytes"] for item in roots["items"]]
    assert sizes == sorted(sizes, reverse=True)
    assert roots["total_bytes"] == sum(sizes)
    # persisted: a restart (cleared memory) still shows the last numbers
    system.reset_caches()
    again = client.get("/api/system").get_json()["roots"]
    assert {i["id"] for i in again["items"]} == set(by_id)
    assert json.loads(open(system.DISK_USAGE_CACHE_PATH).read())["items"]


def test_refresh_runs_one_job_at_a_time(client, monkeypatch):
    gate = []
    monkeypatch.setattr(system, "compute_disk_usage",
                        lambda progress=None: (gate.append(1), time.sleep(0.3),
                                               {"computed_at": "x", "items": []})[-1])
    first = client.post("/api/system/disk-usage/refresh").get_json()["job_id"]
    second = client.post("/api/system/disk-usage/refresh").get_json()["job_id"]
    assert first == second
    assert client.get("/api/system").get_json()["roots"]["refresh_job"] == first
    _wait(first)


def test_old_cache_reads_stale(client, monkeypatch):
    os.makedirs(os.path.dirname(system.DISK_USAGE_CACHE_PATH), exist_ok=True)
    with open(system.DISK_USAGE_CACHE_PATH, "w") as fh:
        json.dump({"computed_at": "2020-01-01T00:00:00+00:00", "computed_ts": 1.0,
                   "items": [{"id": "data/vis", "bytes": 5}]}, fh)
    roots = client.get("/api/system").get_json()["roots"]
    assert roots["stale"] is True and roots["items"][0]["id"] == "data/vis"


# ---------------------------------------------------------------------------
# Home health checks
# ---------------------------------------------------------------------------

def _only(monkeypatch, **checks):
    """Replace the check list with the given callables (id → fn)."""
    monkeypatch.setattr(system, "CHECKS", tuple(checks.items()))


def _check(state, **extra):
    return lambda: {"state": state, "title": f"{state} title", "detail": "d", "to": "/x", **extra}


def test_alerts_list_only_warn_and_bad_checks(client, monkeypatch):
    _only(monkeypatch, a=_check("ok"), b=_check("warn"), c=_check("bad"), d=_check("unknown"))
    body = client.get("/api/system/alerts").get_json()
    assert [c["id"] for c in body["checks"]] == ["a", "b", "c", "d"]
    assert [a["id"] for a in body["alerts"]] == ["c", "b"]        # worst first
    assert body["counts"] == {"bad": 1, "warn": 1, "ok": 1, "unknown": 1}
    assert body["computed_at"]


def test_a_failing_check_reads_unknown_not_500(client, monkeypatch):
    def boom():
        raise RuntimeError("records dir unreadable")
    _only(monkeypatch, fine=_check("ok"), broken=boom)
    r = client.get("/api/system/alerts")
    assert r.status_code == 200
    broken = next(c for c in r.get_json()["checks"] if c["id"] == "broken")
    assert broken["state"] == "unknown" and "records dir unreadable" in broken["detail"]


def test_alerts_are_memoised_until_fresh(client, monkeypatch):
    calls = []
    _only(monkeypatch, a=lambda: calls.append(1) or {"state": "ok", "title": "t"})
    client.get("/api/system/alerts")
    client.get("/api/system/alerts")
    assert len(calls) == 1
    client.get("/api/system/alerts?fresh=1")
    assert len(calls) == 2


def test_loop_route_serves_the_staleness_service_and_never_starts_a_job(client, monkeypatch):
    """GET /api/system/loop is the one staleness service Home's Loop and
    System › Lineage read: it hands loop_payload the alerts and the
    records-noise check, passes ?fresh=1 through, and starts nothing."""
    seen = {}

    def loop_payload(*, alerts, check_records_noise, fresh=False):
        seen["alerts"] = alerts()
        seen["noise"] = check_records_noise is system.check_records_noise
        seen["fresh"] = fresh
        return {"computed_at": "now", "ttl_s": 60.0, "stages": [{"id": "records", "state": "stale"}],
                "counts": {"current": 0, "stale": 1, "blocked": 0, "unknown": 0}, "errors": {}}

    _only(monkeypatch, a=_check("ok"))
    monkeypatch.setattr(system.system_alerts, "loop_payload", loop_payload)
    before = len(REGISTRY.list())
    body = client.get("/api/system/loop").get_json()
    assert body["stages"] == [{"id": "records", "state": "stale"}]
    assert [c["id"] for c in seen["alerts"]["checks"]] == ["a"]
    assert seen["noise"] is True and seen["fresh"] is False
    client.get("/api/system/loop?fresh=1")
    assert seen["fresh"] is True
    assert len(REGISTRY.list()) == before


def test_disk_check(monkeypatch):
    monkeypatch.setattr(system.shutil, "disk_usage", lambda _p: Usage(460 * GIB, 441 * GIB, 19 * GIB))
    check = system.check_disk()
    assert check["state"] == "warn" and "19.0 GiB free" in check["title"]
    assert check["to"] == "/system/storage"
    monkeypatch.setattr(system.shutil, "disk_usage", lambda _p: Usage(460 * GIB, 300 * GIB, 160 * GIB))
    assert system.check_disk()["state"] == "ok"


def _layer(states):
    return {"features": [{"props": {"state": s}} for s in states]}


def test_real_sr_check_counts_production_states(monkeypatch):
    layers = {"nexus-tiles": _layer(["stale"] * 445), "real-fields": _layer(["missing"] * 100),
              "real-tiles": _layer(["current", "stale"]), "poster": _layer(["stale"] * 4),
              "pairs": _layer([])}
    monkeypatch.setattr(system.sky_atlas, "layer_features", lambda lid: layers[lid])
    check = system.check_real_sr()
    assert check["state"] == "warn"
    assert check["facts"]["stale"] == 450 and check["facts"]["current"] == 1
    by = {s["source"]: s for s in check["facts"]["sources"]}
    assert by["nexus"] == {"source": "nexus", "label": "NEXUS tiles", "current": 0, "stale": 445, "missing": 0}
    assert "450" in check["title"] and check["to"] == "/sky/targets"


def test_real_sr_check_is_ok_when_everything_is_current(monkeypatch):
    monkeypatch.setattr(system.sky_atlas, "layer_features",
                        lambda lid: _layer(["current"] if lid == "nexus-tiles" else []))
    assert system.check_real_sr()["state"] == "ok"


def _spec(spec, available, reason=None, members=()):
    return SimpleNamespace(spec=spec, available=available, reason=reason,
                           member_labels=tuple(members))


def test_combiner_check_follows_the_production_spec(monkeypatch):
    monkeypatch.setattr(system.model_catalog, "list_specs", lambda: [
        _spec("production", False, "the production gate was fitted for 26 members; the active "
                                   "STARFULL ensemble has 30"),
    ])
    check = system.check_combiner()
    assert check["state"] == "warn" and "26 members" in check["detail"]
    assert check["to"] == "/models/combiner"
    monkeypatch.setattr(system.model_catalog, "list_specs",
                        lambda: [_spec("production", True, members=["1·psnr"] * 30)])
    ok = system.check_combiner()
    assert ok["state"] == "ok" and "30 members" in ok["title"]


@pytest.fixture
def ensemble_dirs(tmp_path, monkeypatch):
    regime = tmp_path / "vis" / "ensemble" / "starfull"
    regime.mkdir(parents=True)
    records = tmp_path / "records"
    records.mkdir()
    for name in ("dirty_test", "hr_test"):
        (records / f"{name}.tfrecord").write_bytes(b"x" * 10)
    monkeypatch.setattr(system.model_catalog, "regime_dir", lambda: regime)
    monkeypatch.setattr(system, "records_dir", lambda: str(records))
    monkeypatch.setattr(system.model_catalog, "active_member_labels", lambda: ["1·psnr", "2·psnr"])
    monkeypatch.setattr(system, "experiment_records_root", lambda: tmp_path / "experiments")
    return SimpleNamespace(regime=regime, records=records)


def _records_fp(records):
    parts = []
    for kind in ("dirty", "hr"):
        st = os.stat(records / f"{kind}_test.tfrecord")
        parts.append(f"{kind}:{st.st_size}:{st.st_mtime_ns}")
    return "|".join(parts)


def _summary(dirs, *, labels=("1·psnr", "2·psnr"), records_fp=None):
    (dirs.regime / "eval_summary.json").write_text(json.dumps({
        "member_labels": list(labels),
        "eval_identity": {"records_fp": records_fp or _records_fp(dirs.records), "subset": "test"},
    }))


def test_evaluation_check_current(ensemble_dirs):
    _summary(ensemble_dirs)
    assert system.check_evaluation()["state"] == "ok"


def test_evaluation_check_missing(ensemble_dirs):
    check = system.check_evaluation()
    assert check["state"] == "warn" and "not evaluated" in check["title"].lower()
    assert check["action"]["url"] == "/ensemble/evaluate"


def test_evaluation_check_membership_changed(ensemble_dirs):
    _summary(ensemble_dirs, labels=("1·psnr",))
    check = system.check_evaluation()
    assert check["state"] == "warn" and "members" in check["title"]


def test_evaluation_check_ignores_member_order(ensemble_dirs):
    _summary(ensemble_dirs, labels=("2·psnr", "1·psnr"))
    assert system.check_evaluation()["state"] == "ok"


def test_evaluation_check_records_regenerated(ensemble_dirs):
    _summary(ensemble_dirs, records_fp="dirty:10:1|hr:10:1")
    check = system.check_evaluation()
    assert check["state"] == "warn" and "records" in check["title"]


# ---------------------------------------------------------------------------
# Home's production numbers (/api/system/production)
# ---------------------------------------------------------------------------

def _headline_summary(dirs, labels=("1·psnr", "2·psnr")):
    (dirs.regime / "eval_summary.json").write_text(json.dumps({
        "member_labels": list(labels), "n_scored": 100, "per_member_psnr_stretched": [57.0, 58.0],
        "ensemble_psnr": 58.3753, "mean_member_psnr": 57.22, "ensemble_gain_db": 1.155,
        "combiner_psnr": 99.0,
        "spatial_gate_combiner_psnr": 59.2354, "spatial_gate_combiner_vs_mean_db": 0.8601,
        "spatial_gate_combiner_vs_best_member_db": 0.2948,
        "eval_identity": {"records_fp": _records_fp(dirs.records), "subset": "test"},
    }))


def test_production_reads_the_eval_summary_headline(client, ensemble_dirs, monkeypatch):
    monkeypatch.setattr(system, "starless_member_labels", lambda: ["105·psnr"] * 12)
    _headline_summary(ensemble_dirs)
    body = client.get("/api/system/production").get_json()
    summary = body["eval_summary"]
    assert summary["spatial_gate_combiner_psnr"] == pytest.approx(59.2354)
    assert summary["spatial_gate_combiner_vs_best_member_db"] == pytest.approx(0.2948)
    assert summary["ensemble_psnr"] == pytest.approx(58.3753)
    assert "per_member_psnr_stretched" not in summary and "eval_identity" not in summary  # scalars only
    assert body["stale"] is False and body["stale_reason"] is None
    assert body["evaluated_at"]
    assert body["members"] == 2 and body["starless_members"] == 12


def test_production_flags_a_summary_of_other_members(client, ensemble_dirs, monkeypatch):
    monkeypatch.setattr(system, "starless_member_labels", lambda: [])
    _headline_summary(ensemble_dirs, labels=("1·psnr",))
    body = client.get("/api/system/production").get_json()
    assert body["stale"] is True
    assert "predates the current members" in body["stale_reason"]


def test_production_without_an_evaluation(client, ensemble_dirs, monkeypatch):
    monkeypatch.setattr(system, "starless_member_labels", lambda: [])
    body = client.get("/api/system/production").get_json()
    assert body["eval_summary"] is None and body["stale"] is False and body["evaluated_at"] is None
    assert body["members"] == 2


def test_knee_check(monkeypatch):
    monkeypatch.setattr(system, "knee_psnr_status", lambda starless: {"available": True, "stale": True})
    assert system.check_knee()["state"] == "warn"
    monkeypatch.setattr(system, "knee_psnr_status", lambda starless: {"available": True, "stale": False})
    assert system.check_knee()["state"] == "ok"
    monkeypatch.setattr(system, "knee_psnr_status", lambda starless: {"available": False, "stale": False})
    assert system.check_knee()["state"] == "warn"


def test_records_noise_check(tmp_path, monkeypatch):
    for name in ("dirty_validate", "dirty_test"):
        (tmp_path / f"{name}.tfrecord").write_bytes(b"x")
    monkeypatch.setattr(system, "records_dir", lambda: str(tmp_path))
    seen = {"dirty_validate": Config.NOISE_MODEL, "dirty_test": "noise-v3"}
    monkeypatch.setattr(system, "record_noise_model",
                        lambda path: seen[os.path.basename(path).removesuffix(".tfrecord")])
    check = system.check_records_noise()
    assert check["state"] == "bad" and "dirty_test" in check["detail"] and "noise-v3" in check["detail"]
    seen["dirty_test"] = Config.NOISE_MODEL
    assert system.check_records_noise()["state"] == "ok"
    seen.update(dirty_test=None, dirty_validate=None)
    unknown = system.check_records_noise()
    assert unknown["state"] == "unknown" and "provenance" in unknown["detail"]
    # honest about its reach: FASRC-only splits are never examined
    assert "dirty_train" in unknown["detail"] and unknown["facts"]["local_only"] is True


def test_records_noise_check_without_records(tmp_path, monkeypatch):
    monkeypatch.setattr(system, "records_dir", lambda: str(tmp_path / "none"))
    assert system.check_records_noise()["state"] == "unknown"


def test_tracking_check_flags_results_newer_than_the_last_entry(tmp_path, monkeypatch, ensemble_dirs):
    tracking = tmp_path / "tracking"
    (tracking / "current").mkdir(parents=True)
    (tracking / "current" / "log.md").write_text(
        "# notebook\n\n## 2026-09-20T10:00:00Z\n\nold\n\n## 2026-09-21T14:42:29Z\n\nlast\n")
    monkeypatch.setattr(Config, "TRACKING_DIR", str(tracking))
    _summary(ensemble_dirs)
    stamp = time.mktime(time.strptime("2026-09-25 12:00", "%Y-%m-%d %H:%M"))
    os.utime(ensemble_dirs.regime / "eval_summary.json", (stamp, stamp))
    check = system.check_tracking()
    assert check["state"] == "warn"
    assert check["facts"]["last_entry"].startswith("2026-09-21T14:42:29")
    assert "evaluation" in check["detail"]
    assert check["to"] == "/notebook/log"
    old = time.mktime(time.strptime("2026-09-19 12:00", "%Y-%m-%d %H:%M"))
    os.utime(ensemble_dirs.regime / "eval_summary.json", (old, old))
    assert system.check_tracking()["state"] == "ok"


def test_tracking_check_without_a_log(tmp_path, monkeypatch, ensemble_dirs):
    monkeypatch.setattr(Config, "TRACKING_DIR", str(tmp_path / "nothing"))
    assert system.check_tracking()["state"] == "unknown"


def test_last_log_entry_parses_iso_headings():
    text = "## 2026-09-21T01:36:56Z\n\nx\n## not a date\n## 2026-09-21T14:42:29Z\n"
    assert system.last_log_entry(text).isoformat().startswith("2026-09-21T14:42:29")
    assert system.last_log_entry("no headings") is None


def test_every_default_check_answers_on_an_empty_checkout(client):
    body = client.get("/api/system/alerts").get_json()
    ids = [c["id"] for c in body["checks"]]
    assert ids == [cid for cid, _fn in system.CHECKS]
    for check in body["checks"]:
        assert check["state"] in {"ok", "warn", "bad", "unknown"}
        assert check["title"]
