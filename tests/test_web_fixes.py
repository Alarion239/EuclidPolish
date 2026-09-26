"""Backend fixes of the console rework (spec §11): STARFULL defaults, the
confirmed eval sync, config lost-update protection, a cheap ``/api/status``,
non-blocking TNG radius status, the tracking sandbox ``source`` payload and
the training-curve loss series."""

from __future__ import annotations

import inspect
import json
import os
import time

import pytest

from euclid_polish.tracking import timetravel as tt
from euclid_polish.web import job_config, remote
from euclid_polish.web.app import create_app
from euclid_polish.web.helpers import ensemble_viz, status
from euclid_polish.web.jobs import REGISTRY
from euclid_polish.web.routes import ensemble as ensemble_routes
from euclid_polish.web.routes import tng as tng_routes


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _wait(job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = REGISTRY.get(job_id).to_dict()
        if job["status"] != "running":
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never finished")


# ---------------------------------------------------------------------------
# STARFULL is the default regime of every ensemble route
# ---------------------------------------------------------------------------

def test_ensemble_evaluate_defaults_to_starfull(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(ensemble_routes, "job_ensemble_evaluate",
                        lambda cap, **kw: seen.update(kw))
    job_id = client.post("/ensemble/evaluate", data={"num_images": "2"}).get_json()["job_id"]
    _wait(job_id)
    assert seen["starless"] is False
    assert REGISTRY.get(job_id).label.startswith("ensemble: evaluate starfull")


def test_ensemble_evaluate_job_has_no_starless_default():
    """The job itself must not fall back to the starless regime: a caller
    that forgets ``starless`` fails loudly instead of evaluating the wrong
    members against the wrong target (STARFULL is the default regime)."""
    param = inspect.signature(ensemble_viz.job_ensemble_evaluate).parameters["starless"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="starless"):
        ensemble_viz.job_ensemble_evaluate(None, num_images=1)


def test_ensemble_reads_default_to_starfull(client, monkeypatch):
    regimes = []
    monkeypatch.setattr(ensemble_routes, "ensemble_status",
                        lambda starless: regimes.append(starless) or {})
    monkeypatch.setattr(ensemble_routes, "pixel_trace",
                        lambda starless, *a, **k: regimes.append(starless) or {})
    client.get("/ensemble/status.json")
    client.get("/ensemble/pixel-trace.json?diag=std_err&i=0&j=0")
    assert regimes == [False, False]
    assert client.get("/ensemble/status.json?mode=starless").status_code == 200
    assert regimes[-1] is True


# ---------------------------------------------------------------------------
# /api/evaluation/sync runs rsync --delete-after: explicit confirmation only
# ---------------------------------------------------------------------------

class _Up:
    def is_connected(self) -> bool:
        return True

    def rsync_pull(self, *_a, **_kw):
        raise AssertionError("rsync must not run without confirm=1")


def test_evaluation_sync_requires_confirmation(client, monkeypatch):
    monkeypatch.setattr(remote.STATE, "ssh", _Up())
    response = client.post("/api/evaluation/sync")
    assert response.status_code == 400
    assert "confirm" in response.get_json()["error"]


# ---------------------------------------------------------------------------
# /api/config: version + lost-update protection
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg_path(tmp_path, monkeypatch):
    path = tmp_path / "job_config.json"
    monkeypatch.setattr(job_config, "CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(job_config, "CONFIG_PATH", str(path))
    return path


def test_config_carries_a_version_that_changes_with_the_file(client, cfg_path):
    first = client.get("/api/config").get_json()
    assert isinstance(first["version"], str) and len(first["version"]) >= 12
    saved = client.post("/api/config/save", data={
        "n_train": "123", "base_version": first["version"]}).get_json()
    assert saved["ok"] is True
    assert saved["config"]["n_train"] == 123
    assert saved["version"] != first["version"]
    assert client.get("/api/config").get_json()["version"] == saved["version"]


def test_config_save_refuses_a_field_changed_server_side(client, cfg_path):
    base = client.get("/api/config").get_json()
    # Someone else (another tab, a calibration activation) changes n_train.
    job_config.update({"n_train": "999"})
    conflict = client.post("/api/config/save", data={
        "n_train": "123", "base_version": base["version"]})
    assert conflict.status_code == 409
    body = conflict.get_json()
    assert body["ok"] is False and body["code"] == "config_conflict"
    assert set(body["conflicts"]) == {"n_train"}
    assert body["conflicts"]["n_train"]["current"] == 999
    assert job_config.load().n_train == 999


def test_config_save_merges_fields_the_server_did_not_change(client, cfg_path):
    base = client.get("/api/config").get_json()
    job_config.update({"n_valid": "77"})
    saved = client.post("/api/config/save", data={
        "n_train": "123", "base_version": base["version"]})
    assert saved.status_code == 200
    cfg = job_config.load()
    assert (cfg.n_train, cfg.n_valid) == (123, 77)


def test_config_save_with_an_unknown_base_version_is_a_conflict(client, cfg_path):
    conflict = client.post("/api/config/save", data={
        "n_train": "123", "base_version": "0" * 16})
    assert conflict.status_code == 409


def test_config_save_with_an_unknown_base_version_compares_flags_as_bools(
        client, cfg_path):
    """A flag persisted as a JSON bool must compare equal to the form's
    "1"/"true" — not report a false conflict after a server restart."""
    cfg_path.write_text(json.dumps({"plateau_lr_enabled": True, "n_train": 5}))
    for raw in ("1", "true", "on"):
        same = client.post("/api/config/save", data={
            "plateau_lr_enabled": raw, "base_version": "f" * 16})
        assert same.status_code == 200, same.get_json()
        cfg_path.write_text(json.dumps({"plateau_lr_enabled": True, "n_train": 5}))
    changed = client.post("/api/config/save", data={
        "plateau_lr_enabled": "0", "base_version": "f" * 16})
    assert changed.status_code == 409
    assert set(changed.get_json()["conflicts"]) == {"plateau_lr_enabled"}


def test_config_flags_load_and_update_as_their_declared_int(cfg_path):
    """A flag persisted as a JSON bool loads as the declared 0/1 int (so the
    injected CLI value is "1", never "True"); form words update it."""
    cfg_path.write_text(json.dumps({"plateau_lr_enabled": True}))
    loaded = job_config.load().plateau_lr_enabled
    assert loaded == 1 and type(loaded) is int
    assert job_config.fasrc_params_for("ensemble_train")["plateau_lr_enabled"] == "1"
    assert job_config.update({"plateau_lr_enabled": "off"}).plateau_lr_enabled == 0
    assert job_config.update({"plateau_lr_enabled": "true"}).plateau_lr_enabled == 1
    assert job_config.parse_flag("off") is False and job_config.parse_flag("1") is True
    assert job_config.parse_flag("maybe") is None


def test_config_save_without_a_base_version_keeps_working(client, cfg_path):
    saved = client.post("/api/config/save", data={"n_train": "5"})
    assert saved.status_code == 200 and saved.get_json()["config"]["n_train"] == 5


# ---------------------------------------------------------------------------
# /api/status is cache-only; the catalogue pull is an explicit POST
# ---------------------------------------------------------------------------

def test_api_status_never_pulls_from_fasrc(client, monkeypatch):
    def refuse(*_a, **_kw):
        raise AssertionError("GET /api/status must not rsync")

    monkeypatch.setattr(status._fasrc_fetcher, "fetch_one_file", refuse)
    body = client.get("/api/status").get_json()
    assert set(body) >= {"catalog", "psfs", "tfrecords", "checkpoints"}
    assert body["catalog"]["cached"] is True


def test_refresh_catalog_pulls_while_connected(client, monkeypatch):
    calls = []
    monkeypatch.setattr(remote.STATE, "ssh", _Up())
    monkeypatch.setattr(status, "_fasrc_catalog_dir",
                        lambda force=True: calls.append(force))
    response = client.post("/api/status/refresh-catalog")
    assert response.status_code == 200
    assert response.get_json()["ok"] is True
    assert calls == [True]


def test_refresh_catalog_is_gated_offline(client, monkeypatch):
    monkeypatch.setattr(remote.STATE, "ssh", None)
    assert client.post("/api/status/refresh-catalog").status_code == 503


# ---------------------------------------------------------------------------
# /api/tng/radii/status answers from a cache; the validation is a job
# ---------------------------------------------------------------------------

@pytest.fixture
def radii(tmp_path, monkeypatch):
    monkeypatch.setattr(tng_routes, "_radii_cache_path",
                        lambda: str(tmp_path / "radii_status.json"))
    calls = []

    def validate(ssh, *, cfg, argv, timeout):
        calls.append(argv)
        return 0, '{"valid": true, "expected_count": 5770, "valid_count": 5770}\n', ""

    monkeypatch.setattr(tng_routes.fasrc_jobs, "run_remote_python", validate)
    return calls


def test_radii_status_offline_returns_immediately_without_a_job(client, radii, monkeypatch):
    monkeypatch.setattr(remote.STATE, "ssh", None)
    body = client.get("/api/tng/radii/status").get_json()
    assert body["cached"] is False and body["connected"] is False
    assert body["refresh_job"] is None and radii == []
    assert body["stale"] is True


def test_radii_refresh_job_fills_the_cache(client, radii, monkeypatch):
    monkeypatch.setattr(remote.STATE, "ssh", _Up())
    job_id = client.post("/api/tng/radii/refresh").get_json()["job_id"]
    job = _wait(job_id)
    assert job["status"] == "done" and job["kind"] == "tng-radii"
    assert radii[0][0] == "scripts/validate_tng_radius_manifest.py"
    body = client.get("/api/tng/radii/status").get_json()
    assert body["cached"] is True and body["valid"] is True
    assert body["valid_count"] == 5770 and body["connected"] is True
    assert body["refresh_job"] is None and body["stale"] is False


def test_radii_status_get_never_starts_a_job(client, radii, monkeypatch):
    """GET is read-only (spec §9.5): a stale or missing cache is reported as
    ``stale`` and the client asks for a refresh with the POST."""
    monkeypatch.setattr(remote.STATE, "ssh", _Up())
    running_before = [job for job in REGISTRY.list(summary=True)
                      if job.get("kind") == "tng-radii" and job["status"] == "running"]
    body = client.get("/api/tng/radii/status").get_json()
    assert body["cached"] is False and body["stale"] is True
    assert body["refresh_job"] is None
    running_after = [job for job in REGISTRY.list(summary=True)
                     if job.get("kind") == "tng-radii" and job["status"] == "running"]
    assert running_after == running_before and radii == []


def test_radii_status_reports_a_running_refresh(client, radii, monkeypatch):
    monkeypatch.setattr(remote.STATE, "ssh", _Up())
    monkeypatch.setattr(tng_routes, "_radii_refresh_running", lambda: "job-7")
    assert client.get("/api/tng/radii/status").get_json()["refresh_job"] == "job-7"


def test_a_failed_validation_goes_stale_quickly(client, radii, tmp_path, monkeypatch):
    """A failure (e.g. a transient SSH timeout) is cached, but only for the
    short failure TTL — a success stays fresh for the full hour."""
    monkeypatch.setattr(remote.STATE, "ssh", _Up())
    cache = tmp_path / "radii_status.json"
    age = tng_routes._RADII_FAILED_TTL_S + 60
    assert age < tng_routes._RADII_TTL_S
    cache.write_text(json.dumps({"valid": False, "failed": True, "reasons": ["timeout"],
                                 "checked_at": time.time() - age}))
    assert client.get("/api/tng/radii/status").get_json()["stale"] is True
    cache.write_text(json.dumps({"valid": True, "checked_at": time.time() - age}))
    assert client.get("/api/tng/radii/status").get_json()["stale"] is False


def test_radii_refresh_does_not_start_a_second_job_while_one_runs(client, monkeypatch):
    monkeypatch.setattr(remote.STATE, "ssh", _Up())
    monkeypatch.setattr(tng_routes, "_radii_refresh_running", lambda: "job-7")
    assert client.post("/api/tng/radii/refresh").get_json()["job_id"] == "job-7"


def test_radii_refresh_is_gated_offline(client, monkeypatch):
    monkeypatch.setattr(remote.STATE, "ssh", None)
    assert client.post("/api/tng/radii/refresh").status_code == 503


# ---------------------------------------------------------------------------
# tracking sandbox payload: ``source`` is an object + ``source_label``
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source,label", [
    ({"campaign": "grid-v2", "model": "member_196"}, "grid-v2 · member_196"),
    ({"campaign": "grid-v2", "model": None}, "grid-v2"),
    ("legacy text", "legacy text"),
    (None, "—"),
])
def test_sandbox_listing_carries_a_label_and_an_object_source(source, label):
    root = tt.sandbox_dir("abc123")
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "sandbox.json"), "w") as handle:
        json.dump({"short": "abc123", "created_at": "2026-09-26", "source": source}, handle)
    (sandbox,) = tt.list_sandboxes()
    assert isinstance(sandbox["source"], dict)
    assert sandbox["source_label"] == label


# ---------------------------------------------------------------------------
# training curves keep the loss SERIES next to the loss norm
# ---------------------------------------------------------------------------

def test_training_curves_keep_the_loss_series(tmp_path, monkeypatch):
    base = tmp_path / "ens"
    member = base / "member_00"
    member.mkdir(parents=True)
    (member / "origin.json").write_text(json.dumps({"loss_norm": "l2"}))
    monkeypatch.setattr(ensemble_viz, "ensemble_dir", lambda: str(base))
    monkeypatch.setattr("euclid_polish.ensemble_registry.active_member_dirs",
                        lambda _b: [str(member)])
    monkeypatch.setattr(ensemble_viz, "_sky_records_local_dir", lambda: None)
    series = [[1000, 0.5], [2000, 0.25]]
    monkeypatch.setattr(
        "euclid_polish.training.log_plot.ensemble_training_series",
        lambda _b: [{"name": "member_00", "psnr": [[1000, 40.0]], "loss": series}])
    (entry,) = ensemble_viz.training_curves_payload()
    assert entry["loss_series"] == series
    assert entry["loss_norm"] == "l2"
    assert entry["loss"] == "l2"                 # compat alias of loss_norm
