"""Settings backend: the job-config schema (defaults, types, the FASRC steps
each field feeds) and the one laptop-side Euclid archive session."""

from __future__ import annotations

import pytest

from euclid_polish.web import euclid_session, job_config
from euclid_polish.web.app import create_app
from euclid_polish.web.fasrc_pipeline import REGISTRY as STEPS
from euclid_polish.web.routes import auth


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def cfg_path(tmp_path, monkeypatch):
    path = tmp_path / "job_config.json"
    monkeypatch.setattr(job_config, "CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(job_config, "CONFIG_PATH", str(path))
    return path


def test_defaults_are_the_dataclass_defaults(cfg_path):
    assert job_config.defaults() == job_config.JobConfig().to_dict()
    job_config.update({"n_train": "5"})
    assert job_config.defaults()["n_train"] == job_config.JobConfig().n_train


def test_used_by_inverts_the_step_mapping():
    used = job_config.used_by()
    assert used["vis_pixels"] == ["download_euclid_cutouts", "extract_euclid_psf"]
    assert "synthetic_generate" in used["psf_warp_prob"] and "ensemble_train" in used["psf_warp_prob"]
    assert used["hr_image_size"] == ["synthetic_generate"]           # injected as image_size
    assert used["lr_peak"] == ["ensemble_train"]
    assert "asinh_scale" not in used                                 # a local display knob


def test_field_types_follow_the_declared_defaults():
    types = job_config.field_types()
    assert types["n_train"] == "int" and types["lr_peak"] == "float"
    assert types["plateau_lr_metric"] == "str" and types["plateau_lr_enabled"] == "int"
    assert set(types) == set(job_config.JobConfig().to_dict())


def test_get_config_carries_the_schema(client, cfg_path):
    body = client.get("/api/config").get_json()
    assert body["ok"] is True and body["version"]
    assert body["defaults"] == job_config.defaults()
    assert body["used_by"] == job_config.used_by()
    assert body["types"] == job_config.field_types()
    for step_ids in body["used_by"].values():
        for step_id in step_ids:
            assert body["steps"][step_id] == STEPS.get(step_id).label


def test_resetting_a_field_is_a_save_of_its_default(client, cfg_path):
    base = client.get("/api/config").get_json()
    changed = client.post("/api/config/save", data={"n_train": "7", "base_version": base["version"]})
    assert changed.get_json()["config"]["n_train"] == 7
    reset = client.post("/api/config/save", data={
        "n_train": str(base["defaults"]["n_train"]), "base_version": changed.get_json()["version"]})
    assert reset.get_json()["config"]["n_train"] == base["defaults"]["n_train"]


# ---------------------------------------------------------------------------
# the one laptop-side Euclid archive session
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_archive(monkeypatch):
    logins = []

    def login(user, password):
        if password != "right":
            raise RuntimeError("Invalid credentials")
        logins.append(user)
        monkeypatch.setattr(euclid_session, "_catalog", object())
        monkeypatch.setattr(euclid_session, "_user", user)

    def logout():
        monkeypatch.setattr(euclid_session, "_catalog", None)
        monkeypatch.setattr(euclid_session, "_user", None)

    monkeypatch.setattr(euclid_session, "login", login)
    monkeypatch.setattr(euclid_session, "logout", logout)
    monkeypatch.setattr(euclid_session, "_catalog", None)
    monkeypatch.setattr(euclid_session, "_user", None)
    monkeypatch.setattr(auth, "_SESSION", {"logged_in_at": None})
    return logins


def test_status_describes_the_one_session_and_its_consumers(client, fake_archive):
    body = client.get("/auth/status").get_json()
    assert body["authenticated"] is False and body["user"] is None
    assert body["logged_in_at"] is None
    used = {item["id"] for item in body["used_by"]}
    assert {"catalog-eval", "galaxies", "stars"} <= used
    assert all(item["to"].startswith("/") for item in body["used_by"])


def test_login_records_the_time_and_logout_clears_it(client, fake_archive):
    ok = client.post("/auth/login", data={"username": "abelo", "password": "right"}).get_json()
    assert ok == {"ok": True, "user": "abelo"}          # the legacy reply shape is kept
    status = client.get("/auth/status").get_json()
    assert status["authenticated"] is True and status["user"] == "abelo" and status["logged_in_at"]
    assert client.post("/auth/logout").get_json() == {"ok": True}
    after = client.get("/auth/status").get_json()
    assert after["authenticated"] is False and after["logged_in_at"] is None


def test_a_failed_login_keeps_the_previous_state(client, fake_archive):
    bad = client.post("/auth/login", data={"username": "abelo", "password": "wrong"})
    assert bad.status_code == 500 and "Invalid credentials" in bad.get_json()["error"]
    assert client.get("/auth/status").get_json()["logged_in_at"] is None


def test_login_requires_both_fields(client, fake_archive):
    r = client.post("/auth/login", data={"username": "abelo"})
    assert r.status_code == 400
