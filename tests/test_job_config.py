"""Universal job-config persistence + the /config save endpoint."""

from __future__ import annotations

import pytest

from euclid_polish.web import job_config
from euclid_polish.web.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def cfg_path(tmp_path, monkeypatch):
    p = tmp_path / "job_config.json"
    monkeypatch.setattr(job_config, "CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(job_config, "CONFIG_PATH", str(p))
    return p


def test_defaults_and_odd_vis_pixels(cfg_path):
    c = job_config.load()
    assert c.vis_pixels % 2 == 1            # default must be odd


def test_update_persists_and_forces_odd(cfg_path):
    c = job_config.update({"vis_pixels": "512", "n_train": "1234",
                           "asinh_scale": "250"})
    assert c.vis_pixels == 513              # even → bumped to odd
    assert c.n_train == 1234
    assert c.asinh_scale == 250.0
    # survives a reload (persisted to disk)
    again = job_config.load()
    assert again.vis_pixels == 513
    assert again.n_train == 1234


def test_blank_fields_are_ignored(cfg_path):
    job_config.update({"n_valid": "42"})
    c = job_config.update({"n_valid": "", "n_train": "77"})
    assert c.n_valid == 42                  # blank didn't wipe it
    assert c.n_train == 77


def test_save_endpoint_round_trips(client, cfg_path):
    r = client.post("/api/config/save", data={
        "vis_pixels": "300",               # even → coerced
        "n_train": "8000", "n_valid": "200",
        "hr_image_size": "510", "asinh_scale": "500",
    })
    assert r.status_code == 200
    d = r.get_json()
    assert d["ok"] is True
    assert d["config"]["vis_pixels"] == 301
    assert d["config"]["n_train"] == 8000
    assert d["note"] and "odd" in d["note"]


def test_config_page_renders(client, cfg_path):
    r = client.get("/config")
    assert r.status_code == 200
    assert b"Universal job config" in r.data
    assert b"Star field" in r.data
    assert b"Strong lenses" in r.data


def test_lens_field_defaults_update_and_mapping(cfg_path):
    from euclid_polish.config import Config
    c = job_config.load()
    assert c.lens_density_arcmin2 == Config.LENS_DENSITY_ARCMIN2
    assert c.lens_sigma_v_min_kms == Config.LENS_SIGMA_V_MIN_KMS
    c = job_config.update({"lens_density_arcmin2": "8", "lens_sigma_v_min_kms": "180",
                           "lens_sigma_v_max_kms": "400"})
    assert c.lens_density_arcmin2 == 8.0 and c.lens_sigma_v_min_kms == 180.0
    assert c.lens_sigma_v_max_kms == 400.0
    assert job_config.load().lens_density_arcmin2 == 8.0      # persisted
    m = job_config.FASRC_STEP_PARAMS["synthetic_generate"]
    for k in ("lens_density_arcmin2", "lens_sigma_v_min_kms", "lens_sigma_v_max_kms"):
        assert m[k] == k


def test_star_field_defaults_and_update(cfg_path):
    from euclid_polish.config import Config
    c = job_config.load()
    assert c.star_mag_bright == Config.STAR_MAG_BRIGHT      # default from Config
    assert c.star_density_arcmin2 == Config.DEFAULT_STAR_DENSITY_ARCMIN2
    c = job_config.update({"star_mag_bright": "9.5", "star_mag_slope": "0.3",
                           "star_density_arcmin2": "4.2", "star_mag_faint": "24"})
    assert c.star_mag_bright == 9.5 and c.star_mag_slope == 0.3
    assert c.star_density_arcmin2 == 4.2 and c.star_mag_faint == 24.0
    assert job_config.load().star_mag_bright == 9.5        # persisted


def test_star_field_mapped_for_synthetic_generate():
    m = job_config.FASRC_STEP_PARAMS["synthetic_generate"]
    for k in ("star_density_arcmin2", "star_mag_slope",
              "star_mag_bright", "star_mag_faint"):
        assert m[k] == k
