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
    assert b'id="root"' in r.data


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


def test_galaxy_density_defaults_updates_and_maps_to_generators(cfg_path):
    c = job_config.load()
    assert c.galaxy_density_arcmin2 == 245.0
    c = job_config.update({"galaxy_density_arcmin2": "175"})
    assert c.galaxy_density_arcmin2 == 175.0
    assert job_config.load().galaxy_density_arcmin2 == 175.0
    for step_id in ("synthetic_generate", "lensfinder_generate"):
        assert (
            job_config.FASRC_STEP_PARAMS[step_id]["galaxy_density_arcmin2"]
            == "galaxy_density_arcmin2"
        )


def test_psf_warp_mapped_for_all_generation_and_training_steps():
    keys = ("psf_warp_prob", "psf_warp_alpha_max", "psf_warp_sigma",
            "saturation_mask_prob")
    for step_id in (
        "synthetic_generate", "lensfinder_generate", "ensemble_train",
    ):
        mapping = job_config.FASRC_STEP_PARAMS[step_id]
        for key in keys:
            assert mapping[key] == key


def test_lensfinder_training_defaults_and_update(cfg_path):
    c = job_config.load()
    assert c.lensfinder_epochs == 10          # mirrors scripts/lensfinder_train.py
    assert c.lensfinder_patience == 6
    assert c.lensfinder_batch_size == 64
    assert c.lensfinder_learning_rate == 1e-4
    c = job_config.update({"lensfinder_epochs": "100", "lensfinder_patience": "10",
                           "lensfinder_batch_size": "32",
                           "lensfinder_learning_rate": "5e-5"})
    assert c.lensfinder_epochs == 100 and c.lensfinder_patience == 10
    assert c.lensfinder_batch_size == 32
    assert c.lensfinder_learning_rate == 5e-5   # coerced as float, not int
    assert job_config.load().lensfinder_epochs == 100   # persisted


def test_lensfinder_training_mapped_for_train_step():
    m = job_config.FASRC_STEP_PARAMS["lensfinder_train"]
    assert m == {"epochs": "lensfinder_epochs",
                 "patience": "lensfinder_patience",
                 "batch_size": "lensfinder_batch_size",
                 "learning_rate": "lensfinder_learning_rate",
                 "training_mode": "lensfinder_training_mode"}


def test_save_endpoint_persists_lensfinder_fields(client, cfg_path):
    # These fields were previously dropped by the route's hand-maintained
    # allowlist; the whole form is now forwarded to update().
    r = client.post("/api/config/save", data={
        "lensfinder_n_fields": "1200", "lensfinder_epochs": "80",
        "lensfinder_patience": "9", "lensfinder_learning_rate": "2e-4",
    })
    assert r.status_code == 200 and r.get_json()["ok"] is True
    c = job_config.load()
    assert c.lensfinder_n_fields == 1200      # dataset field now persists too
    assert c.lensfinder_epochs == 80
    assert c.lensfinder_patience == 9
    assert c.lensfinder_learning_rate == 2e-4


def test_config_page_renders_training_section(client, cfg_path):
    r = client.get("/config")
    assert r.status_code == 200
    assert b'id="root"' in r.data


def test_training_mode_default_and_string_update(cfg_path):
    c = job_config.load()
    assert c.lensfinder_training_mode == "head_only"      # default
    # String fields must persist through update() — previously dropped because
    # update() coerced every value to int/float.
    c = job_config.update({"lensfinder_training_mode": "full"})
    assert c.lensfinder_training_mode == "full"
    assert job_config.load().lensfinder_training_mode == "full"   # persisted
    # numeric fields still coerce alongside a string field
    c = job_config.update({"lensfinder_training_mode": "head_only",
                           "lensfinder_epochs": "50"})
    assert c.lensfinder_training_mode == "head_only" and c.lensfinder_epochs == 50


def test_training_mode_mapped_for_train_step():
    m = job_config.FASRC_STEP_PARAMS["lensfinder_train"]
    assert m["training_mode"] == "lensfinder_training_mode"


# -- WDSR LR schedule + plateau guard knobs -------------------------------- #

def test_lr_and_plateau_defaults_from_config(cfg_path):
    from euclid_polish.config import Config
    c = job_config.load()
    assert c.lr_peak == Config.LR_PEAK
    assert c.lr_final == Config.LR_FINAL
    assert c.lr_warmup_steps == Config.LR_WARMUP_STEPS
    assert c.plateau_lr_enabled == int(Config.PLATEAU_LR_ENABLED)
    assert c.plateau_lr_metric == Config.PLATEAU_LR_METRIC
    assert c.psf_warp_prob == Config.TRAIN_PSF_WARP_PROB
    assert c.psf_warp_alpha_max == Config.TRAIN_PSF_WARP_ALPHA_MAX
    assert c.psf_warp_sigma == Config.TRAIN_PSF_WARP_SIGMA
    assert c.saturation_mask_prob == Config.TRAIN_SATURATION_MASK_PROB


def test_lr_and_plateau_update_and_persist(cfg_path):
    c = job_config.update({
        "lr_peak": "3e-4", "lr_final": "1e-5", "lr_warmup_steps": "1500",
        "plateau_lr_enabled": "0", "plateau_lr_factor": "0.3",
        "plateau_lr_patience": "8000", "plateau_lr_min_delta": "0.05",
        "plateau_lr_cooldown": "3000", "plateau_lr_min_lr": "1e-7",
        "plateau_lr_metric": "psnr_stretched",
    })
    assert c.lr_peak == 3e-4 and c.lr_final == 1e-5 and c.lr_warmup_steps == 1500
    assert c.plateau_lr_enabled == 0                    # "off" coerced to int 0
    assert c.plateau_lr_factor == 0.3 and c.plateau_lr_patience == 8000
    assert c.plateau_lr_min_delta == 0.05 and c.plateau_lr_cooldown == 3000
    assert c.plateau_lr_min_lr == 1e-7
    assert c.plateau_lr_metric == "psnr_stretched"     # string kept verbatim
    again = job_config.load()
    assert again.lr_peak == 3e-4 and again.plateau_lr_enabled == 0
    assert again.plateau_lr_metric == "psnr_stretched"


def test_lr_plateau_mapped_for_ensemble_train():
    m = job_config.FASRC_STEP_PARAMS["ensemble_train"]
    for k in ("lr_peak", "lr_final", "lr_warmup_steps", "plateau_lr_enabled",
              "plateau_lr_factor", "plateau_lr_patience", "plateau_lr_min_delta",
              "plateau_lr_cooldown", "plateau_lr_min_lr", "plateau_lr_metric"):
        assert m[k] == k                               # identity mapping
    for k in ("psf_warp_prob", "psf_warp_alpha_max", "psf_warp_sigma",
              "saturation_mask_prob"):
        assert m[k] == k


def test_saturation_mask_probability_is_capped_at_half(cfg_path):
    c = job_config.update({"saturation_mask_prob": "0.9"})
    assert c.saturation_mask_prob == 0.5
    assert job_config.load().saturation_mask_prob == 0.5


def test_config_page_renders_lr_plateau_section(client, cfg_path):
    r = client.get("/config")
    assert r.status_code == 200
    assert b'id="root"' in r.data


def test_ensemble_train_build_command_injects_lr_plateau_flags(monkeypatch):
    from euclid_polish.web.fasrc_pipeline import EnsembleTrainStep
    monkeypatch.setattr(
        "euclid_polish.web.fasrc_pipeline.next_member_names",
        lambda base, k: [f"member_{i:02d}" for i in range(k)])
    params = {
        "n_members": "5", "steps": "100000",
        "lr_peak": "3e-4", "lr_warmup_steps": "1500",
        "plateau_lr_enabled": "0", "plateau_lr_metric": "psnr_stretched",
    }
    cmd = EnsembleTrainStep().build_command(params)
    assert "--lr-peak" in cmd and cmd[cmd.index("--lr-peak") + 1] == "3e-4"
    assert "--lr-warmup-steps" in cmd
    assert cmd[cmd.index("--plateau-lr-enabled") + 1] == "0"
    assert cmd[cmd.index("--plateau-lr-metric") + 1] == "psnr_stretched"
    # Absent knobs contribute no flags (blank-safe).
    assert "--lr-final" not in cmd
