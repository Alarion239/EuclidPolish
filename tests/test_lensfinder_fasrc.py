"""The lens-finder FASRC steps are registered with the right env/resources."""

from __future__ import annotations

from euclid_polish.web.fasrc_pipeline import REGISTRY


def test_build_stamps_step_uses_main_env_gpu():
    s = REGISTRY.get("lensfinder_build_stamps")
    assert s.needs_gpu is True
    assert s.conda_env is None                   # SR inference → main TF env
    cmd = s.build_command({})
    assert cmd[0] == "scripts/lensfinder_build_stamps.py"
    assert "--stamp-m" in cmd


def test_train_step_uses_zoobot_env_gpu():
    s = REGISTRY.get("lensfinder_train")
    assert s.needs_gpu is True
    assert s.conda_env == "EuclidPolishZoobot"   # the critical env switch
    cmd = s.build_command({})
    assert cmd[0] == "scripts/lensfinder_train.py"
    assert "--recon" in cmd


def test_build_command_honors_params():
    s = REGISTRY.get("lensfinder_build_stamps")
    cmd = s.build_command({"stamp_m": 256, "max_fields": 50})
    assert "256" in cmd and "--max-fields" in cmd and "50" in cmd


def test_generate_step_uses_dedicated_records_dir():
    from euclid_polish.web import job_config as jc

    s = REGISTRY.get("lensfinder_generate")
    assert s.conda_env is None and s.job_name == "lensfinder-data"
    params = jc.fasrc_params_for("lensfinder_generate")
    cmd = s.build_command(dict(params))
    # never clobbers the main training set
    assert "--records-dir" in cmd
    assert "data/images/records_lensfinder" in cmd
    # field count + size come from the dedicated config-panel section
    assert cmd[cmd.index("--ntrain") + 1] == str(params["n_train"])
    assert cmd[cmd.index("--image-size") + 1] == str(params["image_size"])


def test_jobconfig_has_lensfinder_section_defaults():
    from euclid_polish.web.job_config import JobConfig

    c = JobConfig()
    assert c.lensfinder_n_fields == 800
    assert c.lensfinder_image_size == 2040
