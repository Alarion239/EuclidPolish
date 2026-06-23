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
