"""The lens-finder FASRC steps are registered with the right env/resources."""

from __future__ import annotations

from euclid_polish.web.fasrc_pipeline import REGISTRY


def test_build_stamps_step_runs_cpu_shared_main_env():
    s = REGISTRY.get("lensfinder_build_stamps")
    # SR here is sequential batch-1 inference interleaved with CPU stamp/PNG
    # work — the GPU would sit idle, so it defaults to the CPU shared partition.
    assert s.needs_gpu is False
    assert s.defaults.partition == "shared"
    assert s.defaults.n_gpus == 0
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
    cmd = s.build_command({"stamp_m": 200, "max_fields": 50})
    assert "200" in cmd and "--max-fields" in cmd and "50" in cmd


def test_build_stamps_defaults_106_and_drops_checkpoint():
    s = REGISTRY.get("lensfinder_build_stamps")
    cmd = s.build_command({})
    assert cmd[cmd.index("--stamp-m") + 1] == "106"      # 424/4
    assert "--checkpoint" not in cmd                     # moved to sr_infer
    assert "--num-res-blocks" not in cmd


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


def test_train_step_emits_patience_param():
    s = REGISTRY.get("lensfinder_train")
    cmd = s.build_command({"patience": 9})
    assert cmd[cmd.index("--patience") + 1] == "9"


def test_sr_infer_step_is_gpu_main_env():
    s = REGISTRY.get("lensfinder_sr_infer")
    assert s.needs_gpu is True and s.defaults.partition == "gpu"
    assert s.defaults.n_gpus == 1
    assert s.conda_env is None                     # SR inference -> main TF env
    assert s.job_name == "lensfinder-sr-infer"
    cmd = s.build_command({"checkpoint": "ckpt/x", "num_res_blocks": 8})
    assert cmd[0] == "scripts/lensfinder_sr_infer.py"
    assert cmd[cmd.index("--checkpoint") + 1] == "ckpt/x"
    assert cmd[cmd.index("--num-res-blocks") + 1] == "8"


def test_train_step_receives_config_training_knobs():
    from euclid_polish.web import job_config as jc

    s = REGISTRY.get("lensfinder_train")
    params = jc.fasrc_params_for("lensfinder_train")   # injected from /config
    cmd = s.build_command(dict(params))
    assert cmd[cmd.index("--epochs") + 1] == str(params["epochs"])
    assert cmd[cmd.index("--patience") + 1] == str(params["patience"])
    assert cmd[cmd.index("--batch-size") + 1] == str(params["batch_size"])
    assert cmd[cmd.index("--learning-rate") + 1] == str(params["learning_rate"])
