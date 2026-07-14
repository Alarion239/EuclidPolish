from euclid_polish.web.fasrc_pipeline import REGISTRY


def test_lens_isolation_steps_are_additive_and_use_only_normal_record_controls():
    generate = REGISTRY.get("lens_isolation_generate")
    train = REGISTRY.get("lens_isolation_train")
    evaluate = REGISTRY.get("lens_isolation_evaluate")
    assert generate.needs_gpu is False
    assert train.needs_gpu is evaluate.needs_gpu is True
    generate_cmd = generate.build_command({"n_cpus": 32, "ntrain": 4, "nvalid": 2, "ntest": 2})
    assert generate_cmd[0].endswith("lens_isolation_generate.py")
    assert generate_cmd[generate_cmd.index("--workers") + 1] == "32"
    assert "--force" not in generate_cmd
    assert "--force" in generate.build_command(
        {"n_cpus": 16, "ntrain": 4, "nvalid": 2, "ntest": 2, "force": "1"}
    )
    train_cmd = train.build_command(
        {"sources": "member_01", "loss_norm": "l2", "lens_weight": "8", "crops_per_field": "16"}
    )
    eval_cmd = evaluate.build_command({"seed": "7", "crop_size": "96", "limit": "10"})
    joined = " ".join(generate_cmd + train_cmd + eval_cmd)
    assert joined.count("lens_isolation_") == 3
    assert "records_v2" not in joined
    assert "--loss-norm" in train_cmd
    assert "--lens-weight" not in train_cmd
    assert "--crops-per-field" not in train_cmd
    assert "--force" not in train_cmd
    assert "--force" in train.build_command({"sources": "member_01", "force": "1"})
    assert "--seed" in eval_cmd
    assert "--crop-size" in eval_cmd


def test_existing_ensemble_step_is_still_registered():
    assert REGISTRY.get("ensemble_train").step_id == "ensemble_train"
