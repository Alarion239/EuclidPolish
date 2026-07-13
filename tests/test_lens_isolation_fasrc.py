from euclid_polish.web.fasrc_pipeline import REGISTRY


def test_lens_isolation_steps_are_additive_and_isolated():
    generate = REGISTRY.get("lens_isolation_generate")
    train = REGISTRY.get("lens_isolation_train")
    evaluate = REGISTRY.get("lens_isolation_evaluate")
    assert generate.needs_gpu is False
    assert train.needs_gpu is evaluate.needs_gpu is True
    assert generate.build_command({"ntrain": 4, "nvalid": 2, "ntest": 2})[0].endswith(
        "lens_isolation_generate.py"
    )
    train_cmd = train.build_command({"sources": "member_01"})
    eval_cmd = evaluate.build_command({})
    joined = " ".join(generate.build_command({}) + train_cmd + eval_cmd)
    assert joined.count("lens_isolation_") == 3
    assert "records_v2" not in joined


def test_existing_ensemble_step_is_still_registered():
    assert REGISTRY.get("ensemble_train").step_id == "ensemble_train"
