"""Milestone checkpoints (``Trainer.train(milestones=)``, ``--milestones``,
the ensemble_train ``milestones`` task param): the state at each milestone
step is kept for good in ``milestones/step_<N>/``, a checkpoint dir of its
own that restores the exact weights of that step; milestones must be
multiples of the validation period; the CLI and the console command carry
them."""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish.training.models.wdsr import wdsr
from euclid_polish.training.trainer import Trainer
from euclid_polish.web.fasrc_pipeline import REGISTRY
from scripts.train_ensemble import _milestones
from tests.test_step_task_params import hermetic_build  # noqa: F401 — the fixture


def _model(seed=7):
    tf.keras.utils.set_random_seed(seed)
    return wdsr(scale=2, nchan_in=4, nchan_out=1, num_res_blocks=1, num_filters=4)


def _pairs(n=6, seed=2, batch=2, repeat=True):
    rng = np.random.default_rng(seed)
    lr = rng.normal(size=(n, 8, 8, 4)).astype(np.float32)
    hr = rng.normal(size=(n, 16, 16, 1)).astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((lr, hr)).batch(batch)
    return ds.repeat() if repeat else ds


def _step_of(path):
    return int(tf.train.load_checkpoint(path).get_tensor("step/.ATTRIBUTES/VARIABLE_VALUE"))


def test_each_milestone_keeps_its_own_state_for_good(tmp_path):
    d = tmp_path / "run"
    kept = {}

    def snapshot(row):            # the weights right after each validation
        kept[int(row["step"])] = [v.numpy().copy() for v in trainer.model.trainable_variables]

    trainer = Trainer(_model(), checkpoint_dir=str(d), steps_per_call=4)
    trainer.train(_pairs(), _pairs(n=2, seed=10, batch=1, repeat=False), steps=16, evaluate_every=4,
                  save_best_only=True, validate_images=2, milestones=(4, 12, 40), eval_callback=snapshot)
    folders = sorted(p.name for p in (d / "milestones").iterdir())
    assert folders == ["step_000000004", "step_000000012"]             # 40 > steps: ignored
    for step in (4, 12):
        path = tf.train.latest_checkpoint(str(d / "milestones" / f"step_{step:09d}"))
        assert path is not None and path.endswith(f"ckpt-{step}") and _step_of(path) == step
        restored = Trainer(_model(seed=99), checkpoint_dir=str(tmp_path / f"other{step}"))
        tf.train.Checkpoint(model=restored.model).restore(path).expect_partial()
        for a, b in zip(restored.model.trainable_variables, kept[step], strict=True):
            np.testing.assert_array_equal(a.numpy(), b)
    # the rotating tracks are untouched: latest/final still hold the end
    assert _step_of(tf.train.latest_checkpoint(str(d / "final"))) == 16


def test_milestones_must_be_reachable_at_a_validation(tmp_path):
    trainer = Trainer(_model(), checkpoint_dir=str(tmp_path / "bad"))
    with pytest.raises(ValueError, match="multiples of evaluate_every"):
        trainer.train(_pairs(), _pairs(n=2, seed=10, batch=1, repeat=False), steps=8, evaluate_every=4,
                      validate_images=2, milestones=(6,))


def test_the_cli_parses_and_checks_milestones():
    assert _milestones("1000000,200000, 500000", 5000) == (200000, 500000, 1000000)
    assert _milestones(None, 5000) == () and _milestones("", 5000) == ()
    for bad in ("200000,123", "x", "-5000"):
        with pytest.raises(SystemExit):
            _milestones(bad, 5000)


def test_the_console_command_carries_milestones(hermetic_build):  # noqa: F811
    step = REGISTRY.get("ensemble_train")
    params = step.fill_task_params({"evaluate_every": "5000", "milestones": "200000, 500000,1000000"})
    joined = " ".join(step.build_command(params))
    assert "--evaluate-every 5000" in joined and "--milestones 200000,500000,1000000" in joined
    assert "--milestones" not in " ".join(step.build_command(step.fill_task_params({})))
    with pytest.raises(ValueError, match="milestones"):
        step.build_command(step.fill_task_params({"milestones": "2e5"}))
