"""Several training steps per compiled call (``Trainer(steps_per_call=)``)
and the device feed (``device_prefetch``): the same training as one step
per call — weights, logged windows, callbacks and per-step learning rates —
calls that respect validation windows and callback cadence, and a finite
dataset that ends cleanly."""
from __future__ import annotations

import csv
import os

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish.training import trainer as trainer_module
from euclid_polish.training.lr_schedule import WarmupCosineDecay
from euclid_polish.training.models.wdsr import wdsr
from euclid_polish.training.trainer import Trainer


def _model(seed=7):
    tf.keras.utils.set_random_seed(seed)
    return wdsr(scale=2, nchan_in=4, nchan_out=1, num_res_blocks=1, num_filters=4)


def _pairs(n=6, seed=2, batch=2, repeat=True):
    rng = np.random.default_rng(seed)
    lr = rng.normal(size=(n, 8, 8, 4)).astype(np.float32)
    hr = rng.normal(size=(n, 16, 16, 1)).astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((lr, hr)).batch(batch)
    return ds.repeat() if repeat else ds


def _run(tmp_path, name, *, steps_per_call, steps=23, evaluate_every=10, callback_every=5,
         train=None):
    trainer = Trainer(_model(), checkpoint_dir=str(tmp_path / name), seed=11,
                      learning_rate=WarmupCosineDecay(peak_lr=1e-3, final_lr=1e-5,
                                                      warmup_steps=6, total_steps=steps),
                      steps_per_call=steps_per_call)
    calls = []
    trainer.train(train if train is not None else _pairs(), _pairs(n=2, seed=10, batch=1, repeat=False),
                  steps=steps, evaluate_every=evaluate_every, save_best_only=False,
                  validate_images=2, step_callback=lambda s, total: calls.append(s),
                  step_callback_every=callback_every)
    with open(os.path.join(str(tmp_path / name), "training_log.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    return trainer, calls, rows


@pytest.mark.parametrize("per_call", [4, 50])
def test_several_steps_per_call_train_exactly_like_one(tmp_path, per_call):
    one, calls_one, rows_one = _run(tmp_path, "one", steps_per_call=1)
    many, calls_many, rows_many = _run(tmp_path, f"many{per_call}", steps_per_call=per_call)
    assert int(many.checkpoint.step.numpy()) == int(one.checkpoint.step.numpy()) == 23
    assert calls_many == calls_one == [1, 5, 10, 15, 20]
    assert [r["step"] for r in rows_many] == [r["step"] for r in rows_one] == ["10", "20"]
    for a, b in zip(rows_one, rows_many, strict=True):
        for key in ("loss", "psnr_stretched", "gnorm_avg", "gnorm_max"):
            assert float(a[key]) == pytest.approx(float(b[key]), rel=1e-5, abs=1e-7), key
    for a, b in zip(one.model.trainable_variables, many.model.trainable_variables, strict=True):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-5, atol=1e-7)
    # The last step's learning rate is what both left on the optimiser.
    np.testing.assert_allclose(float(many.checkpoint.optimizer.learning_rate.numpy()),
                               float(one.checkpoint.optimizer.learning_rate.numpy()), rtol=1e-6)


def test_each_step_in_a_call_takes_its_own_learning_rate(tmp_path):
    """A constant-LR model makes the update size proportional to the LR, so
    a wrong per-step LR inside a call shows up in the weights: compare a
    4-steps-per-call run with a run that sets each step's LR itself."""
    schedule = WarmupCosineDecay(peak_lr=1e-2, final_lr=1e-4, warmup_steps=3, total_steps=8)
    ref = Trainer(_model(), checkpoint_dir=str(tmp_path / "ref"), learning_rate=schedule)
    it = iter(_pairs())
    for step in range(1, 9):
        ref._apply_lr(step)
        ref.train_step(*next(it))
    many = Trainer(_model(), checkpoint_dir=str(tmp_path / "many"), learning_rate=schedule,
                   steps_per_call=4)
    many.train(_pairs(), _pairs(n=2, seed=10, batch=1, repeat=False), steps=8,
               evaluate_every=8, save_best_only=False, validate_images=2)
    for a, b in zip(ref.model.trainable_variables, many.model.trainable_variables, strict=True):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-5, atol=1e-7)


def test_a_finite_dataset_ends_the_run_cleanly(tmp_path):
    """Seven batches, more steps asked for: the run stops at seven."""
    trainer = Trainer(_model(), checkpoint_dir=str(tmp_path / "finite"), steps_per_call=4)
    trainer.train(_pairs(n=14, batch=2, repeat=False), _pairs(n=2, seed=10, batch=1, repeat=False),
                  steps=50, evaluate_every=100, save_best_only=False, validate_images=2)
    assert int(trainer.checkpoint.step.numpy()) == 7


@pytest.mark.parametrize(("step", "expected"), [
    (0, 1),        # the run's first step alone (its callback fires at once)
    (1, 4),        # up to the callback mark at 5
    (5, 5),        # a full call
    (18, 2),       # stops at the window's end, 20
    (20, 3),       # never past the run's 23 steps
])
def test_call_length_respects_windows_cadence_and_the_end(step, expected):
    n = Trainer._call_length(step, start_step=0, steps=23 if step == 20 else 100, per_call=5,
                             evaluate_every=20, callback_every=5)
    assert n == expected


def test_call_length_without_a_callback_and_on_resume():
    assert Trainer._call_length(1000, start_step=1000, steps=5000, per_call=50,
                                evaluate_every=1000, callback_every=0) == 1
    assert Trainer._call_length(1001, start_step=1000, steps=5000, per_call=50,
                                evaluate_every=1000, callback_every=0) == 50
    assert Trainer._call_length(1990, start_step=1000, steps=5000, per_call=50,
                                evaluate_every=1000, callback_every=0) == 10


def test_device_prefetch_without_a_gpu_is_the_plain_feed(tmp_path):
    trainer = Trainer(_model(), checkpoint_dir=str(tmp_path / "feed"), device_prefetch=True)
    if tf.config.list_logical_devices("GPU"):
        pytest.skip("a GPU is present: the feed copies to it")
    first = next(trainer._feed(_pairs()))
    np.testing.assert_array_equal(first[0].numpy(), next(iter(_pairs()))[0].numpy())


def test_steps_per_call_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="steps_per_call"):
        Trainer(_model(), checkpoint_dir=str(tmp_path / "bad"), steps_per_call=0)


def test_the_fused_xla_step_trains_like_the_unfused_one(tmp_path, monkeypatch):
    """The whole step as one XLA function: the same weights, window
    statistics and step count, the spike watch's warmup gate included (its
    step number now comes in as an argument)."""
    monkeypatch.setattr(trainer_module, "GRAD_SPIKE_SKIP_WARMUP_STEPS", 5)
    runs = {}
    for fused in (False, True):
        tf.keras.utils.set_random_seed(7)
        net = wdsr(scale=2, nchan_in=4, nchan_out=1, num_res_blocks=1, num_filters=4,
                   eager_weight_norm_init=True)
        t = Trainer(net, checkpoint_dir=str(tmp_path / f"fused{fused}"), seed=11,
                    learning_rate=WarmupCosineDecay(peak_lr=1e-3, final_lr=1e-5, warmup_steps=4,
                                                    total_steps=12),
                    jit_compile=True, steps_per_call=4, fused_step=fused)
        assert (t._fused_step is not None) == fused
        # No validation inside: the window keeps every step.
        t.train(_pairs(), _pairs(n=2, seed=10, batch=1, repeat=False), steps=12,
                evaluate_every=100, save_best_only=False, validate_images=2)
        runs[fused] = t
    plain, fused = runs[False], runs[True]
    assert int(fused.checkpoint.step.numpy()) == int(plain.checkpoint.step.numpy()) == 12
    for name in ("_win_loss_sum", "_win_gnorm_sum", "_win_gnorm_max", "_win_spike_norm", "_win_steps"):
        np.testing.assert_allclose(getattr(fused, name).numpy(), getattr(plain, name).numpy(),
                                   rtol=1e-4, err_msg=name)
    assert float(fused._win_spike_norm.numpy()) > 0.0              # steps 6–12 watched
    for a, b in zip(plain.model.trainable_variables, fused.model.trainable_variables, strict=True):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-4, atol=1e-6)


def test_fused_needs_xla_and_several_steps_per_call(tmp_path):
    assert Trainer(_model(), checkpoint_dir=str(tmp_path / "a"), steps_per_call=4,
                   fused_step=True)._fused_step is None                      # no XLA
    assert Trainer(_model(), checkpoint_dir=str(tmp_path / "b"), jit_compile=True,
                   fused_step=True)._fused_step is None                      # one step per call


def test_a_finite_dataset_ends_the_fused_loop_with_the_right_count(tmp_path):
    tf.keras.utils.set_random_seed(7)
    net = wdsr(scale=2, nchan_in=4, nchan_out=1, num_res_blocks=1, num_filters=4,
               eager_weight_norm_init=True)
    trainer = Trainer(net, checkpoint_dir=str(tmp_path / "fused_finite"), jit_compile=True,
                      steps_per_call=4, fused_step=True)
    trainer.train(_pairs(n=14, batch=2, repeat=False), _pairs(n=2, seed=10, batch=1, repeat=False),
                  steps=50, evaluate_every=100, save_best_only=False, validate_images=2)
    assert int(trainer.checkpoint.step.numpy()) == 7
    assert float(trainer._win_steps.numpy()) == 7.0
