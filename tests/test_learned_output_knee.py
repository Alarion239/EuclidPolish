"""Learned output knee: a single-image multi-knee member whose last layer
learns one asinh output knee per band and outputs electrons."""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf
from tf_keras.layers import Input, Lambda
from tf_keras.models import Model as KerasModel

from euclid_polish.training.inference import (
    infer_checkpoint_learned_output_knee,
    infer_checkpoint_nchan_in,
    infer_checkpoint_nchan_out,
    infer_checkpoint_num_res_blocks,
    load_model_from_checkpoint,
)
from euclid_polish.training.models.output_knee import (
    KNEE_MAX_E,
    KNEE_MIN_E,
    LearnedOutputKnee,
    knee_logit_for,
    knees_from_logits,
    learned_output_knees,
)
from euclid_polish.training.models.wdsr import wdsr

KNEES = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def test_head_starts_at_its_initial_knee_in_every_band():
    head = LearnedOutputKnee(init_knee_e=10.0)
    head.build((None, None, None, 4))
    np.testing.assert_allclose(head.knees().numpy(), [10.0] * 4, rtol=1e-5)


def test_head_maps_asinh_values_to_electrons_with_its_knee():
    head = LearnedOutputKnee(init_knee_e=10.0)
    y = tf.constant(np.linspace(-3.0, 25.0, 32).reshape(1, 2, 4, 4).astype(np.float32))
    expected = 10.0 * np.sinh(np.clip(y.numpy(), -20.0, 20.0))
    np.testing.assert_allclose(head(y).numpy(), expected, rtol=1e-5)


@pytest.mark.parametrize("logit", [-1.0e4, 1.0e4])
def test_head_knee_stays_inside_its_bounds(logit):
    head = LearnedOutputKnee()
    head.build((None, None, None, 4))
    head.output_knee_logit.assign([logit] * 4)
    k = head.knees().numpy()
    assert np.all(k >= KNEE_MIN_E * (1 - 1e-5)) and np.all(k <= KNEE_MAX_E * (1 + 1e-5))


def test_head_knee_gets_a_gradient():
    head = LearnedOutputKnee()
    y = tf.constant(np.full((1, 2, 2, 4), 3.0, np.float32))
    with tf.GradientTape() as tape:
        total = tf.reduce_sum(head(y))
    grad = tape.gradient(total, head.output_knee_logit)
    assert grad is not None and np.all(np.abs(grad.numpy()) > 0)


def test_knee_logit_round_trips_and_rejects_knees_outside_the_bounds():
    for knee in (0.5, 10.0, 3000.0):
        assert knees_from_logits([knee_logit_for(knee)])[0] == pytest.approx(knee, rel=1e-6)
    for knee in (KNEE_MIN_E, KNEE_MAX_E, 0.0, -1.0):
        with pytest.raises(ValueError):
            knee_logit_for(knee)


def test_learned_output_knees_finds_the_head_of_a_model():
    inp = Input(shape=(None, None, 4))
    model = KerasModel(inp, LearnedOutputKnee(init_knee_e=30.0)(inp))
    np.testing.assert_allclose(learned_output_knees(model), [30.0] * 4, rtol=1e-5)
    plain = KerasModel(inp, Lambda(lambda t: t)(inp))
    assert learned_output_knees(plain) is None


def _learned_wdsr(blocks: int = 1, knee: float = 10.0):
    return wdsr(scale=2, num_res_blocks=blocks, nchan_in=24, nchan_out=4, input_knees=6,
                learned_output_knee=knee)


def _x24(seed: int):
    return tf.constant(np.random.default_rng(seed).normal(size=(1, 6, 6, 24)).astype(np.float32))


def test_wdsr_learned_head_outputs_electrons_from_the_stretched_image():
    model = _learned_wdsr()
    assert isinstance(model.layers[-1], LearnedOutputKnee)
    pre = KerasModel(model.inputs, model.layers[-2].output)
    x = _x24(1)
    expected = 10.0 * np.sinh(np.clip(pre(x).numpy(), -20.0, 20.0))
    np.testing.assert_allclose(model(x).numpy(), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("blocks", [1, 2])
def test_checkpoint_with_a_learned_head_keeps_depth_channels_and_knees(tmp_path, blocks):
    learned = (3.0, 7.0, 20.0, 400.0)
    model = _learned_wdsr(blocks=blocks)
    model.layers[-1].output_knee_logit.assign([knee_logit_for(q) for q in learned])
    d = str(tmp_path / "ckpt")
    tf.train.Checkpoint(model=model).save(d + "/ckpt")
    assert infer_checkpoint_num_res_blocks(d) == blocks
    assert infer_checkpoint_nchan_in(d) == 24
    assert infer_checkpoint_nchan_out(d, scale=2, nchan_in=24) == 4
    assert infer_checkpoint_learned_output_knee(d) == pytest.approx(learned, rel=1e-5)
    loaded = load_model_from_checkpoint(d, scale=2, num_res_blocks=32)
    np.testing.assert_allclose(learned_output_knees(loaded), learned, rtol=1e-5)
    x = _x24(2)
    np.testing.assert_allclose(loaded(x).numpy(), model(x).numpy(), rtol=1e-5, atol=1e-5)


def test_checkpoint_without_the_head_reports_no_learned_knee(tmp_path):
    model = wdsr(scale=2, num_res_blocks=1, nchan_in=24, nchan_out=4, input_knees=6)
    d = str(tmp_path / "ckpt")
    tf.train.Checkpoint(model=model).save(d + "/ckpt")
    assert infer_checkpoint_learned_output_knee(d) is None
    assert infer_checkpoint_num_res_blocks(d) == 1
    assert learned_output_knees(load_model_from_checkpoint(d, scale=2, num_res_blocks=1)) is None
