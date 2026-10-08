"""Learned output knee: a single-image multi-knee member whose last layer
learns one asinh output knee per band and outputs electrons."""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf
from tf_keras.layers import Input, Lambda
from tf_keras.models import Model as KerasModel

from euclid_polish.training.models.output_knee import (
    KNEE_MAX_E,
    KNEE_MIN_E,
    LearnedOutputKnee,
    knee_logit_for,
    knees_from_logits,
    learned_output_knees,
)

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
