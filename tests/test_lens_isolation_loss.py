from __future__ import annotations

import numpy as np
import tensorflow as tf

from euclid_polish.experiments.lens_isolation.loss import LensIsolationLoss


def test_perfect_prediction_is_zero_and_negative_residual_is_penalized():
    loss = LensIsolationLoss(lens_weight=4, flux_weight=0.2)
    target = tf.constant(np.zeros((1, 4, 4, 1), np.float32))
    assert float(tf.reduce_mean(loss(target, target))) == 0
    assert float(tf.reduce_mean(loss(target, tf.ones_like(target)))) > 0


def test_zero_positive_prediction_has_nonzero_loss_and_gradient():
    loss = LensIsolationLoss(lens_weight=4, flux_weight=0.2)
    target = tf.constant(np.ones((1, 4, 4, 1), np.float32))
    prediction = tf.Variable(tf.zeros_like(target))
    with tf.GradientTape() as tape:
        value = loss(target, prediction)
    gradient = tape.gradient(value, prediction)
    assert float(tf.reduce_mean(value)) > 0
    assert gradient is not None
    assert float(tf.reduce_sum(tf.abs(gradient))) > 0


def test_flux_term_improves_as_positive_flux_is_retained():
    loss = LensIsolationLoss(lens_weight=1, flux_weight=1)
    target = tf.ones((1, 4, 4, 1), tf.float32)
    empty = float(tf.reduce_mean(loss(target, tf.zeros_like(target))))
    half = float(tf.reduce_mean(loss(target, target * 0.5)))
    full = float(tf.reduce_mean(loss(target, target)))
    assert empty > half > full
