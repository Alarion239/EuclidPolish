"""Collapse-resistant balanced reconstruction objective."""

from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="euclid_polish")
class LensIsolationLoss(tf.keras.losses.Loss):
    """Per-example weighted MAE plus positive-system flux retention."""

    def __init__(
        self,
        lens_weight: float = 8.0,
        flux_weight: float = 0.1,
        name: str = "lens_isolation_loss",
        **kwargs,
    ) -> None:
        kwargs.setdefault("reduction", tf.keras.losses.Reduction.NONE)
        super().__init__(name=name, **kwargs)
        self.lens_weight = float(lens_weight)
        self.flux_weight = float(flux_weight)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        signal = tf.cast(y_true > 0, y_pred.dtype)
        weights = 1.0 + self.lens_weight * signal
        reconstruction = tf.reduce_sum(tf.abs(y_true - y_pred) * weights, axis=(1, 2, 3))
        reconstruction /= tf.maximum(tf.reduce_sum(weights, axis=(1, 2, 3)), 1.0)

        target_flux = tf.reduce_sum(tf.nn.relu(y_true), axis=(1, 2, 3))
        prediction_flux = tf.reduce_sum(tf.nn.relu(y_pred), axis=(1, 2, 3))
        positive = tf.cast(target_flux > 0, y_pred.dtype)
        relative_flux_error = tf.abs(prediction_flux - target_flux) / tf.maximum(
            target_flux, tf.cast(1e-6, y_pred.dtype)
        )
        flux = positive * relative_flux_error
        return reconstruction + self.flux_weight * flux

    def get_config(self):
        return {
            **super().get_config(),
            "lens_weight": self.lens_weight,
            "flux_weight": self.flux_weight,
        }
