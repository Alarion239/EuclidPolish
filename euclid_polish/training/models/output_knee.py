"""The learned output knee of a single-image multi-knee member.

A single-image multi-knee member predicts one 4-band image ``y`` in asinh
space; its electrons are ``x = k · sinh(y)``. With a fixed ``k`` (the
``output_knee``, 10 e⁻ for members 195/196) the knee is a hand-picked number.
:class:`LearnedOutputKnee` makes ``k`` one trainable value per band, so the
member learns where its output turns from linear to logarithmic. The layer is
the model's last, so the model outputs electrons.

The knee is bounded to the knee-integrated PSNR range, ``KNEE_MIN_E`` to
``KNEE_MAX_E``, through a sigmoid (no dead gradient at a hard clip), and its
logit is scaled by ``KNEE_LOGIT_SCALE`` so it moves slower than the network's
weights. Checkpoint introspection finds the head by its weight's attribute
name, :data:`KNEE_WEIGHT_NAME`.
"""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf
from tf_keras.initializers import Constant
from tf_keras.layers import Layer

#: Attribute name of the trainable weight: the checkpoint key
#: (``model/layer_with_weights-N/output_knee_logit/…``) introspection looks for.
KNEE_WEIGHT_NAME = "output_knee_logit"
#: Layer name, so a built model's head can be found.
LAYER_NAME = "learned_output_knee"
#: Bounds of the learned knee (electrons): the knee-integrated PSNR range.
KNEE_MIN_E = 0.1
KNEE_MAX_E = 1.0e4
#: Starting knee a loader builds the head with (the restore overwrites it).
DEFAULT_INIT_KNEE_E = 10.0
#: The logit is multiplied by this before the sigmoid, so under Adam the knee
#: moves about ten times slower than an unscaled logit would.
KNEE_LOGIT_SCALE = 0.1
#: Stretched values are clipped to ±this before ``sinh`` (as every member's
#: inference does), so the output stays finite.
SINH_CLIP = 20.0

_LOG_MIN = math.log(KNEE_MIN_E)
_LOG_SPAN = math.log(KNEE_MAX_E) - math.log(KNEE_MIN_E)


def knee_logit_for(knee_e: float) -> float:
    """The weight value at which the head's knee equals ``knee_e`` (e⁻)."""
    knee = float(knee_e)
    if not KNEE_MIN_E < knee < KNEE_MAX_E:
        raise ValueError(f"a learned output knee must start inside "
                         f"({KNEE_MIN_E:g}, {KNEE_MAX_E:g}) e⁻, got {knee_e!r}")
    frac = (math.log(knee) - _LOG_MIN) / _LOG_SPAN
    return math.log(frac / (1.0 - frac)) / KNEE_LOGIT_SCALE


def knees_from_logits(logits) -> np.ndarray:
    """Knees (e⁻) of the head's weight values (NumPy, for checkpoints)."""
    z = KNEE_LOGIT_SCALE * np.asarray(logits, np.float64)
    return np.exp(_LOG_MIN + _LOG_SPAN / (1.0 + np.exp(-z)))


class LearnedOutputKnee(Layer):
    """``x_b = k_b · sinh(clip(y_b, ±SINH_CLIP))`` with one trainable knee
    ``k_b`` per band, bounded to ``(KNEE_MIN_E, KNEE_MAX_E)`` electrons."""

    def __init__(self, init_knee_e: float = DEFAULT_INIT_KNEE_E, **kwargs):
        kwargs.setdefault("name", LAYER_NAME)
        super().__init__(**kwargs)
        self.init_knee_e = float(init_knee_e)
        self._init_logit = knee_logit_for(self.init_knee_e)

    def build(self, input_shape):
        self.output_knee_logit = self.add_weight(
            name=KNEE_WEIGHT_NAME, shape=(int(input_shape[-1]),),
            initializer=Constant(self._init_logit), trainable=True)
        super().build(input_shape)

    def knees(self) -> tf.Tensor:
        """The current knee of every band, electrons, ``(bands,)``."""
        z = KNEE_LOGIT_SCALE * self.output_knee_logit
        return tf.exp(_LOG_MIN + _LOG_SPAN * tf.sigmoid(z))

    def call(self, y):
        k = tf.cast(self.knees(), y.dtype)
        return k * tf.sinh(tf.clip_by_value(y, -SINH_CLIP, SINH_CLIP))

    def get_config(self):
        return {**super().get_config(), "init_knee_e": self.init_knee_e}


def learned_output_knees(model) -> np.ndarray | None:
    """The knees (e⁻) of ``model``'s learned output head, else ``None``."""
    for layer in getattr(model, "layers", ()):
        if isinstance(layer, LearnedOutputKnee):
            return layer.knees().numpy()
    return None
