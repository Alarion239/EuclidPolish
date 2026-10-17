"""The learned output knee of a single-image multi-knee member.

A single-image multi-knee member predicts one 4-band image ``y`` in asinh
space; its electrons are ``x = k · sinh(y)``. With a fixed ``k`` (the
``output_knee``, 10 e⁻ for members 195/196) the knee is a hand-picked number.
:class:`LearnedOutputKnee` makes ``k`` one trainable value per band, so the
member learns where its output turns from linear to logarithmic. The layer is
the model's last, so the model outputs electrons.

The knee is bounded through a sigmoid (no dead gradient at a hard clip), and
its logit is scaled by ``KNEE_LOGIT_SCALE`` so it moves slower than the
network's weights. The head stores its bounds as a non-trainable weight, so a
checkpoint says how to decode its own knees: heads saved before the bounds
were stored used ``LEGACY_KNEE_MIN_E``…``KNEE_MAX_E``. A loader builds the
head with the checkpoint's bounds. Checkpoint introspection finds the head by
its weight's attribute name, :data:`KNEE_WEIGHT_NAME`.
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
#: Attribute name of the head's bounds, ``[min, max]`` electrons (not trained).
KNEE_BOUNDS_NAME = "output_knee_bounds_e"
#: Layer name, so a built model's head can be found.
LAYER_NAME = "learned_output_knee"
#: Bounds of a NEW learned knee (electrons). The floor was 0.1 e⁻ until
#: 2026-10-17: member_227's knee was still falling at 0.2 e⁻ after 6.5M
#: steps, and about half of its slowdown came from the sigmoid flattening
#: near that floor.
KNEE_MIN_E = 0.01
KNEE_MAX_E = 1.0e4
#: The floor of every head built before the bounds were stored in it.
LEGACY_KNEE_MIN_E = 0.1
#: Starting knee of a new learned head (e⁻) when a member spec names none.
#: 10 e⁻ until 2026-10-17: every member then spent its run dragging the knee
#: down (about 2.5 e⁻ after 200k steps, member_227 0.2 e⁻ at 6.5M).
DEFAULT_INIT_KNEE_E = 0.1
#: The logit is multiplied by this before the sigmoid, so under Adam the knee
#: moves about ten times slower than an unscaled logit would.
KNEE_LOGIT_SCALE = 0.1
#: Stretched values are clipped to ±this before ``sinh`` (as every member's
#: inference does), so the output stays finite.
SINH_CLIP = 20.0


def knee_logit_for(knee_e: float, min_e: float = KNEE_MIN_E,
                   max_e: float = KNEE_MAX_E) -> float:
    """The weight value at which a head bounded to ``(min_e, max_e)`` has
    knee ``knee_e`` (e⁻)."""
    knee = float(knee_e)
    if not min_e < knee < max_e:
        raise ValueError(f"a learned output knee must start inside "
                         f"({min_e:g}, {max_e:g}) e⁻, got {knee_e!r}")
    frac = (math.log(knee) - math.log(min_e)) / (math.log(max_e) - math.log(min_e))
    return math.log(frac / (1.0 - frac)) / KNEE_LOGIT_SCALE


def knees_from_logits(logits, min_e: float, max_e: float) -> np.ndarray:
    """Knees (e⁻) of the head's weight values (NumPy, for checkpoints); the
    bounds are the head's own, so a caller must say which."""
    z = KNEE_LOGIT_SCALE * np.asarray(logits, np.float64)
    log_min = math.log(min_e)
    return np.exp(log_min + (math.log(max_e) - log_min) / (1.0 + np.exp(-z)))


def stored_bounds(values) -> tuple[float, float]:
    """The ``(min, max)`` a head was built with, from its stored float32
    weight. Rounding to 7 significant digits undoes the float32 rounding
    (0.0099999998 → 0.01), so a reloaded head computes exactly what it trained with."""
    lo, hi = (float(f"{float(v):.7g}") for v in np.asarray(values).ravel())
    return lo, hi


def middle_knee(min_e: float, max_e: float) -> float:
    """The knee halfway (in log) between the bounds: a starting value for a
    head whose weights a restore or ``set_weights`` then overwrites."""
    return math.sqrt(min_e * max_e)


class LearnedOutputKnee(Layer):
    """``x_b = k_b · sinh(clip(y_b, ±SINH_CLIP))`` with one trainable knee
    ``k_b`` per band, bounded to ``(min_e, max_e)`` electrons."""

    def __init__(self, init_knee_e: float = DEFAULT_INIT_KNEE_E,
                 min_e: float = KNEE_MIN_E, max_e: float = KNEE_MAX_E, **kwargs):
        kwargs.setdefault("name", LAYER_NAME)
        super().__init__(**kwargs)
        self.init_knee_e = float(init_knee_e)
        self.min_e = float(min_e)
        self.max_e = float(max_e)
        self._init_logit = knee_logit_for(self.init_knee_e, self.min_e, self.max_e)
        # Logs in double precision, so a head's arithmetic is the same as when
        # the bounds were module constants (an old checkpoint's output is
        # bit-identical). The stored weight is the record loaders read.
        self._log_min = math.log(self.min_e)
        self._log_span = math.log(self.max_e) - self._log_min

    def build(self, input_shape):
        self.output_knee_logit = self.add_weight(
            name=KNEE_WEIGHT_NAME, shape=(int(input_shape[-1]),),
            initializer=Constant(self._init_logit), trainable=True)
        self.output_knee_bounds_e = self.add_weight(
            name=KNEE_BOUNDS_NAME, shape=(2,),
            initializer=Constant([self.min_e, self.max_e]), trainable=False)
        super().build(input_shape)

    def knees(self) -> tf.Tensor:
        """The current knee of every band, electrons, ``(bands,)``."""
        z = KNEE_LOGIT_SCALE * self.output_knee_logit
        return tf.exp(self._log_min + self._log_span * tf.sigmoid(z))

    def call(self, y):
        k = tf.cast(self.knees(), y.dtype)
        return k * tf.sinh(tf.clip_by_value(y, -SINH_CLIP, SINH_CLIP))

    def get_config(self):
        return {**super().get_config(), "init_knee_e": self.init_knee_e,
                "min_e": self.min_e, "max_e": self.max_e}


def learned_output_head(model) -> LearnedOutputKnee | None:
    """``model``'s learned output head, else ``None``."""
    for layer in getattr(model, "layers", ()):
        if isinstance(layer, LearnedOutputKnee):
            return layer
    return None


def learned_output_knees(model) -> np.ndarray | None:
    """The knees (e⁻) of ``model``'s learned output head, else ``None``."""
    head = learned_output_head(model)
    return None if head is None else head.knees().numpy()


def learned_knee_bounds(model) -> tuple[float, float] | None:
    """The ``(min, max)`` electrons of ``model``'s learned output head (its
    stored bounds, which a restore sets), else ``None``."""
    head = learned_output_head(model)
    return None if head is None else stored_bounds(head.output_knee_bounds_e.numpy())
