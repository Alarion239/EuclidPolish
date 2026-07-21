"""Training losses: the Lp reconstruction-norm family, MSE, and reverse-Huber.

``lp_loss(p)`` returns ``(mean |a − b|^p)^(1/p)`` — the p-NORM, not the raw
p-th-power mean. The root keeps every p on the same scale/units as the L1
default (asinh residuals), so the LR schedule, gradient-spike threshold and
plateau guard need no per-p retuning, and members trained with different p
log comparable loss curves. p=1 is exactly MAE; p=2 is RMSE; p=3 weights the
worst residuals hardest (a sharper penalty on rare large errors — bright
structure and hallucinated features).

``mse_loss`` returns ``mean((a − b)²)`` without the outer square root. Unlike
RMSE, it is a linear mean over pixels and examples, so regrouping the same
residual pixels into full fields or cutouts does not change the scalar
objective when every pixel receives the same weight.

``berhu_loss`` is the reverse Huber (BerHu): L1 on residuals below a threshold
δ, L2 above it — the mirror image of Huber. That is exactly the estimator the
L1-galaxy / L2-star split calls for: robust (median-like) in the extended,
noise-dominated flux where L2 rings, but quadratic on the bright point-source
peaks where L1 leaves stars soft. δ is set adaptively per batch to
``c · max|residual|`` (c=0.2, the Laina+16 / Zwald+12 default), so the loss is
scale-adaptive, parameter-free and sits on the same L1 magnitude scale — no
LR/threshold retuning, like the p-norms.

``build_loss(name)`` is the single dispatcher over all of the above; it is what
the trainer wires up from a member's ``loss_norm`` knob.
"""

from __future__ import annotations

import tensorflow as tf

from euclid_polish.training.loss_names import (  # noqa: F401  (re-exported)
    BERHU_DEFAULT_C,
    LOSS_NAMES,
    LOSS_NORMS,
    MSE_NAME,
)


def lp_loss(norm: str = "l1"):
    """The loss callable for a ``LOSS_NORMS`` key; signature ``loss(a, b)``."""
    p = LOSS_NORMS.get(str(norm).lower())
    if p is None:
        raise ValueError(f"unknown loss norm {norm!r}; use one of "
                         f"{sorted(LOSS_NORMS)}")
    if p == 1:
        def _l1(a, b):
            return tf.reduce_mean(tf.abs(a - b))
        return _l1

    def _lp(a, b):
        return tf.reduce_mean(tf.abs(a - b) ** p) ** (1.0 / p)
    return _lp


def mse_loss():
    """Mean squared error callable; signature ``loss(a, b)``."""
    def _mse(a, b):
        return tf.reduce_mean(tf.square(a - b))
    return _mse


def berhu_loss(c: float = BERHU_DEFAULT_C):
    """Reverse-Huber (BerHu) loss; signature ``loss(a, b)``.

    ``B(x) = |x|`` for ``|x| ≤ δ`` and ``(x² + δ²) / 2δ`` for ``|x| > δ``,
    averaged over the batch. δ is a per-batch constant ``c · max|residual|``
    (``stop_gradient`` — δ is a threshold, not a learned quantity), so the
    L1↔L2 crossover tracks the batch's own dynamic range. The two branches
    meet at ``|x| = δ`` in both value and slope, so B is C¹.
    """
    c = float(c)

    def _berhu(a, b):
        d = tf.abs(a - b)
        # δ from the batch max; floored so an all-zero (perfect) batch, where
        # d ≤ δ holds trivially, takes the L1 branch instead of dividing by 0.
        delta = tf.maximum(c * tf.stop_gradient(tf.reduce_max(d)), 1e-8)
        quad = (d * d + delta * delta) / (2.0 * delta)
        return tf.reduce_mean(tf.where(d <= delta, d, quad))
    return _berhu


def build_loss(name: str = "l1"):
    """Reconstruction loss for a ``loss_norm`` knob value."""
    key = str(name).lower()
    if key == "berhu":
        return berhu_loss()
    if key == MSE_NAME:
        return mse_loss()
    if key in LOSS_NORMS:
        return lp_loss(key)
    raise ValueError(f"unknown loss {name!r}; use one of {sorted(LOSS_NAMES)}")
