"""Training losses: the Lp reconstruction-norm family (L1 / L2 / L3).

``lp_loss(p)`` returns ``(mean |a − b|^p)^(1/p)`` — the p-NORM, not the raw
p-th-power mean. The root keeps every p on the same scale/units as the L1
default (asinh residuals), so the LR schedule, gradient-spike threshold and
plateau guard need no per-p retuning, and members trained with different p
log comparable loss curves. p=1 is exactly MAE; p=2 is RMSE; p=3 weights the
worst residuals hardest (a sharper penalty on rare large errors — bright
structure and hallucinated features).
"""

from __future__ import annotations

import tensorflow as tf

#: Loss-norm knob values → exponent.
LOSS_NORMS = {"l1": 1, "l2": 2, "l3": 3}


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
