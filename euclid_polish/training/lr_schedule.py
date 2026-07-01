"""Warmup + cosine learning-rate schedule for WDSR SR training.

Replaces the old ``PiecewiseConstantDecay`` (5e-4 flat → 1e-4 @ 50% → 2e-5 @
80%). The flat 5e-4 head was the cause of the long ``~43.5 dB`` "skip-only"
plateau: too hot to settle into the sharp deconvolution minimum, so members sat
on the degenerate floor until the fixed 50%-of-steps step-down — and members
that never gradient-spiked (so never got an early guard halving) waited the
whole way. A schedule that decays smoothly *from the start* removes the hot flat
region entirely, so every member leaves the floor early and uniformly, without
depending on a fixed milestone or on random spikes.

The value is pure Python (no TF) so it is unit-testable and can be sampled by
:class:`~euclid_polish.training.trainer.Trainer` each step via its manual
``_apply_lr`` path — the same path the gradient-spike guard uses to re-assert a
(halved) LR. ``WarmupCosineDecay`` is a thin callable wrapper so it drops into
the ``learning_rate=`` slot that previously took a Keras schedule.
"""

from __future__ import annotations

import math


def warmup_cosine_lr(
    step: int,
    *,
    peak_lr: float,
    final_lr: float,
    warmup_steps: int,
    total_steps: int,
    start_lr: float,
) -> float:
    """LR at ``step`` for a linear-warmup → cosine-decay schedule.

    * ``[0, warmup_steps)``      — linear ramp ``start_lr → peak_lr``.
    * ``[warmup_steps, total]``  — cosine decay ``peak_lr → final_lr``.
    * ``step >= total_steps``    — clamped at ``final_lr``.

    All rates are absolute (not multipliers). ``warmup_steps=0`` skips the ramp
    and starts the cosine at ``peak_lr``.
    """
    step = max(0, int(step))
    w = max(0, int(warmup_steps))
    total = max(w + 1, int(total_steps))
    if w > 0 and step < w:
        return start_lr + (peak_lr - start_lr) * (step / w)
    decay_steps = total - w
    t = min(1.0, (step - w) / decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * t))       # 1 → 0 over the decay
    return final_lr + (peak_lr - final_lr) * cosine


class WarmupCosineDecay:
    """Callable warmup→cosine LR schedule (see :func:`warmup_cosine_lr`).

    Accepts either a Python ``int`` step or an eager TF scalar (``int(step)``
    coerces both), so it slots into ``Trainer``'s manual per-step LR sampling
    exactly where a ``tf.keras`` ``LearningRateSchedule`` used to.
    """

    def __init__(
        self,
        *,
        peak_lr: float,
        final_lr: float,
        warmup_steps: int,
        total_steps: int,
        start_lr: float | None = None,
    ) -> None:
        self.peak_lr = float(peak_lr)
        self.final_lr = float(final_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        # Warmup starts from a small fraction of the peak rather than exactly 0,
        # so the very first steps still move (a 0 start wastes them).
        self.start_lr = (float(start_lr) if start_lr is not None
                         else 0.05 * self.peak_lr)

    def __call__(self, step) -> float:
        return warmup_cosine_lr(
            int(step),
            peak_lr=self.peak_lr, final_lr=self.final_lr,
            warmup_steps=self.warmup_steps, total_steps=self.total_steps,
            start_lr=self.start_lr,
        )

    def get_config(self) -> dict:
        return {
            "peak_lr": self.peak_lr, "final_lr": self.final_lr,
            "warmup_steps": self.warmup_steps, "total_steps": self.total_steps,
            "start_lr": self.start_lr,
        }
