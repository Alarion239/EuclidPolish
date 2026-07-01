"""Reduce-LR-on-plateau guard for WDSR SR training.

The gradient-spike guard (see :mod:`~euclid_polish.training.trainer`) halves the
LR when the *pre-clip gradient norm* explodes. This is the symmetric other half:
halve the LR when the validation metric **stalls** — no progress for
``patience`` steps. Together they mean the LR drops on either failure mode
(divergence OR stagnation) instead of only on a fixed schedule milestone.

It is the standard ``ReduceLROnPlateau`` policy (Keras/PyTorch), phrased in
*steps* rather than epochs because this trainer is step-based and evaluates
every ``evaluate_every`` steps. Pure Python (no TF) so the decision logic is
unit-testable; the trainer owns the actual LR-scale bookkeeping and the absolute
``min_lr`` floor.
"""

from __future__ import annotations


class PlateauLRReducer:
    """Decide when a stalled validation metric warrants an LR cut.

    Parameters
    ----------
    mode : {"min", "max"}
        Whether lower (e.g. ``combined_loss``) or higher (e.g.
        ``psnr_stretched``) is better.
    patience : int
        Steps of no improvement before a cut fires.
    min_delta : float
        Minimum change (absolute, in metric units) that counts as an
        improvement. Micro-creep smaller than this does NOT reset patience — the
        whole point, since the skip-only plateau's loss drifts down by ~1e-5 per
        eval while genuinely flat.
    cooldown : int
        Steps to wait after a cut before the stall counter re-arms, so one
        stall doesn't fire repeatedly on the next few evals.

    Usage
    -----
    Call :meth:`should_reduce` once per validation eval with the current
    ``(step, metric)``. It returns ``True`` exactly on the evals where the LR
    should be cut; the caller applies the factor and re-asserts the LR.
    """

    def __init__(
        self,
        *,
        mode: str = "min",
        patience: int = 5000,
        min_delta: float = 1e-4,
        cooldown: int = 2000,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = abs(float(min_delta))
        self.cooldown = int(cooldown)
        self._best: float | None = None
        self._best_step = 0
        self._last_reduce_step: int | None = None

    def _is_improvement(self, metric: float) -> bool:
        if self._best is None:
            return True
        if self.mode == "min":
            return metric < self._best - self.min_delta
        return metric > self._best + self.min_delta

    def should_reduce(self, step: int, metric: float) -> bool:
        """Feed one eval; return ``True`` iff the LR should be cut now.

        Non-finite metrics (e.g. ``+inf`` combined loss when no lane is active)
        are ignored so they neither reset nor trip the guard.
        """
        step = int(step)
        if metric != metric or metric in (float("inf"), float("-inf")):  # NaN/inf
            return False
        if self._is_improvement(metric):
            self._best = float(metric)
            self._best_step = step
            return False
        # In a cooldown window after a recent cut: hold off and keep the
        # post-cut step as the new "no-progress-since" anchor.
        if (self._last_reduce_step is not None
                and step - self._last_reduce_step < self.cooldown):
            self._best_step = max(self._best_step, self._last_reduce_step)
            return False
        if step - self._best_step >= self.patience:
            self._last_reduce_step = step
            self._best_step = step          # re-arm: wait another `patience`
            return True
        return False

    def reset(self, step: int = 0) -> None:
        """Forget the stall history (e.g. after a gradient-spike rollback
        rewinds the step counter, so ``step - best_step`` stays meaningful)."""
        self._best = None
        self._best_step = int(step)
        self._last_reduce_step = None
