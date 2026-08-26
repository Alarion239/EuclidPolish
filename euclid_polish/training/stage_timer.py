"""Backward-compatible import for the pipeline stage timer.

The implementation belongs to :mod:`euclid_polish.observability.stage_timer`
because generation-only jobs must not initialize the training stack merely to
record stage timings.
"""

from euclid_polish.observability.stage_timer import StageTimer

__all__ = ["StageTimer"]
