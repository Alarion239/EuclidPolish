"""Reconstruction-loss knob names — the TF-free constants.

Kept separate from :mod:`euclid_polish.training.losses` (which imports
TensorFlow) so lightweight consumers — the CLI/web submit path that only need
to validate a ``loss`` string — can import the vocabulary without paying for
TF. Add a new loss here once and every validator picks it up.
"""

from __future__ import annotations

#: Loss-norm knob values → exponent for the rooted p-norm family.
LOSS_NORMS = {"l1": 1, "l2": 2, "l3": 3}

#: The fraction of the batch's max residual used as the BerHu L1↔L2 threshold.
BERHU_DEFAULT_C = 0.2

#: Every valid ``loss_norm`` knob value (p-norms + reverse-Huber). Shared by
#: the CLI / web validators so a new loss is allowed in exactly one place.
LOSS_NAMES = (*LOSS_NORMS, "berhu")

#: Losses whose median-like optimum admits the DEGENERATE skip-only basin — the
#: flat ~43.5 dB PSNR floor where the trunk collapses to 0 and only the bilinear
#: skip survives. L1's median doesn't move for rare point sources, so erasing
#: them is a low-loss solution and the basin is a real local optimum. L2/L3
#: weight large residuals quadratically-or-worse, and BerHu's L2 branch
#: penalises exactly those erased-star residuals, so for them the skip-only
#: solution is high-loss, NOT a basin. The reduce-LR / rollback plateau guard
#: exists only to escape this basin — see :func:`plateau_guard_applies`.
DEGENERATE_PLATEAU_LOSSES = frozenset({"l1"})


def plateau_guard_applies(loss_norm) -> bool:
    """Whether the plateau LR guard is meaningful for this loss.

    True only for losses in :data:`DEGENERATE_PLATEAU_LOSSES` (L1). For the
    large-residual-weighted losses the guard never helps — the degenerate basin
    isn't a low-loss optimum — and its stall detector misfires on their genuine
    slow climbs, cutting the LR mid-improvement. So it is switched off for them.
    """
    return str(loss_norm).lower() in DEGENERATE_PLATEAU_LOSSES
