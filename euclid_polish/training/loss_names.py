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

#: Every SELECTABLE ``loss_norm`` knob value. Shared by the CLI / web validators
#: so a new loss is allowed in exactly one place.
#:
#: ``berhu`` is DEPRECATED (2026-07-08) and deliberately absent: the experiment
#: failed — BerHu members did not resolve stars, had worse PSNR and power
#: spectra than L2, and hit degenerate plateaus. It stays dispatchable in
#: :func:`~euclid_polish.training.losses.build_loss` (so the few existing
#: members still load / continue / display) but can no longer be SELECTED for a
#: new member. Do not re-add it without a new, better result.
LOSS_NAMES = tuple(LOSS_NORMS)

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
