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
