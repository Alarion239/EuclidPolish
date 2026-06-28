"""Which generated subset evaluations read.

Training fits on ``train`` and selects checkpoints (save-best) on ``validate``.
Reporting final metrics on the same ``validate`` set the model was selected on
is optimistic, so the pipeline also generates a held-out ``test`` set and the
eval code reads *that* instead.

:func:`eval_subset` is the single source of truth: it returns ``"test"`` when a
test split is on disk, and falls back to ``"validate"`` for datasets generated
before the test split existed (so old data keeps working unchanged).
"""

from __future__ import annotations

import os

from euclid_polish.image.tfio import tfrecord_path

#: The held-out subset evaluations should use (never seen during training or
#: model selection).
EVAL_SUBSET = "test"
#: What to fall back to when no test split is present (pre-test-split datasets).
EVAL_SUBSET_FALLBACK = "validate"


def eval_subset(records_dir: str) -> str:
    """``"test"`` when ``dirty_test.tfrecord`` exists in ``records_dir``, else
    ``"validate"`` (back-compat for datasets generated before the test split)."""
    if os.path.exists(tfrecord_path(records_dir, f"dirty_{EVAL_SUBSET}")):
        return EVAL_SUBSET
    return EVAL_SUBSET_FALLBACK
