"""eval_subset: evals read the held-out test split, falling back to validate."""

from __future__ import annotations

from euclid_polish.eval.subsets import (
    EVAL_SUBSET,
    EVAL_SUBSET_FALLBACK,
    eval_subset,
)


def test_falls_back_to_validate_without_test(tmp_path):
    # Pre-test-split datasets (no dirty_test.tfrecord) keep using validate.
    assert EVAL_SUBSET_FALLBACK == "validate"
    assert eval_subset(str(tmp_path)) == "validate"


def test_prefers_test_when_present(tmp_path):
    assert EVAL_SUBSET == "test"
    (tmp_path / "dirty_test.tfrecord").write_bytes(b"")   # presence marker
    assert eval_subset(str(tmp_path)) == "test"
