"""can_reuse_eval_object: membership fingerprint gates ensemble-product reuse."""
from __future__ import annotations

import json
import os

from euclid_polish.eval.catalog_runner import can_reuse_eval_object


def _touch(d, *names):
    os.makedirs(d, exist_ok=True)
    for n in names:
        with open(os.path.join(d, n), "wb") as f:
            f.write(b"x")


def test_reuse_requires_matching_member_labels(tmp_path):
    d = str(tmp_path / "obj")
    _touch(d, "original_stack.fits", "SR.fits", "std.fits", "pca0.fits")
    labels = ["00·psnr", "01·psnr"]
    # no members.json yet → pre-fingerprint outputs are stale when labels demanded
    assert not can_reuse_eval_object(d, require_disagreement=True,
                                     member_labels=labels)
    with open(os.path.join(d, "members.json"), "w") as f:
        json.dump({"member_labels": labels}, f)
    assert can_reuse_eval_object(d, require_disagreement=True,
                                 member_labels=labels)
    assert not can_reuse_eval_object(d, require_disagreement=True,
                                     member_labels=["00·psnr"])


def test_single_member_reuse_unchanged(tmp_path):
    d = str(tmp_path / "obj")
    _touch(d, "original_stack.fits", "SR.fits")
    assert can_reuse_eval_object(d, require_disagreement=False)
    # disagreement demanded but cubes absent → not reusable
    assert not can_reuse_eval_object(d, require_disagreement=True)
