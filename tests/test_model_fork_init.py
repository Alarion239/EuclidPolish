"""Model(init_weights_from=…): fork a member's weights at step 0."""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish.model import Model


def _save_weights_ckpt(model: Model, d: str) -> None:
    # Same key layout the Trainer saves under (model=...), so the fork's
    # weights-only restore matches a real member checkpoint.
    ck = tf.train.Checkpoint(step=tf.Variable(500), model=model._tf_model)
    tf.train.CheckpointManager(ck, d, max_to_keep=1).save()


def test_fork_copies_weights_and_is_virgin(tmp_path):
    src_dir = str(tmp_path / "src")
    src = Model(src_dir, num_res_blocks=1, seed=1)
    _save_weights_ckpt(src, src_dir)

    dst = Model(str(tmp_path / "dst"), num_res_blocks=1, seed=2,
                init_weights_from=src_dir)
    for a, b in zip(src._tf_model.get_weights(),
                    dst._tf_model.get_weights(), strict=True):
        assert np.array_equal(a, b)
    # the fork target itself has NO checkpoint → training starts at step 0
    assert tf.train.latest_checkpoint(str(tmp_path / "dst")) is None


def test_fork_refuses_nonvirgin_target(tmp_path):
    src_dir = str(tmp_path / "src")
    src = Model(src_dir, num_res_blocks=1, seed=1)
    _save_weights_ckpt(src, src_dir)
    dst_dir = str(tmp_path / "dst")
    dst = Model(dst_dir, num_res_blocks=1, seed=2)
    _save_weights_ckpt(dst, dst_dir)          # target already trained
    with pytest.raises(ValueError, match="virgin"):
        Model(dst_dir, num_res_blocks=1, init_weights_from=src_dir)


def test_fork_refuses_missing_source(tmp_path):
    with pytest.raises(ValueError, match="no checkpoint"):
        Model(str(tmp_path / "dst"), num_res_blocks=1,
              init_weights_from=str(tmp_path / "empty_src"))
