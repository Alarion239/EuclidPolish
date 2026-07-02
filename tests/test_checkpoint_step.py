"""checkpoint_step: read the persisted trainer step without a model build."""
from __future__ import annotations

import tensorflow as tf

from euclid_polish.training.inference import checkpoint_step


def test_checkpoint_step_reads_step_var(tmp_path):
    d = str(tmp_path / "ck")
    ck = tf.train.Checkpoint(step=tf.Variable(1234))
    mgr = tf.train.CheckpointManager(ck, d, max_to_keep=1)
    mgr.save()
    assert checkpoint_step(d) == 1234


def test_checkpoint_step_none_when_missing(tmp_path):
    assert checkpoint_step(str(tmp_path / "nope")) is None
