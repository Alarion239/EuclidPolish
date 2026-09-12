"""Rollbacks rewind the model but never loosen the loss-track bar."""

from __future__ import annotations

import pytest
import tensorflow as tf

from euclid_polish.training.trainer import restore_keeping_loss_bar


def test_rollback_restore_keeps_lower_loss_bar(tmp_path):
    checkpoint = tf.train.Checkpoint(
        step=tf.Variable(10),
        psnr=tf.Variable(40.0),
        best_loss=tf.Variable(0.5),
    )
    saved = checkpoint.save(str(tmp_path / "ckpt"))
    checkpoint.step.assign(99)
    checkpoint.psnr.assign(38.0)
    checkpoint.best_loss.assign(0.1)

    restore_keeping_loss_bar(checkpoint, saved)

    assert int(checkpoint.step.numpy()) == 10
    assert float(checkpoint.psnr.numpy()) == 40.0
    assert float(checkpoint.best_loss.numpy()) == pytest.approx(0.1)
