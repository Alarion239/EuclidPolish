from __future__ import annotations

import json
import os

import pytest
import tensorflow as tf
import tf_keras

from euclid_polish.experiments.lens_isolation.training import (
    LensIsolationTrainer,
    checkpoint_fingerprint,
    fork_member,
)


class FakeModel:
    def __init__(self, target, *, seed, init_weights_from):
        self.target = target
        self.seed = seed
        self.source = init_weights_from


def test_fork_is_virgin_records_provenance_and_never_mutates_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "checkpoint").write_text("weights")
    target = tmp_path / "experiment" / "member_00"
    before = checkpoint_fingerprint(str(source))
    model = fork_member(
        str(source),
        str(target),
        seed=7,
        dataset_fingerprint="dataset-sha",
        model_factory=FakeModel,
        protected_roots=(),
    )
    assert model.seed == 7
    assert checkpoint_fingerprint(str(source)) == before
    origin = json.loads((target / "origin.json").read_text())
    assert origin["source_fingerprint"] == before
    assert origin["dataset_fingerprint"] == "dataset-sha"
    assert origin["initial_step"] == 0


def test_fork_rejects_nonvirgin_target(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "checkpoint").write_text("weights")
    (target / "old").write_text("do not overwrite")
    with pytest.raises(ValueError, match="virgin"):
        fork_member(
            str(source),
            str(target),
            seed=1,
            dataset_fingerprint="x",
            model_factory=FakeModel,
            protected_roots=(),
        )


def test_dedicated_trainer_runs_from_step_zero_and_writes_both_tracks(tmp_path):
    tf_model = tf_keras.Sequential(
        [
            tf_keras.layers.Input((2, 2, 1)),
            tf_keras.layers.UpSampling2D(size=2),
            tf_keras.layers.Conv2D(1, 1),
        ]
    )
    wrapper = type("Wrapper", (), {"_tf_model": tf_model})()
    inputs = tf.ones((2, 2, 2, 1), tf.float32)
    targets = tf.concat([tf.ones((1, 4, 4, 1)), tf.zeros((1, 4, 4, 1))], axis=0)
    train = tf.data.Dataset.from_tensor_slices((inputs, targets)).repeat().batch(2)
    validation = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(2)
    trainer = LensIsolationTrainer(wrapper, str(tmp_path), steps=1)
    assert int(trainer.checkpoint.step) == 0
    trainer.train(train, validation, steps=1, evaluate_every=1)
    assert int(trainer.checkpoint.step) == 1
    assert (tmp_path / "checkpoint").exists()
    assert (tmp_path / "loss_best" / "checkpoint").exists()
