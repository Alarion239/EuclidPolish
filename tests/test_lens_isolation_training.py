from __future__ import annotations

import json
import os

import pytest

from euclid_polish.experiments.lens_isolation.training import (
    checkpoint_fingerprint,
    fork_member,
    train_member,
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


class RecordingModel:
    def __init__(self):
        self.calls = []

    def train(self, **kwargs):
        self.calls.append(kwargs)


class RecordingReporter:
    def __init__(self):
        self.steps = []
        self.metrics = []

    def set_step(self, *args):
        self.steps.append(args)

    def metric(self, value):
        self.metrics.append(value)


def test_training_dispatches_to_normal_model_train_fixed_record_mode(tmp_path):
    from euclid_polish.experiments.lens_isolation.config import TrainConfig

    model = RecordingModel()
    reporter = RecordingReporter()
    train_member(
        model,
        str(tmp_path),
        TrainConfig(sources=("member_00",), steps=12, batch_size=3, evaluate_every=4),
        reporter=reporter,
        member_index=1,
        member_count=2,
    )
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["lr_path"].endswith("dirty_train.tfrecord")
    assert call["hr_path"].endswith("lens_train.tfrecord")
    assert call["forward_onthefly"] is False
    assert "loss" not in call
    assert "crops_per_field" not in call
    call["step_callback"](2, 12)
    call["eval_callback"]({"step": 4, "loss": 0.2})
    assert reporter.steps[-1] == (14, 24, "member 2 step 2")
    assert reporter.metrics[-1]["member"] == 2
