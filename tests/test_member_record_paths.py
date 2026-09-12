"""Each member reads the target records of its own star regime."""

from __future__ import annotations

import pytest

from euclid_polish.model import member_record_paths


def test_starless_members_use_clean_targets():
    assert member_record_paths(
        "/r/dirty_train.tfrecord", "/r/clean_train.tfrecord",
        starless=True, forward_onthefly=False,
    ) == (
        "/r/clean_train.tfrecord",
        "/r/dirty_validate.tfrecord",
        "/r/clean_validate.tfrecord",
    )


def test_starfull_record_members_train_and_validate_on_hr():
    assert member_record_paths(
        "/r/dirty_train.tfrecord", "/r/clean_train.tfrecord",
        starless=False, forward_onthefly=False,
    ) == (
        "/r/hr_train.tfrecord",
        "/r/dirty_validate.tfrecord",
        "/r/hr_validate.tfrecord",
    )


def test_starfull_onthefly_members_keep_scene_but_validate_on_hr():
    assert member_record_paths(
        "/r/dirty_train.tfrecord", "/r/clean_train.tfrecord",
        starless=False, forward_onthefly=True,
    ) == (
        "/r/clean_train.tfrecord",
        "/r/dirty_validate.tfrecord",
        "/r/hr_validate.tfrecord",
    )


def test_validation_path_requires_train_split_name():
    with pytest.raises(ValueError, match="_train"):
        member_record_paths(
            "/r/dirty.tfrecord", "/r/clean.tfrecord",
            starless=True, forward_onthefly=False,
        )
