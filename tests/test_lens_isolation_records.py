from __future__ import annotations

import csv
import os

import numpy as np
import pytest

from euclid_polish.experiments.lens_isolation.generation import GeneratedExample
from euclid_polish.experiments.lens_isolation.records import (
    dataset_fingerprint,
    generate_split,
    validate_split,
)
from euclid_polish.image import Image
from euclid_polish.image.tfio import read_images, tfrecord_path


def _image(value):
    return Image(
        data=np.full((4, 4, 4), value, np.float32),
        pixel_scale_arcsec=0.05,
        band_names=("VIS", "Y_E", "J_E", "H_E"),
        is_clean=True,
    )


class TinyGenerator:
    def __init__(self, *, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def generate_example(self, _rng, *, label, fixed_dirty):
        if self.fail_at is not None and self.calls == self.fail_at:
            raise RuntimeError("generation exploded")
        self.calls += 1
        lens = _image(5 if label else 0)
        return GeneratedExample(
            scene=_image(10 + label),
            lens=lens,
            dirty=_image(100 + label) if fixed_dirty else None,
            row={"schema_version": 1, "label": label, "theta_E_arcsec": 1.0 if label else ""},
        )


def _rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def test_train_split_is_balanced_aligned_and_has_no_dirty_record(tmp_path):
    out = str(tmp_path / "records")
    summary = generate_split(TinyGenerator(), out, "train", count=6, seed=42)

    rows = _rows(os.path.join(out, "manifest_train.csv"))
    assert [int(row["index"]) for row in rows] == list(range(6))
    assert sum(int(row["label"]) for row in rows) == 3
    assert summary.n_positive == summary.n_negative == 3
    assert len(read_images(tfrecord_path(out, "scene_train"), 10)) == 6
    assert len(read_images(tfrecord_path(out, "lens_train"), 10)) == 6
    assert not os.path.exists(tfrecord_path(out, "dirty_train"))
    assert validate_split(out, "train", 6)


@pytest.mark.parametrize("subset", ["validate", "test"])
def test_fixed_splits_include_dirty_records(tmp_path, subset):
    out = str(tmp_path / subset)
    generate_split(TinyGenerator(), out, subset, count=4, seed=9)
    assert len(read_images(tfrecord_path(out, f"dirty_{subset}"), 10)) == 4
    assert validate_split(out, subset, 4)


def test_same_seed_replays_label_and_example_seed_order(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    generate_split(TinyGenerator(), a, "train", count=8, seed=123)
    generate_split(TinyGenerator(), b, "train", count=8, seed=123)
    ar, br = _rows(os.path.join(a, "manifest_train.csv")), _rows(os.path.join(b, "manifest_train.csv"))
    assert [(r["label"], r["example_seed"]) for r in ar] == [(r["label"], r["example_seed"]) for r in br]


def test_parallel_generation_preserves_deterministic_manifest_order(tmp_path):
    sequential, parallel = str(tmp_path / "sequential"), str(tmp_path / "parallel")
    generate_split(TinyGenerator(), sequential, "train", count=8, seed=123)
    generate_split(TinyGenerator(), parallel, "train", count=8, seed=123, workers=3)
    a = _rows(os.path.join(sequential, "manifest_train.csv"))
    b = _rows(os.path.join(parallel, "manifest_train.csv"))
    assert [(row["index"], row["label"], row["example_seed"]) for row in a] == [
        (row["index"], row["label"], row["example_seed"]) for row in b
    ]


def test_failed_generation_never_publishes_partial_split(tmp_path):
    out = str(tmp_path / "records")
    with pytest.raises(RuntimeError, match="exploded"):
        generate_split(TinyGenerator(fail_at=1), out, "train", count=4, seed=1)
    assert not os.path.exists(tfrecord_path(out, "scene_train"))
    assert not os.path.exists(tfrecord_path(out, "lens_train"))
    assert not os.path.exists(os.path.join(out, "manifest_train.csv"))


def test_existing_complete_split_requires_force_to_replace(tmp_path):
    out = str(tmp_path / "records")
    first = TinyGenerator()
    generate_split(first, out, "train", count=4, seed=1)
    second = TinyGenerator()
    summary = generate_split(second, out, "train", count=4, seed=2)
    assert summary.reused is True
    assert second.calls == 0
    generate_split(second, out, "train", count=4, seed=2, force=True)
    assert second.calls == 4


def test_dataset_fingerprint_changes_when_manifest_changes(tmp_path):
    out = str(tmp_path / "records")
    generate_split(TinyGenerator(), out, "train", count=4, seed=1)
    before = dataset_fingerprint(out)
    with open(os.path.join(out, "manifest_train.csv"), "a") as handle:
        handle.write("changed\n")
    assert dataset_fingerprint(out) != before
