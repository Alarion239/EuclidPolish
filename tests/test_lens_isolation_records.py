from __future__ import annotations

import json
import os
from hashlib import sha256

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.experiments.lens_isolation.generation import GeneratedExample
from euclid_polish.experiments.lens_isolation.records import (
    dataset_fingerprint,
    generate_split,
    make_shards,
    merge_shards,
    validate_split,
    write_dataset_metadata,
    write_shard,
)
from euclid_polish.image import Image
from euclid_polish.image.tfio import parse_example, tfrecord_path


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

    def generate_example(self, _rng):
        if self.fail_at is not None and self.calls == self.fail_at:
            raise RuntimeError("generation exploded")
        self.calls += 1
        return GeneratedExample(
            dirty=_image(100 + self.calls),
            lens=_image(5 if self.calls % 2 else 0),
            sources={
                "galaxies": [{"type": "galaxy", "render": "tng", "x_pix": 1.0, "y_pix": 1.0}],
                "lenses": [],
                "stars": [],
                "n_galaxies": 1,
                "n_lenses": 0,
                "n_stars": 0,
            },
        )


def _count(path):
    import tensorflow as tf

    return sum(1 for _ in tf.data.TFRecordDataset(path))


def _record_indices(path):
    import tensorflow as tf

    indices = []
    for raw in tf.data.TFRecordDataset(path):
        example = tf.train.Example.FromString(raw.numpy())
        indices.append(example.features.feature["index"].int64_list.value[0])
    return indices


def _source_field_indices(path):
    import csv

    with open(path, newline="", encoding="utf-8") as handle:
        return [int(row["field_index"]) for row in csv.DictReader(handle)]


@pytest.mark.parametrize("subset", ["train", "validate", "test"])
def test_every_split_writes_aligned_dirty_lens_records_and_sources(tmp_path, subset):
    out = str(tmp_path / "records")
    summary = generate_split(TinyGenerator(), out, subset, count=3, seed=42, config_fingerprint="cfg")

    dirty = tfrecord_path(out, f"dirty_{subset}")
    lens = tfrecord_path(out, f"lens_{subset}")
    sources = os.path.join(out, f"sources_{subset}.csv")
    assert summary.count == 3
    assert _count(dirty) == _count(lens) == 3
    assert os.path.isfile(sources)
    assert validate_split(out, subset, 3, config_fingerprint="cfg")


def test_paired_records_use_the_normal_tensorflow_example_parser(tmp_path):
    out = str(tmp_path / "records")
    generate_split(TinyGenerator(), out, "train", count=1, seed=7, config_fingerprint="cfg")
    import tensorflow as tf

    dirty_raw = next(iter(tf.data.TFRecordDataset(tfrecord_path(out, "dirty_train"))))
    lens_raw = next(iter(tf.data.TFRecordDataset(tfrecord_path(out, "lens_train"))))
    dirty = parse_example(dirty_raw, Config.NUM_LR_CHANNELS)
    lens = parse_example(lens_raw, Config.NUM_HR_CHANNELS)
    assert tuple(dirty.shape) == (4, 4, Config.NUM_LR_CHANNELS)
    assert tuple(lens.shape) == (4, 4, Config.NUM_HR_CHANNELS)


def test_process_shards_merge_pairs_and_sources_in_global_index_order(tmp_path):
    records = str(tmp_path / "records")
    shards = make_shards("train", count=5, workers=2, seed=17)

    summaries = [
        write_shard(TinyGenerator(), records, shard, config_fingerprint="cfg")
        for shard in reversed(shards)
    ]

    summary = merge_shards(records, "train", reversed(summaries), config_fingerprint="cfg")

    assert summary.count == 5
    assert _record_indices(tfrecord_path(records, "dirty_train")) == [0, 1, 2, 3, 4]
    assert _record_indices(tfrecord_path(records, "lens_train")) == [0, 1, 2, 3, 4]
    assert _source_field_indices(tmp_path / "records" / "sources_train.csv") == [0, 1, 2, 3, 4]
    assert validate_split(records, "train", 5, config_fingerprint="cfg")


def test_generation_resumes_completed_paired_shards_after_interruption(tmp_path):
    records = str(tmp_path / "records")
    generator = TinyGenerator()
    shards = make_shards("train", count=4, workers=2, seed=17)
    write_shard(generator, records, shards[1], config_fingerprint="cfg")

    summary = generate_split(
        generator,
        records,
        "train",
        count=4,
        seed=17,
        config_fingerprint="cfg",
        workers=1,
    )

    assert summary.count == 4
    assert generator.calls == 4
    assert validate_split(records, "train", 4, config_fingerprint="cfg")


def test_split_reuse_requires_matching_schema_and_configuration_fingerprint(tmp_path):
    out = str(tmp_path / "records")
    first = TinyGenerator()
    generate_split(first, out, "train", count=2, seed=1, config_fingerprint="old")
    assert generate_split(TinyGenerator(), out, "train", count=2, seed=1, config_fingerprint="old").reused
    assert not validate_split(out, "train", 2, config_fingerprint="new")
    replacement = TinyGenerator()
    generate_split(replacement, out, "train", count=2, seed=2, config_fingerprint="new")
    assert replacement.calls == 2


@pytest.mark.parametrize(
    ("data", "suffix"),
    [
        (np.zeros((3, 4, 4), np.float32), "shape"),
        (np.zeros((4, 4, 3), np.float32), "channels"),
    ],
)
def test_split_validation_rejects_tampered_tensor_shape_or_channel_count(tmp_path, data, suffix):
    out = str(tmp_path / "records")
    generate_split(TinyGenerator(), out, "train", count=2, seed=1, config_fingerprint="cfg")
    dirty_path = tfrecord_path(out, "dirty_train")
    tampered = Image(
        data=data,
        pixel_scale_arcsec=0.05,
        band_names=("VIS", "Y_E", "J_E", "H_E")[: data.shape[-1]],
        is_clean=False,
    )
    import tensorflow as tf

    with tf.io.TFRecordWriter(dirty_path) as writer:
        writer.write(tampered.to_tfrecord(index=0))
        writer.write(tampered.to_tfrecord(index=1))
    split_path = os.path.join(out, "split_train.json")
    metadata = json.loads(open(split_path, encoding="utf-8").read())
    metadata["fingerprints"]["dirty"] = sha256(open(dirty_path, "rb").read()).hexdigest()
    with open(split_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle)

    assert not validate_split(out, "train", 2, config_fingerprint="cfg"), suffix


def test_failed_generation_never_publishes_partial_split(tmp_path):
    out = str(tmp_path / "records")
    with pytest.raises(RuntimeError, match="exploded"):
        generate_split(TinyGenerator(fail_at=1), out, "train", count=4, seed=1, config_fingerprint="cfg")
    assert not os.path.exists(tfrecord_path(out, "dirty_train"))
    assert not os.path.exists(tfrecord_path(out, "lens_train"))
    assert not os.path.exists(os.path.join(out, "sources_train.csv"))


def test_dataset_metadata_persists_schema_config_and_record_fingerprint(tmp_path):
    out = str(tmp_path / "records")
    summaries = {
        "train": generate_split(TinyGenerator(), out, "train", count=2, seed=1, config_fingerprint="cfg")
    }
    path = write_dataset_metadata(
        out,
        config={"schema_version": 2, "lens_density_arcmin2": 20.0},
        master_seed=9,
        split_summaries=summaries,
        source_commit="abc123",
    )
    metadata = json.loads(open(path, encoding="utf-8").read())
    assert metadata["schema_version"] == 2
    assert metadata["config"]["lens_density_arcmin2"] == 20.0
    assert metadata["fingerprint"] == dataset_fingerprint(out)
