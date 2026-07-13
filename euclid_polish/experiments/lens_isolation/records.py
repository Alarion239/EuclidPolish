"""Atomic paired TFRecord persistence for the lens-isolation experiment."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf

from euclid_polish.image.tfio import tfrecord_path

_SUBSETS = {"train", "validate", "test"}


@dataclass(frozen=True)
class SplitSummary:
    subset: str
    count: int
    n_positive: int
    n_negative: int
    reused: bool = False


def _record_count(path: str) -> int:
    if not os.path.isfile(path):
        return -1
    return sum(1 for _ in tf.data.TFRecordDataset([path]))


def validate_split(records_dir: str, subset: str, expected_count: int) -> bool:
    """Return whether all paired files exist and contain aligned row counts."""
    if subset not in _SUBSETS:
        raise ValueError(f"unknown subset {subset!r}")
    paths = [
        tfrecord_path(records_dir, f"scene_{subset}"),
        tfrecord_path(records_dir, f"lens_{subset}"),
    ]
    if subset != "train":
        paths.append(tfrecord_path(records_dir, f"dirty_{subset}"))
    manifest = os.path.join(records_dir, f"manifest_{subset}.csv")
    if not os.path.isfile(manifest):
        return False
    try:
        with open(manifest, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return len(rows) == expected_count and all(_record_count(path) == expected_count for path in paths)
    except (OSError, tf.errors.OpError, UnicodeError, csv.Error):
        return False


def _temp_path(final_path: str, token: str) -> str:
    return f"{final_path}.tmp-{token}"


def generate_split(
    generator,
    records_dir: str,
    subset: str,
    *,
    count: int,
    seed: int,
    force: bool = False,
    workers: int = 1,
) -> SplitSummary:
    """Generate one balanced split and publish it as one atomic file set.

    Files are first completed under unique temporary names. Existing complete
    splits remain readable until every replacement is ready.
    """
    if subset not in _SUBSETS:
        raise ValueError(f"unknown subset {subset!r}")
    if count < 0 or count % 2:
        raise ValueError("balanced split count must be a non-negative even integer")
    if int(workers) < 1:
        raise ValueError("workers must be >= 1")
    if not force and validate_split(records_dir, subset, count):
        return SplitSummary(subset, count, count // 2, count // 2, reused=True)

    os.makedirs(records_dir, exist_ok=True)
    token = uuid.uuid4().hex
    final_records = {
        "scene": tfrecord_path(records_dir, f"scene_{subset}"),
        "lens": tfrecord_path(records_dir, f"lens_{subset}"),
    }
    if subset != "train":
        final_records["dirty"] = tfrecord_path(records_dir, f"dirty_{subset}")
    final_manifest = os.path.join(records_dir, f"manifest_{subset}.csv")
    temp_records = {key: _temp_path(path, token) for key, path in final_records.items()}
    temp_manifest = _temp_path(final_manifest, token)

    rng = np.random.default_rng(seed)
    labels = np.repeat(np.array([0, 1], dtype=np.int8), count // 2)
    rng.shuffle(labels)
    example_seeds = rng.integers(0, np.iinfo(np.int64).max, size=count, dtype=np.int64)
    writers: dict[str, tf.io.TFRecordWriter] = {}
    rows: list[dict[str, Any]] = []
    try:
        writers = {key: tf.io.TFRecordWriter(path) for key, path in temp_records.items()}

        jobs = list(zip(labels, example_seeds, strict=True))

        def _generate(job):
            label, example_seed = job
            return generator.generate_example(
                np.random.default_rng(int(example_seed)),
                label=int(label),
                fixed_dirty=subset != "train",
            )

        def _examples():
            if int(workers) == 1:
                for job in jobs:
                    yield _generate(job)
                return
            # Only one worker-sized chunk is live at once: parallel simulation
            # without retaining thousands of 510x510x4 examples in futures.
            with ThreadPoolExecutor(max_workers=int(workers)) as executor:
                for start in range(0, len(jobs), int(workers)):
                    futures = [executor.submit(_generate, job) for job in jobs[start : start + int(workers)]]
                    for future in futures:
                        yield future.result()

        for index, (job, example) in enumerate(zip(jobs, _examples(), strict=True)):
            label, example_seed = job
            writers["scene"].write(example.scene.to_tfrecord(index=index))
            writers["lens"].write(example.lens.to_tfrecord(index=index))
            if subset != "train":
                if example.dirty is None:
                    raise ValueError("fixed validation/test examples require dirty images")
                writers["dirty"].write(example.dirty.to_tfrecord(index=index))
            rows.append(
                {
                    "index": index,
                    "split": subset,
                    "example_seed": int(example_seed),
                    **example.row,
                    "label": int(label),
                }
            )
        for writer in writers.values():
            writer.close()
        writers.clear()

        fieldnames = list(rows[0]) if rows else ["index", "split", "example_seed", "label"]
        with open(temp_manifest, "w", newline="", encoding="utf-8") as handle:
            manifest_writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            manifest_writer.writeheader()
            manifest_writer.writerows(rows)

        for key, final_path in final_records.items():
            os.replace(temp_records[key], final_path)
        os.replace(temp_manifest, final_manifest)
    finally:
        for writer in writers.values():
            writer.close()
        for path in (*temp_records.values(), temp_manifest):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)

    return SplitSummary(subset, count, count // 2, count // 2)


def dataset_fingerprint(records_dir: str) -> str:
    """Hash the byte content and names of published experiment records."""
    digest = hashlib.sha256()
    if not os.path.isdir(records_dir):
        return digest.hexdigest()
    for name in sorted(os.listdir(records_dir)):
        if ".tmp-" in name or not (name.startswith("manifest_") or name.endswith(".tfrecord")):
            continue
        full = os.path.join(records_dir, name)
        digest.update(name.encode("utf-8"))
        if name.startswith("manifest_"):
            with open(full, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            # TFRecords can be tens of gigabytes. The paired manifests are the
            # replay identity; include record sizes and boundary bytes to catch
            # truncation/replacement without re-reading the whole dataset at
            # every training launch.
            size = os.path.getsize(full)
            digest.update(str(size).encode("ascii"))
            with open(full, "rb") as handle:
                digest.update(handle.read(64 * 1024))
                if size > 64 * 1024:
                    handle.seek(max(0, size - 64 * 1024))
                    digest.update(handle.read(64 * 1024))
    return digest.hexdigest()
