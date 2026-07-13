"""Atomic normal-format paired records for the lens-isolation experiment."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.experiments.lens_isolation.config import SCHEMA_VERSION
from euclid_polish.image.tfio import parse_example, tfrecord_path
from euclid_polish.observability import Reporter
from euclid_polish.sky.generation.source_catalog import SourceCatalogWriter

_SUBSETS = {"train", "validate", "test"}


@dataclass(frozen=True)
class SplitSummary:
    subset: str
    count: int
    reused: bool = False


def _record_count(path: str) -> int:
    if not os.path.isfile(path):
        return -1
    try:
        return sum(1 for _ in tf.data.TFRecordDataset([path]))
    except tf.errors.OpError:
        return -1


def _split_metadata_path(records_dir: str, subset: str) -> str:
    return os.path.join(records_dir, f"split_{subset}.json")


def _read_json(path: str) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, TypeError):
        return None


def _file_fingerprint(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_paths(records_dir: str, subset: str) -> dict[str, str]:
    return {
        "dirty": tfrecord_path(records_dir, f"dirty_{subset}"),
        "lens": tfrecord_path(records_dir, f"lens_{subset}"),
        "sources": os.path.join(records_dir, f"sources_{subset}.csv"),
    }


def validate_split(
    records_dir: str,
    subset: str,
    expected_count: int,
    *,
    config_fingerprint: str | None = None,
) -> bool:
    """Return whether one published dirty/lens/source set is reusable."""
    if subset not in _SUBSETS:
        raise ValueError(f"unknown subset {subset!r}")
    paths = _split_paths(records_dir, subset)
    metadata = _read_json(_split_metadata_path(records_dir, subset))
    if metadata is None or int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        return False
    if int(metadata.get("count", -1)) != int(expected_count):
        return False
    if config_fingerprint is not None and metadata.get("config_fingerprint") != config_fingerprint:
        return False
    if not all(os.path.isfile(path) for path in paths.values()):
        return False
    if any(_record_count(paths[kind]) != expected_count for kind in ("dirty", "lens")):
        return False
    fingerprints = metadata.get("fingerprints")
    if not isinstance(fingerprints, dict):
        return False
    shapes = metadata.get("shapes")
    if not isinstance(shapes, dict):
        return False
    try:
        if not all(fingerprints.get(name) == _file_fingerprint(path) for name, path in paths.items()):
            return False
        for name, channels in (("dirty", Config.NUM_LR_CHANNELS), ("lens", Config.NUM_HR_CHANNELS)):
            expected_shape = tuple(shapes.get(name, ()))
            if (
                len(expected_shape) != 3
                or any(not isinstance(value, int) or value < 1 for value in expected_shape)
            ):
                return False
            if expected_shape[-1] != channels:
                return False
            raw = next(iter(tf.data.TFRecordDataset([paths[name]])))
            actual_shape = tuple(int(value) for value in parse_example(raw, channels).shape)
            if actual_shape != expected_shape:
                return False
        return True
    except (OSError, StopIteration, tf.errors.OpError, ValueError):
        return False


def _temporary_path(final_path: str, token: str) -> str:
    return f"{final_path}.tmp-{token}"


def _write_json_atomic(path: str, payload: Mapping[str, Any]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".tmp-",
        dir=os.path.dirname(path) or ".",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
    return path


def generate_split(
    generator,
    records_dir: str,
    subset: str,
    *,
    count: int,
    seed: int,
    config_fingerprint: str,
    force: bool = False,
    workers: int = 1,
) -> SplitSummary:
    """Generate and atomically publish one unbiased normal-field split.

    The split contains every field that the ordinary simulator produces.  A
    field with zero lenses is a valid all-zero target; a field with many lenses
    keeps every captured system.  Per-field seeds and ordered writer output
    make serial and worker-assisted execution reproducible.
    """
    if subset not in _SUBSETS:
        raise ValueError(f"unknown subset {subset!r}")
    if int(count) < 0:
        raise ValueError("count must be non-negative")
    if int(workers) < 1:
        raise ValueError("workers must be >= 1")
    if not force and validate_split(
        records_dir,
        subset,
        count,
        config_fingerprint=config_fingerprint,
    ):
        return SplitSummary(subset=subset, count=count, reused=True)

    os.makedirs(records_dir, exist_ok=True)
    token = uuid.uuid4().hex
    final_paths = _split_paths(records_dir, subset)
    temporary_paths = {name: _temporary_path(path, token) for name, path in final_paths.items()}
    temporary_metadata = _temporary_path(_split_metadata_path(records_dir, subset), token)
    reporter = Reporter.from_env()
    reporter.set_stage(f"lens isolation: generate {subset}")
    reporter.set_worker_step(0, 0, count, subset)
    master_rng = np.random.default_rng(seed)
    field_seeds = master_rng.integers(0, np.iinfo(np.int64).max, size=count)

    writers: dict[str, tf.io.TFRecordWriter] = {}
    try:
        writers = {
            name: tf.io.TFRecordWriter(path)
            for name, path in temporary_paths.items()
            if name in {"dirty", "lens"}
        }
        shapes: dict[str, tuple[int, int, int]] = {}

        def generate_one(field_seed: int):
            rng = np.random.default_rng(int(field_seed))
            return generator.generate_example(rng)

        if workers == 1:
            examples = map(generate_one, field_seeds)
        else:
            # Capture state is protected by the adapter, while this preserves
            # deterministic input/output order and lets observation work overlap.
            executor = ThreadPoolExecutor(max_workers=workers)
            examples = executor.map(generate_one, field_seeds)

        try:
            with SourceCatalogWriter(temporary_paths["sources"]) as sources:
                for index, example in enumerate(examples):
                    images = {"dirty": example.dirty, "lens": example.lens}
                    for name, image in images.items():
                        shape = tuple(int(value) for value in image.shape)
                        expected_channels = (
                            Config.NUM_LR_CHANNELS if name == "dirty" else Config.NUM_HR_CHANNELS
                        )
                        if (
                            len(shape) != 3
                            or any(value < 1 for value in shape)
                            or shape[-1] != expected_channels
                        ):
                            raise ValueError(f"{name} record has incompatible shape {shape}")
                        previous = shapes.setdefault(name, shape)
                        if previous != shape:
                            raise ValueError(
                                f"{name} records must share one shape; saw {previous} then {shape}"
                            )
                    writers["dirty"].write(example.dirty.to_tfrecord(index=index))
                    writers["lens"].write(example.lens.to_tfrecord(index=index))
                    sources.add_field(index, example.sources)
                    reporter.set_worker_step(0, index + 1, count, subset)
        finally:
            if workers != 1:
                executor.shutdown(wait=True, cancel_futures=True)
        for writer in writers.values():
            writer.close()
        writers.clear()

        fingerprints = {name: _file_fingerprint(path) for name, path in temporary_paths.items()}
        split_metadata = {
            "schema_version": SCHEMA_VERSION,
            "subset": subset,
            "count": int(count),
            "seed": int(seed),
            "config_fingerprint": config_fingerprint,
            "fingerprints": fingerprints,
            "shapes": {name: list(shape) for name, shape in shapes.items()},
        }
        _write_json_atomic(temporary_metadata, split_metadata)
        for name, final_path in final_paths.items():
            os.replace(temporary_paths[name], final_path)
        os.replace(temporary_metadata, _split_metadata_path(records_dir, subset))
    finally:
        for writer in writers.values():
            writer.close()
        for path in (*temporary_paths.values(), temporary_metadata):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)

    return SplitSummary(subset=subset, count=count)


def dataset_fingerprint(records_dir: str) -> str:
    """Hash published experiment record files, sidecars, and split metadata."""
    digest = hashlib.sha256()
    if not os.path.isdir(records_dir):
        return digest.hexdigest()
    for name in sorted(os.listdir(records_dir)):
        if ".tmp-" in name or not (
            name.endswith(".tfrecord") or name.startswith("sources_") or name.startswith("split_")
        ):
            continue
        path = os.path.join(records_dir, name)
        if not os.path.isfile(path):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(_file_fingerprint(path).encode("ascii"))
    return digest.hexdigest()


def write_dataset_metadata(
    records_dir: str,
    *,
    config: Mapping[str, Any],
    master_seed: int,
    split_summaries: Mapping[str, SplitSummary],
    source_commit: str,
) -> str:
    """Atomically record the complete dataset identity after all split writes."""
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "config": dict(config),
        "master_seed": int(master_seed),
        "source_commit": source_commit,
        "splits": {name: asdict(summary) for name, summary in split_summaries.items()},
        "fingerprint": dataset_fingerprint(records_dir),
    }
    return _write_json_atomic(os.path.join(records_dir, "dataset.json"), metadata)
