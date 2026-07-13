"""Atomic normal-format paired records for the lens-isolation experiment."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
import tempfile
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.experiments.lens_isolation.config import SCHEMA_VERSION
from euclid_polish.image.tfio import parse_example, tfrecord_path
from euclid_polish.observability import Reporter
from euclid_polish.sky.generation.source_catalog import SourceCatalogWriter, concat_source_csvs

_SUBSETS = {"train", "validate", "test"}


@dataclass(frozen=True)
class SplitSummary:
    subset: str
    count: int
    reused: bool = False


@dataclass(frozen=True)
class ShardSpec:
    """One deterministic contiguous field range within a split."""

    subset: str
    shard_id: int
    start: int
    count: int
    split_seed: int


@dataclass(frozen=True)
class ShardSummary:
    """Published paired records and their shared shapes for one shard."""

    subset: str
    shard_id: int
    start: int
    count: int
    split_seed: int
    shapes: Mapping[str, tuple[int, int, int]]


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


def _shard_bounds(count: int, n_shards: int) -> Iterable[tuple[int, int]]:
    for shard_id in range(n_shards):
        start = count * shard_id // n_shards
        end = count * (shard_id + 1) // n_shards
        yield start, end


def make_shards(
    subset: str,
    *,
    count: int,
    workers: int,
    seed: int,
    start: int = 0,
    first_shard_id: int = 0,
) -> list[ShardSpec]:
    """Split a normal-field subset into deterministic, contiguous ranges."""
    if subset not in _SUBSETS:
        raise ValueError(f"unknown subset {subset!r}")
    if count < 0:
        raise ValueError("count must be non-negative")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if start < 0 or first_shard_id < 0:
        raise ValueError("shard starts and ids must be non-negative")
    if count == 0:
        return []
    n_shards = min(count, max(workers, math.ceil(count / 256)))
    return [
        ShardSpec(subset, first_shard_id + shard_id, start + range_start, end - range_start, int(seed))
        for shard_id, (range_start, end) in enumerate(_shard_bounds(count, n_shards))
        if end > range_start
    ]


def _shard_paths(records_dir: str, shard: ShardSpec) -> dict[str, str]:
    suffix = f".part{shard.shard_id:04d}"
    return {
        "dirty": tfrecord_path(records_dir, f"dirty_{shard.subset}{suffix}"),
        "lens": tfrecord_path(records_dir, f"lens_{shard.subset}{suffix}"),
        "sources": os.path.join(records_dir, f"sources_{shard.subset}{suffix}.csv"),
    }


def _shard_metadata_path(records_dir: str, shard: ShardSpec) -> str:
    return os.path.join(records_dir, f"split_{shard.subset}.part{shard.shard_id:04d}.json")


def _discard_shard(records_dir: str, shard: ShardSpec) -> None:
    paths = (*_shard_paths(records_dir, shard).values(), _shard_metadata_path(records_dir, shard))
    for path in paths:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)
    if not os.path.isdir(records_dir):
        return
    bases = tuple(os.path.basename(path) + ".tmp-" for path in paths)
    for name in os.listdir(records_dir):
        if name.startswith(bases):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(os.path.join(records_dir, name))


def _image_shape(image, name: str) -> tuple[int, int, int]:
    shape = tuple(int(value) for value in image.shape)
    expected_channels = Config.NUM_LR_CHANNELS if name == "dirty" else Config.NUM_HR_CHANNELS
    if len(shape) != 3 or any(value < 1 for value in shape) or shape[-1] != expected_channels:
        raise ValueError(f"{name} record has incompatible shape {shape}")
    return shape


def _field_rng(split_seed: int, field_index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(split_seed), int(field_index)]))


def write_shard(
    generator,
    records_dir: str,
    shard: ShardSpec,
    *,
    config_fingerprint: str,
) -> ShardSummary:
    """Write one process-safe dirty/lens/source shard without shared state."""
    if shard.count < 1:
        raise ValueError("shards must contain at least one field")
    os.makedirs(records_dir, exist_ok=True)
    final_paths = _shard_paths(records_dir, shard)
    token = uuid.uuid4().hex
    temporary_paths = {name: _temporary_path(path, token) for name, path in final_paths.items()}
    temporary_metadata = _temporary_path(_shard_metadata_path(records_dir, shard), token)
    reporter = Reporter.from_env()
    reporter.set_worker_step(shard.shard_id, 0, shard.count, shard.subset)
    writers: dict[str, tf.io.TFRecordWriter] = {}
    shapes: dict[str, tuple[int, int, int]] = {}
    try:
        writers = {
            name: tf.io.TFRecordWriter(temporary_paths[name])
            for name in ("dirty", "lens")
        }
        with SourceCatalogWriter(temporary_paths["sources"]) as sources:
            for local_index in range(shard.count):
                field_index = shard.start + local_index
                example = generator.generate_example(_field_rng(shard.split_seed, field_index))
                for name, image in (("dirty", example.dirty), ("lens", example.lens)):
                    shape = _image_shape(image, name)
                    previous = shapes.setdefault(name, shape)
                    if previous != shape:
                        raise ValueError(f"{name} records must share one shape; saw {previous} then {shape}")
                    writers[name].write(image.to_tfrecord(index=field_index))
                sources.add_field(field_index, example.sources)
                reporter.set_worker_step(shard.shard_id, local_index + 1, shard.count, shard.subset)
        for writer in writers.values():
            writer.close()
        writers.clear()
        _write_json_atomic(
            temporary_metadata,
            {
                "schema_version": SCHEMA_VERSION,
                "subset": shard.subset,
                "shard_id": shard.shard_id,
                "start": shard.start,
                "count": shard.count,
                "seed": shard.split_seed,
                "config_fingerprint": config_fingerprint,
                "shapes": {name: list(shape) for name, shape in shapes.items()},
            },
        )
        for name, final_path in final_paths.items():
            os.replace(temporary_paths[name], final_path)
        os.replace(temporary_metadata, _shard_metadata_path(records_dir, shard))
    finally:
        for writer in writers.values():
            writer.close()
        for path in (*temporary_paths.values(), temporary_metadata):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)
    return ShardSummary(
        shard.subset,
        shard.shard_id,
        shard.start,
        shard.count,
        shard.split_seed,
        dict(shapes),
    )


def _concat_tfrecords(part_paths: Sequence[str], output_path: str) -> None:
    with open(output_path, "wb") as output:
        for path in part_paths:
            with open(path, "rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def clear_shards(records_dir: str, subset: str) -> None:
    prefix = (
        f"dirty_{subset}.part",
        f"lens_{subset}.part",
        f"sources_{subset}.part",
        f"split_{subset}.part",
    )
    with contextlib.suppress(FileNotFoundError):
        for name in os.listdir(records_dir):
            if name.startswith(prefix):
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(os.path.join(records_dir, name))


def clear_split(records_dir: str, subset: str) -> None:
    """Discard one final split and every resumable shard it owns."""
    clear_shards(records_dir, subset)
    for path in (*_split_paths(records_dir, subset).values(), _split_metadata_path(records_dir, subset)):
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)


def _part_shard_ids(records_dir: str, subset: str) -> set[int]:
    if not os.path.isdir(records_dir):
        return set()
    prefixes = (
        f"dirty_{subset}.part",
        f"lens_{subset}.part",
        f"sources_{subset}.part",
        f"split_{subset}.part",
    )
    ids = set()
    for name in os.listdir(records_dir):
        for prefix in prefixes:
            if not name.startswith(prefix):
                continue
            suffix = name.removeprefix(prefix).split(".", 1)[0]
            if suffix.isdigit():
                ids.add(int(suffix))
            break
    return ids


def _summary_from_shard_metadata(
    records_dir: str,
    subset: str,
    shard_id: int,
    *,
    total_count: int,
    split_seed: int,
    config_fingerprint: str,
) -> ShardSummary | None:
    metadata_path = os.path.join(records_dir, f"split_{subset}.part{shard_id:04d}.json")
    metadata = _read_json(metadata_path)
    if not isinstance(metadata, dict):
        return None
    try:
        shard = ShardSpec(
            subset=str(metadata["subset"]),
            shard_id=int(metadata["shard_id"]),
            start=int(metadata["start"]),
            count=int(metadata["count"]),
            split_seed=int(metadata["seed"]),
        )
        shapes = {name: tuple(int(value) for value in metadata["shapes"][name]) for name in ("dirty", "lens")}
    except (KeyError, TypeError, ValueError):
        return None
    paths = _shard_paths(records_dir, shard)
    valid = (
        metadata.get("schema_version") == SCHEMA_VERSION
        and shard.subset == subset
        and shard.shard_id == shard_id
        and shard.count > 0
        and shard.start >= 0
        and shard.start + shard.count <= total_count
        and shard.split_seed == split_seed
        and metadata.get("config_fingerprint") == config_fingerprint
        and all(os.path.isfile(path) for path in paths.values())
        and all(_record_count(paths[name]) == shard.count for name in ("dirty", "lens"))
        and all(
            len(shape) == 3
            and all(value > 0 for value in shape)
            and shape[-1] == (Config.NUM_LR_CHANNELS if name == "dirty" else Config.NUM_HR_CHANNELS)
            for name, shape in shapes.items()
        )
    )
    if not valid:
        return None
    return ShardSummary(
        shard.subset,
        shard.shard_id,
        shard.start,
        shard.count,
        shard.split_seed,
        shapes,
    )


def completed_shards(
    records_dir: str,
    subset: str,
    *,
    total_count: int,
    split_seed: int,
    config_fingerprint: str,
) -> list[ShardSummary]:
    """Discover matching completed shards independently of the current CPUs."""
    completed = []
    seen_ids = _part_shard_ids(records_dir, subset)
    for shard_id in seen_ids:
        summary = _summary_from_shard_metadata(
            records_dir,
            subset,
            shard_id,
            total_count=total_count,
            split_seed=split_seed,
            config_fingerprint=config_fingerprint,
        )
        if summary is None:
            _discard_shard(records_dir, ShardSpec(subset, shard_id, 0, 1, split_seed))
            continue
        completed.append(summary)
    completed.sort(key=lambda summary: (summary.start, summary.shard_id))
    valid = []
    expected_start = 0
    for summary in completed:
        if summary.start < expected_start:
            _discard_shard(
                records_dir,
                ShardSpec(summary.subset, summary.shard_id, summary.start, summary.count, summary.split_seed),
            )
            continue
        valid.append(summary)
        expected_start = summary.start + summary.count
    return valid


def missing_shards(
    subset: str,
    *,
    count: int,
    workers: int,
    seed: int,
    completed: Iterable[ShardSummary],
) -> list[ShardSpec]:
    """Partition only the gaps left by already-published paired shards."""
    completed = sorted(completed, key=lambda summary: (summary.start, summary.shard_id))
    next_shard_id = max((summary.shard_id for summary in completed), default=-1) + 1
    pending = []
    start = 0
    for summary in completed:
        if start < summary.start:
            gap = make_shards(
                subset,
                count=summary.start - start,
                workers=workers,
                seed=seed,
                start=start,
                first_shard_id=next_shard_id,
            )
            pending.extend(gap)
            next_shard_id += len(gap)
        start = summary.start + summary.count
    if start < count:
        pending.extend(
            make_shards(
                subset,
                count=count - start,
                workers=workers,
                seed=seed,
                start=start,
                first_shard_id=next_shard_id,
            )
        )
    return pending
    return completed


def merge_shards(
    records_dir: str,
    subset: str,
    shards: Iterable[ShardSpec | ShardSummary],
    *,
    config_fingerprint: str,
) -> SplitSummary:
    """Publish all completed paired shards in global field-index order."""
    ordered = sorted(shards, key=lambda shard: (shard.start, shard.shard_id))
    if not ordered:
        raise ValueError("cannot merge an empty split")
    if any(shard.subset != subset for shard in ordered):
        raise ValueError("shard subset does not match merge subset")
    expected_start = 0
    if len({shard.shard_id for shard in ordered}) != len(ordered):
        raise ValueError("shard ids must be unique")
    for shard in ordered:
        if shard.start != expected_start or shard.count < 1:
            raise ValueError("shards must cover contiguous field ranges")
        expected_start += shard.count
    total_count = expected_start
    specs = [
        shard if isinstance(shard, ShardSpec) else ShardSpec(
            shard.subset, shard.shard_id, shard.start, shard.count, shard.split_seed
        )
        for shard in ordered
    ]
    part_paths = [_shard_paths(records_dir, shard) for shard in specs]
    if not all(os.path.isfile(path) for paths in part_paths for path in paths.values()):
        raise ValueError("cannot merge incomplete paired shards")
    if any(
        _record_count(paths[name]) != shard.count
        for paths, shard in zip(part_paths, ordered, strict=True)
        for name in ("dirty", "lens")
    ):
        raise ValueError("paired shard record counts do not match their ranges")
    summaries = [shard for shard in ordered if isinstance(shard, ShardSummary)]
    if len(summaries) != len(ordered):
        raise ValueError("merge requires shard summaries with record shapes")
    if len({summary.split_seed for summary in summaries}) != 1:
        raise ValueError("shards must share one split seed")
    shapes = dict(summaries[0].shapes)
    if any(dict(summary.shapes) != shapes for summary in summaries[1:]):
        raise ValueError("paired shards do not share one record shape")

    final_paths = _split_paths(records_dir, subset)
    token = uuid.uuid4().hex
    temporary_paths = {name: _temporary_path(path, token) for name, path in final_paths.items()}
    try:
        for name in ("dirty", "lens"):
            _concat_tfrecords([paths[name] for paths in part_paths], temporary_paths[name])
            os.replace(temporary_paths[name], final_paths[name])
        concat_source_csvs([paths["sources"] for paths in part_paths], temporary_paths["sources"])
        os.replace(temporary_paths["sources"], final_paths["sources"])
        fingerprints = {name: _file_fingerprint(path) for name, path in final_paths.items()}
        _write_json_atomic(
            _split_metadata_path(records_dir, subset),
            {
                "schema_version": SCHEMA_VERSION,
                "subset": subset,
                "count": total_count,
                "seed": summaries[0].split_seed,
                "config_fingerprint": config_fingerprint,
                "fingerprints": fingerprints,
                "shapes": {name: list(shape) for name, shape in shapes.items()},
            },
        )
    finally:
        for path in temporary_paths.values():
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)
    for shard in specs:
        _discard_shard(records_dir, shard)
    return SplitSummary(subset=subset, count=total_count)


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
    """Serial compatibility path using the same deterministic shard format."""
    if not force and validate_split(records_dir, subset, count, config_fingerprint=config_fingerprint):
        return SplitSummary(subset=subset, count=count, reused=True)
    if not make_shards(subset, count=count, workers=workers, seed=seed):
        raise ValueError("lens-isolation splits must contain at least one field")
    if force:
        clear_split(records_dir, subset)
        summaries = []
    else:
        summaries = completed_shards(
            records_dir,
            subset,
            total_count=count,
            split_seed=seed,
            config_fingerprint=config_fingerprint,
        )
    pending = missing_shards(
        subset,
        count=count,
        workers=workers,
        seed=seed,
        completed=summaries,
    )
    summaries.extend(
        write_shard(generator, records_dir, shard, config_fingerprint=config_fingerprint)
        for shard in pending
    )
    return merge_shards(records_dir, subset, summaries, config_fingerprint=config_fingerprint)


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
