"""Shard planning and shard purity for the parallel generation pass.

Shards come in whole waves of the worker count and hold at most
``_TARGET_FIELDS_PER_SHARD`` fields, so no worker idles through a tail wave.
Each shard resets the donor balance before its first field, so a shard's
records depend only on (run seed, split, shard id), never on which shards
its worker happened to run before.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import math
import os
from concurrent.futures import Future

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.image.tfio import deserialize_image, tfrecord_path
from euclid_polish.provenance.store import ProvStore
from euclid_polish.psf.psf_library import load_all_band_psfs
from euclid_polish.sky.generation.gen_provenance import begin_generation_run
from euclid_polish.sky.generation.sky_simulator import (
    SkySimulator,
    SkySimulatorConfig,
)
from euclid_polish.sky.observation.observation_simulator import (
    ObservationSimulator,
    ObservationSimulatorConfig,
)
from tests.test_sky_simulator_donor_balance import (
    make_simulator,
    write_fake_atlas,
)


def _load_run_pipeline():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "run_pipeline.py",
    )
    spec = importlib.util.spec_from_file_location("run_pipeline_shard_plan", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rp = _load_run_pipeline()
TARGET = rp._TARGET_FIELDS_PER_SHARD


def _old_shard_count(remaining: int, workers: int) -> int:
    """The pre-2026-10 formula: ~256 fields per shard, at least one per worker."""
    return min(remaining, max(workers, math.ceil(remaining / 256)))


def _sizes(remaining: int, n_shards: int) -> list[int]:
    return [end - start for start, end in rp._shard_bounds(remaining, n_shards)]


# ---------------------------------------------------------------------------
# Shard count
# ---------------------------------------------------------------------------

def test_target_fields_per_shard_is_small():
    assert 32 <= TARGET <= 64


@pytest.mark.parametrize("workers", [1, 2, 3, 7, 16, 20, 32, 48])
def test_shards_fill_whole_waves_with_small_shards(workers):
    for remaining in [*range(1, 200), 255, 960, 961, 2000, 6399, 6400, 6401]:
        n_shards = rp._shard_count(remaining, workers)
        sizes = _sizes(remaining, n_shards)

        assert 1 <= n_shards <= remaining
        assert sum(sizes) == remaining and min(sizes) >= 1
        # Every worker runs the same number of shards ...
        assert n_shards == remaining or n_shards % workers == 0
        # ... each holding at most TARGET fields, and never tiny on big splits.
        assert max(sizes) <= TARGET
        if remaining > workers * TARGET:
            assert min(sizes) >= TARGET // 2


def test_train_split_runs_in_whole_waves():
    # 25 shards of 256 on 20 workers used to leave 15 workers idle for a
    # whole tail wave; now every worker runs seven ~46-field shards.
    assert _old_shard_count(6400, 20) == 25
    assert rp._shard_count(6400, 20) == 140
    assert rp._shard_count(6400, 16) == 144
    assert set(_sizes(6400, 140)) == {45, 46}


@pytest.mark.parametrize("workers", [1, 4, 16, 20, 32])
def test_small_splits_keep_one_shard_per_worker(workers):
    """validate/test sized splits keep today's plan (and so their fields)."""
    for remaining in range(1, workers * TARGET + 1):
        assert rp._shard_count(remaining, workers) == min(remaining, workers)
        assert rp._shard_count(remaining, workers) == _old_shard_count(
            remaining, workers,
        )


def test_no_remaining_fields_plans_no_shards():
    assert rp._shard_count(0, 16) == 0
    assert rp._plan_shard_tasks(
        "train", 0, 16, base_idx=0, base_sid=0, run_seed=1,
        plan=None, write_forward=True,
    ) == []


# ---------------------------------------------------------------------------
# Task plan (fresh ids, contiguous indices, per-shard seeds)
# ---------------------------------------------------------------------------

def test_resumed_shortfall_takes_fresh_ids_and_contiguous_indices():
    tasks = rp._plan_shard_tasks(
        "validate", 130, 4, base_idx=100, base_sid=7, run_seed=99,
        plan="plan", write_forward=False,
    )
    subsets, starts, counts, sids, seeds, plans, forwards = zip(*tasks, strict=True)

    assert len(tasks) == rp._shard_count(130, 4) == 4
    assert set(subsets) == {"validate"}
    assert list(sids) == list(range(7, 7 + len(tasks)))
    assert starts[0] == 100
    assert all(s + c == n for s, c, n in zip(starts, counts, starts[1:], strict=False))
    assert starts[-1] + counts[-1] == 230
    assert list(seeds) == [[99, rp._subset_tag("validate"), sid] for sid in sids]
    assert set(plans) == {"plan"} and set(forwards) == {False}


def test_shard_plan_descriptor_records_the_plan():
    tasks = rp._plan_shard_tasks(
        "train", 200, 4, base_idx=10, base_sid=3, run_seed=5,
        plan=None, write_forward=True,
    )
    descriptor = rp._shard_plan_descriptor(tasks, workers=4, salvaged_fields=10)

    assert descriptor == {
        "workers": 4,
        "target_fields_per_shard": TARGET,
        "n_shards": 8,
        "max_fields_per_shard": 25,
        "new_fields": 200,
        "salvaged_fields": 10,
        "first_shard_id": 3,
        "donor_balance": "per_shard",
    }
    json.dumps(descriptor)                              # sidecar-serialisable


# ---------------------------------------------------------------------------
# Shard purity: donor balance resets per shard
# ---------------------------------------------------------------------------

def _forward() -> ObservationSimulator:
    return ObservationSimulator(
        psfs_by_band=load_all_band_psfs(psf_dir="/nonexistent_dir_for_test"),
        config=ObservationSimulatorConfig(add_noise=True),
    )


def _records(path: str) -> list[tuple[int, bytes]]:
    out = []
    for raw in tf.data.TFRecordDataset(path):
        image = deserialize_image(raw)
        out.append((int(image.index), np.asarray(image.data).tobytes()))
    return out


def _shard_outputs(records_dir: str, subset: str, sid: int) -> dict:
    tag = f"{subset}.part{sid:04d}"
    sources = tfrecord_path(records_dir, f"sources_{tag}").replace(
        ".tfrecord", ".csv")
    with open(sources) as handle:
        rows = handle.read()
    return {
        kind: _records(tfrecord_path(records_dir, f"{kind}_{tag}"))
        for kind in ("clean", "hr", "dirty")
    } | {"sources": rows}


@pytest.fixture(scope="module")
def atlas_paths(tmp_path_factory):
    return write_fake_atlas(tmp_path_factory.mktemp("shard_atlas"))


def test_shard_is_identical_on_fresh_and_reused_workers(atlas_paths, tmp_path):
    fwd = _forward()
    fresh_dir, reused_dir = str(tmp_path / "fresh"), str(tmp_path / "reused")
    os.makedirs(fresh_dir)
    os.makedirs(reused_dir)

    fresh = make_simulator(atlas_paths, image_size=144)
    rp._generate_convolve_range(
        fresh, fwd, fresh_dir, "validate", 40, 3, 5, seed=[11, 2, 5],
    )

    reused = make_simulator(atlas_paths, image_size=144)
    # The worker ran another shard first (as the pool's dynamic dispatch does).
    rp._generate_convolve_range(
        reused, fwd, reused_dir, "validate", 0, 6, 4, seed=[11, 2, 4],
    )
    assert reused._morphology_use_counts.sum() > 0
    rp._generate_convolve_range(
        reused, fwd, reused_dir, "validate", 40, 3, 5, seed=[11, 2, 5],
    )

    expected = _shard_outputs(fresh_dir, "validate", 5)
    actual = _shard_outputs(reused_dir, "validate", 5)
    assert [index for index, _ in actual["clean"]] == [40, 41, 42]
    assert actual["clean"] == expected["clean"]
    assert actual["sources"] == expected["sources"]
    assert actual["hr"] == expected["hr"]
    assert actual["dirty"] == expected["dirty"]
    assert expected["sources"].count("\n") > 3          # galaxies were drawn


# ---------------------------------------------------------------------------
# The parallel step end to end (in-process pool): resume + provenance
# ---------------------------------------------------------------------------

class _InlinePool:
    """Runs each submitted shard immediately, in this process."""

    def __init__(self, max_workers, initializer=None, initargs=()):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args):
        future = Future()
        future.set_result(fn(*args))
        return future


def test_parallel_resume_plans_fresh_shards_and_records_the_plan(
    tmp_path, monkeypatch,
):
    records_dir = str(tmp_path / "records")
    os.makedirs(records_dir)
    sim = SkySimulator(None, SkySimulatorConfig(
        image_size=96, pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        galaxy_density_arcmin2=0.0, star_density_arcmin2=0.0,
        lens_density_arcmin2=0.0,
    ))
    fwd = _forward()
    # A killed earlier run left shard 0 with two intact fields.
    rp._generate_convolve_range(
        sim, fwd, records_dir, "train", 0, 2, 0, seed=[3, 1, 0],
    )

    submitted = []
    original_shard = rp._gen_convolve_shard

    def recording_shard(task):
        submitted.append(task)
        return original_shard(task)

    store = ProvStore(str(tmp_path / "prov"))
    # One field per shard: the old formula would plan 2 shards for these 3
    # fields on 2 workers; whole waves of 1-field shards plan 3.
    monkeypatch.setattr(rp, "_TARGET_FIELDS_PER_SHARD", 1)
    monkeypatch.setattr(rp, "ProcessPoolExecutor", _InlinePool)
    monkeypatch.setattr(rp, "_gen_convolve_shard", recording_shard)
    monkeypatch.setattr(rp, "_W_SIM", sim)
    monkeypatch.setattr(rp, "_W_FWD", fwd)
    monkeypatch.setattr(rp, "_W_RECORDS_DIR", records_dir)
    monkeypatch.setattr(
        rp, "make_generation_context",
        lambda cfg, seed=None: begin_generation_run(store, cfg, seed=seed),
    )
    args = rp.parse_args([
        "--records-dir", records_dir, "--seed", "3",
        "--ntrain", "5", "--nvalid", "0", "--ntest", "0",
        "--gen-workers", "2", "--image-size", "96",
        "--galaxy-density-arcmin2", "0", "--lens-density-arcmin2", "0",
        "--star-density-arcmin2", "0",
    ])

    rp.step_generate_and_convolve_parallel(args)

    # The shortfall of 3 fields went to fresh shard ids above the salvaged 0,
    # with field indices continuing after the salvaged ones.
    assert _old_shard_count(3, 2) == 2
    assert [task[3] for task in submitted] == [1, 2, 3]
    assert [(task[1], task[2]) for task in submitted] == [(2, 1), (3, 1), (4, 1)]
    merged = _records(tfrecord_path(records_dir, "clean_train"))
    assert [index for index, _ in merged] == [0, 1, 2, 3, 4]
    assert not glob.glob(os.path.join(records_dir, "*_train.part*"))

    plans = []
    for sidecar in glob.glob(
        os.path.join(records_dir, "*.skytfrecordartifact.json"),
    ):
        with open(sidecar) as handle:
            descriptors = json.load(handle)["descriptors"]
        plans.append((descriptors["kind"], descriptors.get("shard_plan")))
    assert sorted(kind for kind, _ in plans) == ["clean", "dirty", "hr"]
    assert {json.dumps(plan, sort_keys=True) for _, plan in plans} == {
        json.dumps({
            "workers": 2,
            "target_fields_per_shard": 1,
            "n_shards": 3,
            "max_fields_per_shard": 1,
            "new_fields": 3,
            "salvaged_fields": 2,
            "first_shard_id": 1,
            "donor_balance": "per_shard",
        }, sort_keys=True),
    }
