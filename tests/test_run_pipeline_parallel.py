"""Parallel synthetic-generation shard logic in scripts/run_pipeline.py.

The ProcessPoolExecutor + per-worker catalog load is thin glue over two
testable pieces: the per-shard worker core (``_generate_convolve_range``)
and the byte-concat merge (``_concat_tfrecords``). Pairing is by position
(``tf.data.Dataset.zip``), so concatenating shards in id order must keep
clean/hr/dirty aligned and readable by the dataset loader.
"""

from __future__ import annotations

import importlib.util
import os

import tensorflow as tf

import numpy as np

from euclid_polish.config import Config
from euclid_polish.image import Image, ImageSet
from euclid_polish.image.tfio import read_images, tfrecord_path
from euclid_polish.sky.generation.source_catalog import read_sources
from euclid_polish.psf.psf_library import load_all_band_psfs
from euclid_polish.sky.generation.sky_simulator import (
    SkySimulator,
    SkySimulatorConfig,
)
from euclid_polish.sky.observation.observation_simulator import (
    ObservationSimulator,
    ObservationSimulatorConfig,
)
from tests._tiny_catalog import TinyCosmosCatalog


def _load_run_pipeline():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "run_pipeline.py",
    )
    spec = importlib.util.spec_from_file_location("run_pipeline_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rp = _load_run_pipeline()


def _sim_fwd():
    cat = TinyCosmosCatalog(n_galaxies=200, seed=0)
    sim = SkySimulator(
        cat, SkySimulatorConfig(image_size=96,
                                      pixel_scale=Config.DEFAULT_PIXEL_SCALE),
    )
    psfs = load_all_band_psfs(psf_dir="/nonexistent_dir_for_test")  # Gaussian
    fwd = ObservationSimulator(psfs_by_band=psfs,
                           config=ObservationSimulatorConfig(add_noise=True))
    return sim, fwd


def _count(path: str) -> int:
    return sum(1 for _ in tf.data.TFRecordDataset(path))


def test_subset_rng_deterministic_from_run_seed():
    """The whole run is reproducible from one run_seed: same seed -> identical
    draws; different subset or run_seed -> different draws."""
    a = rp._subset_rng(42, "train", rp._STREAM_GEN).random(5)
    b = rp._subset_rng(42, "train", rp._STREAM_GEN).random(5)
    assert (a == b).all()                                 # same seed -> identical
    assert not (a == rp._subset_rng(42, "validate", rp._STREAM_GEN).random(5)).all()
    assert not (a == rp._subset_rng(43, "train", rp._STREAM_GEN).random(5)).all()
    # Distinct streams (gen vs forward) don't share a sequence for one seed.
    assert not (a == rp._subset_rng(42, "train", rp._STREAM_FWD).random(5)).all()


def test_resolve_run_seed_honours_arg_else_entropy():
    import argparse
    assert rp._resolve_run_seed(argparse.Namespace(seed=123)) == 123
    s1 = rp._resolve_run_seed(argparse.Namespace(seed=-1))
    s2 = rp._resolve_run_seed(argparse.Namespace(seed=-1))
    assert s1 != s2                                       # -1 -> fresh entropy


def test_shard_bounds_partition_is_contiguous():
    b = rp._shard_bounds(10, 3)
    assert b[0][0] == 0 and b[-1][1] == 10
    for (_s0, e0), (s1, _e1) in zip(b, b[1:], strict=False):
        assert e0 == s1                     # no gaps / overlaps
    assert sum(e - s for s, e in b) == 10   # full coverage


def test_concat_tfrecords_skips_missing(tmp_path):
    out = str(tmp_path / "out.tfrecord")
    rp._concat_tfrecords([str(tmp_path / "nope.tfrecord")], out)
    assert os.path.exists(out) and os.path.getsize(out) == 0


def test_worker_emits_progress_events(tmp_path, monkeypatch):
    """Each worker reports its shard progress to the shared events file, so
    the job-status consumer can fold a cumulative count + active workers."""
    from euclid_polish.web.job_status import fold_events
    sim, fwd = _sim_fwd()
    events = tmp_path / "ev.jsonl"
    monkeypatch.setenv("EUCLID_POLISH_EVENTS_PATH", str(events))
    rp._generate_convolve_range(sim, fwd, str(tmp_path), "train", 0, 2, 3,
                                seed=[1, 1, 3])
    text = events.read_text()
    assert '"kind":"worker"' in text
    # Prepend the parent's parallel announcement and fold: the shard wrote 2.
    s = fold_events(
        '{"ts":1,"kind":"parallel","value":{"total":2,"workers":1}}\n' + text
    )
    assert s.parallel is not None
    assert s.parallel.current == 2 and s.parallel.total == 2


def test_parallel_shards_merge_into_paired_records(tmp_path):
    sim, fwd = _sim_fwd()
    rdir = str(tmp_path)
    # Two shards covering [0,2) and [2,4) of the train subset — exactly what
    # two pool workers would produce.
    rp._generate_convolve_range(sim, fwd, rdir, "train", 0, 2, 0, seed=[1, 1, 0])
    rp._generate_convolve_range(sim, fwd, rdir, "train", 2, 2, 1, seed=[1, 1, 1])

    for kind in ("clean", "hr", "dirty"):
        parts = [tfrecord_path(rdir, f"{kind}_train.part{sid:04d}") for sid in (0, 1)]
        # Each shard wrote its 2 records.
        assert all(_count(p) == 2 for p in parts)
        rp._concat_tfrecords(parts, tfrecord_path(rdir, f"{kind}_train"))
        # Merged file has all 4, in shard order.
        assert _count(tfrecord_path(rdir, f"{kind}_train")) == 4

    # The merged records are readable and stayed aligned across the shard boundary.
    lr_imgs = list(ImageSet.read(tfrecord_path(rdir, "dirty_train")))
    hr_imgs = list(ImageSet.read(tfrecord_path(rdir, "clean_train")))
    assert len(lr_imgs) == 4
    assert lr_imgs[0].data.shape[-1] == Config.NUM_LR_CHANNELS
    assert hr_imgs[0].data.shape[-1] == Config.NUM_HR_CHANNELS


def test_worker_stamps_records_when_plan_given(tmp_path):
    """A pre-minted ShardStampPlan stamps clean/hr/dirty in the worker; hr+dirty
    parent on the clean file id."""
    from euclid_polish.image.tfio import read_images
    from euclid_polish.provenance.ids import ProvId
    from euclid_polish.sky.generation.gen_provenance import ShardStampPlan
    sim, fwd = _sim_fwd()
    rdir = str(tmp_path)
    plan = ShardStampPlan(run_id=ProvId("7f3a9c21"), clean_id=ProvId("4b1e7a90"),
                          hr_id=ProvId("2b8e44d1"), dirty_id=ProvId("9c1f0a7d"))
    rp._generate_convolve_range(sim, fwd, rdir, "train", 0, 2, 0,
                                seed=[1, 1, 0], plan=plan)

    def _one(kind):
        return read_images(
            tfrecord_path(rdir, f"{kind}_train.part0000"), num_images=1)[0]

    clean, hr, dirty = _one("clean"), _one("hr"), _one("dirty")
    assert clean.prov_stamp().id == ProvId("4b1e7a90")
    assert hr.prov_stamp().id == ProvId("2b8e44d1")
    assert dirty.prov_stamp().id == ProvId("9c1f0a7d")
    assert hr.prov_stamp().parents == (ProvId("4b1e7a90"),)
    assert dirty.prov_stamp().parents == (ProvId("4b1e7a90"),)
    assert clean.prov_stamp().produced_by == ProvId("7f3a9c21")


# ---------------------------------------------------------------------------
# --skip-dirty-train: the train split never runs the forward model
# ---------------------------------------------------------------------------

def test_skip_dirty_shard_writes_no_dirty_and_hr_is_trimmed_clean(tmp_path):
    """write_dirty=False: no dirty part, no forward-model call; hr is the
    plain trim of the scene (position-paired with clean)."""
    import numpy as np

    from euclid_polish.image.tfio import read_images

    sim, fwd = _sim_fwd()

    class _Boom:                              # forward must NEVER run
        def process(self, *a, **k):
            raise AssertionError("forward model ran despite write_dirty=False")

    rdir = str(tmp_path)
    rp._generate_convolve_range(sim, _Boom(), rdir, "train", 0, 2, 0,
                                seed=[1, 1, 0], write_dirty=False)
    assert not os.path.exists(tfrecord_path(rdir, "dirty_train.part0000"))
    clean = read_images(tfrecord_path(rdir, "clean_train.part0000"), 2)
    hr = read_images(tfrecord_path(rdir, "hr_train.part0000"), 2)
    assert len(clean) == len(hr) == 2
    for c, h in zip(clean, hr, strict=False):
        np.testing.assert_array_equal(h.data, c.data)   # 96 % 2 == 0 → no-op trim
    # sources sidecar still written
    src = tfrecord_path(rdir, "sources_train.part0000").replace(".tfrecord", ".csv")
    assert os.path.isfile(src)


# ---------------------------------------------------------------------------
# Starless scenes + stars re-injected in the forward (stars-as-artifacts)
# ---------------------------------------------------------------------------

def _sim_fwd_dense_stars():
    sim = SkySimulator(
        TinyCosmosCatalog(n_galaxies=200, seed=0),
        SkySimulatorConfig(image_size=96, pixel_scale=Config.DEFAULT_PIXEL_SCALE,
                           star_density_arcmin2=5000.0))       # guarantee stars
    fwd = ObservationSimulator(
        psfs_by_band=load_all_band_psfs(psf_dir="/nonexistent_dir_for_test"),
        config=ObservationSimulatorConfig(add_noise=False))
    return sim, fwd


def test_forward_starless_injects_into_lr_not_target():
    """The shared helper forwards the WITH-stars scene (→ LR carries the star
    flux) but returns the STARLESS scene as the target."""
    _sim, fwd = _sim_fwd_dense_stars()
    field = np.abs(np.random.default_rng(0).normal(
        10, 2, (96, 96, 4))).astype(np.float32)
    sky = Image(data=field, pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
                band_names=Config.LR_INPUT_BAND_NAMES, is_clean=True,
                index=0, subset="validate")
    star = [{"x_pix": 48.0, "y_pix": 48.0, "mag_vis": 17.0}]
    lr_s, hr_s = rp._forward_starless(fwd, sky, star, np.random.default_rng(1))
    lr_0, hr_0 = rp._forward_starless(fwd, sky, None, np.random.default_rng(1))
    # target is the starless scene either way (no star flux, no rng)...
    np.testing.assert_array_equal(hr_s.data, field)     # 96 % rebin == 0 → no-op
    np.testing.assert_array_equal(hr_s.data, hr_0.data)
    # ...but the injected star lifts the LR flux (same noise seed).
    assert lr_s.data.sum() > lr_0.data.sum()


def test_validate_shard_records_stars_target_starless(tmp_path):
    """Validate split: fixed stars land in the CSV and the LR (dirty), but the
    clean/hr target is the starless scene."""
    sim, fwd = _sim_fwd_dense_stars()
    rdir = str(tmp_path)
    rp._generate_convolve_range(sim, fwd, rdir, "validate", 0, 2, 0,
                                seed=[1, 1, 0], write_dirty=True)
    src = tfrecord_path(rdir, "sources_validate.part0000").replace(
        ".tfrecord", ".csv")
    stars = [r for rows in read_sources(src).values()
             for r in rows if r["type"] == "star"]
    assert len(stars) > 0                                   # fixed stars recorded
    clean = read_images(tfrecord_path(rdir, "clean_validate.part0000"), 2)
    hr = read_images(tfrecord_path(rdir, "hr_validate.part0000"), 2)
    for c, h in zip(clean, hr, strict=False):
        np.testing.assert_array_equal(h.data, c.data)      # target = starless scene


def test_train_shard_draws_no_fixed_stars(tmp_path):
    """Train split draws NO fixed stars — the on-the-fly forward injects a
    fresh realization per visit, so none are baked into the records/CSV."""
    sim, fwd = _sim_fwd_dense_stars()
    rdir = str(tmp_path)
    rp._generate_convolve_range(sim, fwd, rdir, "train", 0, 2, 0,
                                seed=[1, 1, 0], write_dirty=False)
    src = tfrecord_path(rdir, "sources_train.part0000").replace(
        ".tfrecord", ".csv")
    stars = [r for rows in read_sources(src).values()
             for r in rows if r["type"] == "star"]
    assert stars == []


def test_merge_subset_kinds_skips_dirty(tmp_path):
    sim, fwd = _sim_fwd()
    rdir = str(tmp_path)
    rp._generate_convolve_range(sim, fwd, rdir, "train", 0, 2, 0,
                                seed=[1, 1, 0], write_dirty=False)
    rp._merge_subset(rdir, "train", kinds=("clean", "hr"))
    assert _count(tfrecord_path(rdir, "clean_train")) == 2
    assert _count(tfrecord_path(rdir, "hr_train")) == 2
    # no dirty output — not even an empty file
    assert not os.path.exists(tfrecord_path(rdir, "dirty_train"))


def test_salvage_with_kinds_ignores_missing_dirty(tmp_path):
    sim, fwd = _sim_fwd()
    rdir = str(tmp_path)
    rp._generate_convolve_range(sim, fwd, rdir, "train", 0, 2, 0,
                                seed=[1, 1, 0], write_dirty=False)
    done, used, next_sid = rp._salvage_subset(rdir, "train", 4,
                                              kinds=("clean", "hr"))
    assert done == 2 and sorted(used) == [0, 1] and next_sid == 1


def test_remove_subset_finals_dirty_only(tmp_path):
    import json as _json
    rdir = str(tmp_path)
    dirty = tfrecord_path(rdir, "dirty_train")
    open(dirty, "w").write("stale")
    keep = tfrecord_path(rdir, "dirty_validate")
    open(keep, "w").write("keep")
    for name, kind, subset in (("aa11.skytfrecordartifact.json", "dirty", "train"),
                               ("bb22.skytfrecordartifact.json", "clean", "train"),
                               ("cc33.skytfrecordartifact.json", "dirty", "validate")):
        with open(os.path.join(rdir, name), "w") as f:
            _json.dump({"descriptors": {"kind": kind, "subset": subset}}, f)
    rp._remove_subset_finals(rdir, "train", kinds=("dirty",))
    assert not os.path.exists(dirty)
    assert os.path.exists(keep)                       # other subsets untouched
    assert not os.path.exists(os.path.join(rdir, "aa11.skytfrecordartifact.json"))
    assert os.path.exists(os.path.join(rdir, "bb22.skytfrecordartifact.json"))
    assert os.path.exists(os.path.join(rdir, "cc33.skytfrecordartifact.json"))


def test_remove_subset_finals_all_kinds_scoped_to_subset(tmp_path):
    rdir = str(tmp_path)
    for kind in ("clean", "hr", "dirty"):
        open(tfrecord_path(rdir, f"{kind}_train"), "w").write("stale")
        open(tfrecord_path(rdir, f"{kind}_validate"), "w").write("keep")
    src_tr = tfrecord_path(rdir, "sources_train").replace(".tfrecord", ".csv")
    src_va = tfrecord_path(rdir, "sources_validate").replace(".tfrecord", ".csv")
    open(src_tr, "w").write("stale")
    open(src_va, "w").write("keep")
    rp._remove_subset_finals(rdir, "train")
    for kind in ("clean", "hr", "dirty"):
        assert not os.path.exists(tfrecord_path(rdir, f"{kind}_train"))
        assert os.path.exists(tfrecord_path(rdir, f"{kind}_validate"))
    assert not os.path.exists(src_tr)
    assert os.path.exists(src_va)


def test_subset_incomplete_while_parts_remain(tmp_path):
    """Leftover shard parts = unfinished merge: the subset must read as
    incomplete even when every FINAL file's record count matches — otherwise
    a run killed mid-merge (new clean merged, stale hr not) resumes as
    'already complete' with mixed-calibration finals."""
    sim, fwd = _sim_fwd()
    rdir = str(tmp_path)
    rp._generate_convolve_range(sim, fwd, rdir, "train", 0, 2, 0, seed=[1, 1, 0])
    rp._merge_subset(rdir, "train")                   # parts deleted → complete
    assert rp._subset_complete(rdir, "train", ("clean", "hr", "dirty"), 2)
    # a lingering part from an interrupted follow-up run flips it back
    part = tfrecord_path(rdir, "hr_train.part0007")
    open(part, "w").write("x")
    assert not rp._subset_complete(rdir, "train", ("clean", "hr", "dirty"), 2)
    os.remove(part)
    assert rp._subset_complete(rdir, "train", ("clean", "hr", "dirty"), 2)


def test_synthetic_step_forwards_skip_dirty_flag():
    from euclid_polish.web.fasrc_pipeline import REGISTRY
    step = REGISTRY.get("synthetic_generate")
    base = {"n_train": 10, "n_valid": 2, "n_test": 2, "image_size": 96,
            "batch_size": 4, "steps": 10}
    assert "--skip-dirty-train" in step.build_command(
        {**base, "skip_dirty_train": "1"})
    assert "--skip-dirty-train" not in step.build_command(base)


def test_tng_density_with_empty_atlas_is_fatal(tmp_path):
    """An empty/purged TNG atlas must ABORT generation, not silently render
    star-only fields (netscratch purge incident, 2026-07-06)."""
    import pytest

    from euclid_polish.sky.generation.sky_simulator import (
        SkySimulator,
        SkySimulatorConfig,
    )

    with pytest.raises(RuntimeError, match="ZERO usable TNG galaxies"):
        SkySimulator(None, SkySimulatorConfig(
            image_size=96, pixel_scale=Config.DEFAULT_PIXEL_SCALE,
            sersic_density_arcmin2=0.0, tng_density_arcmin2=60.0,
            tng_galaxy_dir=str(tmp_path / "empty_atlas")))
