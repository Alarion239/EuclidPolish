"""Resume logic in scripts/run_pipeline.py: a subset is 'complete' when its
final TFRecords have the requested record count and the sidecar exists."""

from __future__ import annotations

import importlib.util
import os

import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.euclid.psf_library import load_all_band_psfs
from euclid_polish.sky.observation_simulator import (
    ObservationSimulator, ObservationSimulatorConfig,
)
from euclid_polish.sky.sky_simulator import (
    SkySimulatorConfig, SkySimulator,
)
from euclid_polish.image.tfio import tfrecord_path
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


def _write_dummy_tfrecord(path: str, n: int) -> None:
    with tf.io.TFRecordWriter(path) as w:
        for i in range(n):
            w.write(f"rec{i}".encode())


# --------------------------------------------------------------------------
# _count_tfrecords
# --------------------------------------------------------------------------

def test_count_tfrecords_counts_records(tmp_path):
    p = str(tmp_path / "x.tfrecord")
    _write_dummy_tfrecord(p, 5)
    assert rp._count_tfrecords(p) == 5


def test_count_tfrecords_missing_returns_none(tmp_path):
    assert rp._count_tfrecords(str(tmp_path / "nope.tfrecord")) is None


def test_count_tfrecords_truncated_returns_none(tmp_path):
    p = str(tmp_path / "trunc.tfrecord")
    _write_dummy_tfrecord(p, 3)
    with open(p, "r+b") as f:          # chop the last 4 bytes → DataLossError
        f.truncate(os.path.getsize(p) - 4)
    assert rp._count_tfrecords(p) is None


# --------------------------------------------------------------------------
# _sources_complete
# --------------------------------------------------------------------------

def test_sources_complete_existing_file(tmp_path):
    p = str(tmp_path / "sources_train.csv")
    open(p, "w").write("field_index,type\n0,galaxy\n")
    assert rp._sources_complete(p, expected_n=4) is True


def test_sources_complete_missing_file(tmp_path):
    assert rp._sources_complete(str(tmp_path / "nope.csv"), expected_n=4) is False


def test_sources_complete_zero_expected_is_trivially_true(tmp_path):
    assert rp._sources_complete(str(tmp_path / "nope.csv"), expected_n=0) is True


# --------------------------------------------------------------------------
# _subset_complete
# --------------------------------------------------------------------------

def _make_subset(tmp_path, subset, n, kinds=("clean", "hr", "dirty")):
    for kind in kinds:
        _write_dummy_tfrecord(tfrecord_path(str(tmp_path), f"{kind}_{subset}"), n)
    sidecar = tfrecord_path(str(tmp_path), f"sources_{subset}").replace(
        ".tfrecord", ".csv")
    open(sidecar, "w").write("field_index,type\n0,galaxy\n")


def test_subset_complete_all_kinds_present(tmp_path):
    _make_subset(tmp_path, "train", 4)
    assert rp._subset_complete(
        str(tmp_path), "train", ("clean", "hr", "dirty", "sources"), 4) is True


def test_subset_incomplete_when_kind_short(tmp_path):
    _make_subset(tmp_path, "train", 4)
    _write_dummy_tfrecord(tfrecord_path(str(tmp_path), "hr_train"), 3)  # short
    assert rp._subset_complete(
        str(tmp_path), "train", ("clean", "hr", "dirty", "sources"), 4) is False


def test_subset_incomplete_when_count_mismatch(tmp_path):
    _make_subset(tmp_path, "train", 4)
    assert rp._subset_complete(            # asked for 8, only 4 on disk
        str(tmp_path), "train", ("clean", "hr", "dirty", "sources"), 8) is False


def test_subset_incomplete_when_sidecar_missing(tmp_path):
    _make_subset(tmp_path, "train", 4)
    os.remove(tfrecord_path(str(tmp_path), "sources_train").replace(
        ".tfrecord", ".csv"))
    assert rp._subset_complete(
        str(tmp_path), "train", ("clean", "hr", "dirty", "sources"), 4) is False


# --------------------------------------------------------------------------
# _cleanup_parts
# --------------------------------------------------------------------------

def test_cleanup_parts_removes_only_subset_parts(tmp_path):
    rdir = str(tmp_path)
    # Orphan parts from a dead train run, plus a final file and a validate part.
    for name in ("clean_train.part0000.tfrecord", "hr_train.part0003.tfrecord",
                 "dirty_train.part0001.tfrecord", "sources_train.part0000.csv"):
        open(os.path.join(rdir, name), "w").close()
    open(os.path.join(rdir, "clean_train.tfrecord"), "w").close()      # final
    open(os.path.join(rdir, "clean_validate.part0000.tfrecord"), "w").close()

    rp._cleanup_parts(rdir, "train")

    left = sorted(os.listdir(rdir))
    assert "clean_train.tfrecord" in left                  # final kept
    assert "clean_validate.part0000.tfrecord" in left      # other subset kept
    assert not any(".part" in n and "train" in n for n in left)  # train parts gone


# --------------------------------------------------------------------------
# Integration: real records via the worker core
# --------------------------------------------------------------------------

def _sim_fwd():
    cat = TinyCosmosCatalog(n_galaxies=200, seed=0)
    sim = SkySimulator(
        cat, SkySimulatorConfig(image_size=96,
                                      pixel_scale=Config.DEFAULT_PIXEL_SCALE))
    psfs = load_all_band_psfs(psf_dir="/nonexistent_dir_for_test")  # Gaussian
    fwd = ObservationSimulator(psfs_by_band=psfs,
                           config=ObservationSimulatorConfig(add_noise=True))
    return sim, fwd


def _build_complete_subset(rdir, subset, n):
    """Generate one shard covering [0, n) and merge it to final files —
    exactly the on-disk state of a completed subset."""
    sim, fwd = _sim_fwd()
    rp._generate_convolve_range(sim, fwd, rdir, subset, 0, n, 0, seed=[1, 1, 0])
    for kind in ("clean", "hr", "dirty"):
        rp._concat_tfrecords(
            [tfrecord_path(rdir, f"{kind}_{subset}.part0000")],
            tfrecord_path(rdir, f"{kind}_{subset}"))
    rp.concat_source_csvs(
        [tfrecord_path(rdir, f"sources_{subset}.part0000").replace(
            ".tfrecord", ".csv")],
        tfrecord_path(rdir, f"sources_{subset}").replace(".tfrecord", ".csv"))


def test_completed_subset_detected_and_truncation_busts_it(tmp_path):
    rdir = str(tmp_path)
    _build_complete_subset(rdir, "train", 4)
    kinds = ("clean", "hr", "dirty", "sources")
    assert rp._subset_complete(rdir, "train", kinds, 4) is True
    assert rp._subset_complete(rdir, "validate", kinds, 4) is False  # not built

    # A clean TFRecord truncated by a mid-merge kill must NOT read as complete.
    clean = tfrecord_path(rdir, "clean_train")
    with open(clean, "r+b") as f:
        f.truncate(os.path.getsize(clean) - 8)
    assert rp._subset_complete(rdir, "train", kinds, 4) is False


# --------------------------------------------------------------------------
# Integration: step_convolve resume + --force
# --------------------------------------------------------------------------

class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _convolve_args(rdir, ntrain, nvalid, force):
    return _Args(records_dir=rdir, psf_dir="/nonexistent_dir_for_test",
                 require_empirical_psf=False, ntrain=ntrain, nvalid=nvalid,
                 force=force)


def test_step_convolve_resumes_then_force_regenerates(tmp_path, monkeypatch):
    rdir = str(tmp_path)
    # Build a complete clean_train (4 records) — the input convolve reads.
    sim, fwd = _sim_fwd()
    rp._generate_convolve_range(sim, fwd, rdir, "train", 0, 4, 0, seed=[1, 1, 0])
    rp._concat_tfrecords([tfrecord_path(rdir, "clean_train.part0000")],
                         tfrecord_path(rdir, "clean_train"))
    # Remove hr/dirty so the first convolve must produce them.
    for kind in ("hr", "dirty"):
        os.remove(tfrecord_path(rdir, f"{kind}_train.part0000"))

    opened = []
    real = rp.open_writer

    def spy(name, **kw):
        opened.append(name)
        return real(name, **kw)

    monkeypatch.setattr(rp, "open_writer", spy)

    # First run: hr_train + dirty_train get written.
    rp.step_convolve(_convolve_args(rdir, ntrain=4, nvalid=0, force=False))
    assert "hr_train" in opened and "dirty_train" in opened

    # Second run: train is complete now → no writers opened (skipped).
    opened.clear()
    rp.step_convolve(_convolve_args(rdir, ntrain=4, nvalid=0, force=False))
    assert opened == []

    # --force: writers reopen even though train is complete.
    opened.clear()
    rp.step_convolve(_convolve_args(rdir, ntrain=4, nvalid=0, force=True))
    assert "hr_train" in opened and "dirty_train" in opened
