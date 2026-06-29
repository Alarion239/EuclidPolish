"""Resume logic in scripts/run_pipeline.py: a subset is 'complete' when its
final TFRecords have the requested record count and the sidecar exists."""

from __future__ import annotations

import importlib.util
import os

import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.image.tfio import tfrecord_path
from euclid_polish.psf.psf_library import load_all_band_psfs
from euclid_polish.sky.generation.sky_simulator import (
    SkySimulator,
    SkySimulatorConfig,
)
from euclid_polish.sky.generation.source_catalog import read_sources
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


# --------------------------------------------------------------------------
# Shard-level resume helpers: salvage intact records from a killed run
# --------------------------------------------------------------------------

def test_salvage_tfrecord_drops_truncated_tail(tmp_path):
    p = str(tmp_path / "clean_train.part0000.tfrecord")
    _write_dummy_tfrecord(p, 4)
    with open(p, "r+b") as f:                 # chop into the 4th record
        f.truncate(os.path.getsize(p) - 6)
    assert rp._salvage_tfrecord(p) == 3       # bad tail dropped
    assert rp._count_tfrecords(p) == 3        # file is valid again


def test_salvage_tfrecord_intact_keeps_all(tmp_path):
    p = str(tmp_path / "clean_train.part0000.tfrecord")
    _write_dummy_tfrecord(p, 5)
    assert rp._salvage_tfrecord(p) == 5
    assert rp._count_tfrecords(p) == 5


def test_salvage_tfrecord_missing_or_empty_returns_zero(tmp_path):
    assert rp._salvage_tfrecord(str(tmp_path / "nope.tfrecord")) == 0
    empty = str(tmp_path / "empty.tfrecord")
    open(empty, "wb").close()
    assert rp._salvage_tfrecord(empty) == 0


def test_truncate_tfrecord_keeps_first_k(tmp_path):
    p = str(tmp_path / "x.tfrecord")
    _write_dummy_tfrecord(p, 6)
    rp._truncate_tfrecord(p, 2)
    assert rp._count_tfrecords(p) == 2


def test_filter_sources_part_keeps_valid_rows_in_set(tmp_path):
    p = str(tmp_path / "sources_train.part0000.csv")
    full = ",".join(["0", "galaxy", "sersic", "1.0", "2.0", "3.0", "0.5", "", ""])
    rows = [
        ",".join(rp.SOURCE_COLS),                 # header
        full,                                     # field 0 — keep
        full.replace("0,galaxy", "1,lens", 1),    # field 1 — drop (not in set)
        full.replace("0,galaxy", "2,galaxy", 1),  # field 2 — keep
        "2,galaxy,sersic,1.0",                    # truncated final row — drop
    ]
    open(p, "w", newline="").write("\r\n".join(rows) + "\r\n")
    rp._filter_sources_part(p, {0, 2})
    out = open(p, newline="").read()
    kept = [ln for ln in out.splitlines() if ln and not ln.startswith("field_index")]
    assert len(kept) == 2                         # field-1 row + partial row gone
    assert all(ln.split(",")[0] in ("0", "2") for ln in kept)
    assert "1,lens" not in out                    # filtered out


def test_existing_part_sids_finds_clean_parts(tmp_path):
    rdir = str(tmp_path)
    for sid in (0, 3, 1):
        open(tfrecord_path(rdir, f"clean_train.part{sid:04d}"), "w").close()
    open(tfrecord_path(rdir, "clean_validate.part0000"), "w").close()  # other subset
    assert rp._existing_part_sids(rdir, "train") == [0, 1, 3]


# --------------------------------------------------------------------------
# Integration: salvage a half-written shard, then resume + merge to complete
# --------------------------------------------------------------------------

def _kill_part_tail(path, nbytes):
    """Simulate a SIGKILL mid-record by chopping bytes off a part file."""
    with open(path, "r+b") as f:
        f.truncate(os.path.getsize(path) - nbytes)


def test_salvage_shard_aligns_views_and_filters_sources(tmp_path):
    rdir = str(tmp_path)
    sim, fwd = _sim_fwd()
    # One shard of 3 fields (indices 0,1,2). Simulate a kill that left dirty
    # one record short of clean/hr.
    rp._generate_convolve_range(sim, fwd, rdir, "train", 0, 3, 0, seed=[1, 1, 0])
    _kill_part_tail(tfrecord_path(rdir, "dirty_train.part0000"), 6)

    kept, idx = rp._salvage_shard(rdir, "train", 0, cap=99)
    assert kept == 2 and idx == [0, 1]                  # aligned down to dirty
    for kind in ("clean", "hr", "dirty"):
        assert rp._count_tfrecords(
            tfrecord_path(rdir, f"{kind}_train.part0000")) == 2
    # Sources sidecar keeps only rows for the surviving fields {0, 1}.
    src = tfrecord_path(rdir, "sources_train.part0000").replace(
        ".tfrecord", ".csv")
    for r in read_sources(src):
        assert r in (0, 1)


def test_salvage_shard_cap_caps_kept_records(tmp_path):
    rdir = str(tmp_path)
    sim, fwd = _sim_fwd()
    rp._generate_convolve_range(sim, fwd, rdir, "train", 0, 4, 0, seed=[1, 1, 0])
    kept, idx = rp._salvage_shard(rdir, "train", 0, cap=1)
    assert kept == 1 and idx == [0]
    assert rp._count_tfrecords(tfrecord_path(rdir, "clean_train.part0000")) == 1


def test_salvage_subset_then_resume_and_merge_to_complete(tmp_path):
    rdir = str(tmp_path)
    sim, fwd = _sim_fwd()
    n = 4
    # Killed-run state: shard 0 done over [0,2); shard 1 over [2,4) but its
    # dirty view lost its last record to a mid-write kill.
    rp._generate_convolve_range(sim, fwd, rdir, "train", 0, 2, 0, seed=[1, 1, 0])
    rp._generate_convolve_range(sim, fwd, rdir, "train", 2, 2, 1, seed=[1, 1, 1])
    _kill_part_tail(tfrecord_path(rdir, "dirty_train.part0001"), 6)

    done, used, next_sid = rp._salvage_subset(rdir, "train", n)
    assert done == 3                       # 2 (shard0) + 1 (shard1 salvaged)
    assert sorted(used) == [0, 1, 2]
    assert next_sid == 2                   # new shards start above salvaged ids

    # Resume: generate the 1 missing pair as a new shard with a fresh index
    # above every salvaged one (exactly what the parallel step does).
    base_idx = max(used) + 1
    rp._generate_convolve_range(sim, fwd, rdir, "train", base_idx, n - done,
                                next_sid, seed=[1, 1, next_sid])
    rp._merge_subset(rdir, "train")

    kinds = ("clean", "hr", "dirty", "sources")
    assert rp._subset_complete(rdir, "train", kinds, n) is True
    for kind in ("clean", "hr", "dirty"):
        assert rp._count_tfrecords(tfrecord_path(rdir, f"{kind}_train")) == n
    # Stored indices are unique across the merge (no salvaged/new collision).
    merged_idx = rp._part_indices(tfrecord_path(rdir, "clean_train"))
    assert len(merged_idx) == n and len(set(merged_idx)) == n
    # Parts are cleaned up once the merge succeeds.
    assert rp._existing_part_sids(rdir, "train") == []


def test_force_path_discards_salvageable_parts(tmp_path):
    """--force wipes prior parts rather than resuming them."""
    rdir = str(tmp_path)
    sim, fwd = _sim_fwd()
    rp._generate_convolve_range(sim, fwd, rdir, "train", 0, 2, 0, seed=[1, 1, 0])
    assert rp._existing_part_sids(rdir, "train") == [0]
    rp._cleanup_parts(rdir, "train")       # what the --force branch calls
    assert rp._existing_part_sids(rdir, "train") == []
