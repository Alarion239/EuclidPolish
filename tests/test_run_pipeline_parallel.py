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

from euclid_polish.config import Config
from euclid_polish.euclid.psf_library import load_all_band_psfs
from euclid_polish.sky.multiband_forward import (
    MultiBandForward, MultiBandForwardConfig,
)
from euclid_polish.sky.multiband_generator import (
    MultiBandGeneratorConfig, MultiBandSimulator,
)
from euclid_polish.sky.tfrecord import tfrecord_path
from euclid_polish.training.data_multiband import MultiBandEuclidDataset
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
    sim = MultiBandSimulator(
        cat, MultiBandGeneratorConfig(image_size=96,
                                      pixel_scale=Config.DEFAULT_PIXEL_SCALE),
    )
    psfs = load_all_band_psfs(psf_dir="/nonexistent_dir_for_test")  # Gaussian
    fwd = MultiBandForward(psfs_by_band=psfs,
                           config=MultiBandForwardConfig(add_noise=True))
    return sim, fwd


def _count(path: str) -> int:
    return sum(1 for _ in tf.data.TFRecordDataset(path))


def test_shard_bounds_partition_is_contiguous():
    b = rp._shard_bounds(10, 3)
    assert b[0][0] == 0 and b[-1][1] == 10
    for (s0, e0), (s1, e1) in zip(b, b[1:]):
        assert e0 == s1                     # no gaps / overlaps
    assert sum(e - s for s, e in b) == 10   # full coverage


def test_concat_tfrecords_skips_missing(tmp_path):
    out = str(tmp_path / "out.tfrecord")
    rp._concat_tfrecords([str(tmp_path / "nope.tfrecord")], out)
    assert os.path.exists(out) and os.path.getsize(out) == 0


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

    # The dataset pairs (LR, HR) by position without zip/shape errors — i.e.
    # the merged clean/hr/dirty stayed aligned across the shard boundary.
    ds = MultiBandEuclidDataset(
        subset="train", records_dir=rdir, scale=2, hr_patch_size=16,
    ).dataset(batch_size=1, random_transform=False, repeat_count=1)
    pairs = list(ds)
    assert len(pairs) == 4
    lr, hr = pairs[0]
    assert lr.shape[-1] == Config.NUM_LR_CHANNELS
    assert hr.shape[-1] == Config.NUM_HR_CHANNELS
