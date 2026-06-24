"""SR-inference core in scripts/lensfinder_sr_infer.py: persist sr_{subset}
records (4-band) from dirty_{subset}, resumable by record-count."""

from __future__ import annotations

import importlib.util
import os

import numpy as np

from euclid_polish.config import Config
from euclid_polish.sky.tfrecord import (open_multiband_writer,
                                        read_multiband_skyimages, tfrecord_path)
from euclid_polish.sky.types import MultiBandSkyImage


def _load():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "lensfinder_sr_infer.py")
    spec = importlib.util.spec_from_file_location("lf_sr_infer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sri = _load()


def _write_dirty(rdir, subset, n, shape=(16, 16, 4)):
    with open_multiband_writer(f"dirty_{subset}", records_dir=rdir) as w:
        for i in range(n):
            data = np.full(shape, float(i + 1), np.float32)
            w.write(MultiBandSkyImage(
                data=data, pixel_scale_arcsec=0.1,
                band_names=Config.LR_INPUT_BAND_NAMES, is_clean=False,
                index=i, subset=subset), index=i)


def test_run_sr_inference_writes_4band_records(tmp_path):
    rdir = str(tmp_path)
    _write_dirty(rdir, "train", 3)
    n = sri.run_sr_inference(rdir, "train", sr_fn=lambda lr: lr)   # identity SR
    assert n == 3
    out = read_multiband_skyimages(tfrecord_path(rdir, "sr_train"), num_images=10)
    assert len(out) == 3
    assert out[0].data.shape[-1] == 4              # 4-band preserved


def test_run_sr_inference_resume_skips_complete(tmp_path, monkeypatch):
    rdir = str(tmp_path)
    _write_dirty(rdir, "train", 3)
    sri.run_sr_inference(rdir, "train", sr_fn=lambda lr: lr)

    opened = []
    real = sri.open_multiband_writer
    monkeypatch.setattr(sri, "open_multiband_writer",
                        lambda name, **kw: opened.append(name) or real(name, **kw))
    n = sri.run_sr_inference(rdir, "train", sr_fn=lambda lr: lr)   # second run
    assert n == 3 and opened == []                 # skipped, nothing rewritten

    n2 = sri.run_sr_inference(rdir, "train", sr_fn=lambda lr: lr, force=True)
    assert n2 == 3 and "sr_train" in opened        # force regenerates


def test_count_records_truncated_is_none(tmp_path):
    rdir = str(tmp_path)
    _write_dirty(rdir, "train", 2)
    p = tfrecord_path(rdir, "dirty_train")
    with open(p, "r+b") as f:
        f.truncate(os.path.getsize(p) - 4)
    assert sri._count_records(p) is None


def test_parse_args_defaults():
    args = sri._parse_args(["--records-dir", "data/x"])
    assert args.records_dir == "data/x"
    assert args.subset_all is True          # both subsets by default
    assert args.num_res_blocks == Config.DEFAULT_NUM_RES_BLOCKS
    assert args.force is False
