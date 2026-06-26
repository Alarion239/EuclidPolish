"""run_sr_inference stamps the sr_{subset} records with (model, input) lineage."""

from __future__ import annotations

import importlib.util
import os

import numpy as np

from euclid_polish.config import Config
from euclid_polish.provenance.checkpoint import write_checkpoint_provenance
from euclid_polish.provenance.ids import ProvId
from euclid_polish.provenance.records import Stamp
from euclid_polish.provenance.store import ProvStore
from euclid_polish.image.tfio import (
    read_multiband_skyimages, tfrecord_path, write_multiband_skyimages,
)
from euclid_polish.image import Image

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_sr_infer():
    path = os.path.join(_HERE, "scripts", "lensfinder_sr_infer.py")
    spec = importlib.util.spec_from_file_location("sr_infer_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_sr_inference_stamps_records(tmp_path, monkeypatch):
    rdir = str(tmp_path)
    # A stamped dirty record on disk.
    dirty = Image(
        data=np.zeros((8, 8, 4), np.float32), pixel_scale_arcsec=0.10,
        band_names=Config.LR_INPUT_BAND_NAMES, is_clean=False, index=0,
        subset="train",
    )
    dirty.stamp = Stamp(id=ProvId("4b1e7a90"), produced_by=ProvId("7f3a9c21"),
                        schema_version=3, subset="train")
    write_multiband_skyimages([dirty], "dirty_train", records_dir=rdir)

    # A checkpoint that knows its model id.
    ckpt = str(tmp_path / "ckpt")
    write_checkpoint_provenance(ckpt, Stamp(id=ProvId("2f9c81aa")))

    mod = _load_sr_infer()
    # Isolate the store to tmp (don't touch the repo data dir).
    monkeypatch.setattr(
        mod, "default_store",
        lambda: ProvStore(str(tmp_path / "store"), data_roots=[rdir]))

    def sr_fn(lr_cube):
        h, w = lr_cube.shape[:2]
        return np.zeros((h * 2, w * 2, 4), np.float32)

    n = mod.run_sr_inference(rdir, "train", sr_fn, checkpoint=ckpt)
    assert n == 1

    [sr] = read_multiband_skyimages(tfrecord_path(rdir, "sr_train"), num_images=1)
    st = sr.prov_stamp()
    assert st is not None
    assert ProvId("2f9c81aa") in st.parents      # the producing model
    assert ProvId("4b1e7a90") in st.parents      # the input dirty file
    assert st.produced_by is not None            # an InferenceRun
