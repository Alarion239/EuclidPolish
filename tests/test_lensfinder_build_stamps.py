"""build-stamps reads dirty_/sr_/hr_ (no model) and writes 4-band Lupton RGB
stamps + catalog. Tiny synthetic records; no SR model, no torch."""

from __future__ import annotations

import importlib.util
import os

import numpy as np
from PIL import Image

from euclid_polish.config import Config
from euclid_polish.sky.source_catalog import SOURCE_COLS
from euclid_polish.sky.tfrecord import open_multiband_writer
from euclid_polish.sky.types import MultiBandSkyImage


def _load():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "lensfinder_build_stamps.py")
    spec = importlib.util.spec_from_file_location("lf_build", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bs = _load()


def _write_field(rdir, name, shape, bands):
    with open_multiband_writer(name, records_dir=rdir) as w:
        data = np.random.default_rng(abs(hash(name)) % 1000).random(shape).astype(
            np.float32) * 400.0
        w.write(MultiBandSkyImage(
            data=data, pixel_scale_arcsec=0.05, band_names=bands,
            is_clean=("dirty" not in name), index=0, subset="train"), index=0)


def _write_sources(rdir):
    rows = [
        {"field_index": 0, "type": "lens", "x_pix": 64.0, "y_pix": 64.0,
         "theta_E_arcsec": 1.2, "flux_vis_e": 500},
        {"field_index": 0, "type": "galaxy", "x_pix": 70.0, "y_pix": 60.0,
         "flux_vis_e": 300},
    ]
    with open(os.path.join(rdir, "sources_train.csv"), "w", newline="") as f:
        f.write(",".join(SOURCE_COLS) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in SOURCE_COLS) + "\n")


def test_build_stamps_writes_color_catalog(tmp_path):
    rdir = str(tmp_path / "rec")
    os.makedirs(rdir, exist_ok=True)
    _write_field(rdir, "dirty_train", (64, 64, 4), Config.LR_INPUT_BAND_NAMES)
    _write_field(rdir, "sr_train", (128, 128, 4), Config.HR_TARGET_BAND_NAMES)
    _write_field(rdir, "hr_train", (128, 128, 4), Config.HR_TARGET_BAND_NAMES)
    _write_sources(rdir)
    out = str(tmp_path / "stamps")

    rc = bs.main(["--records-dir", rdir, "--subset", "train", "--out-dir", out,
                  "--stamp-m", "106", "--png-size", "424"])
    assert rc == 0

    cat = os.path.join(out, "catalog.csv")
    assert os.path.exists(cat)
    text = open(cat).read().strip().splitlines()
    # 2 sources x 3 recons = 6 stamp rows + header.
    assert len(text) == 1 + 6
    # spot-check a rendered PNG is 424x424 RGB color.
    png = os.path.join(out, "train", "sr", "00000_lens_0.png")
    assert os.path.exists(png)
    with Image.open(png) as im:
        assert im.size == (424, 424) and im.mode == "RGB"
