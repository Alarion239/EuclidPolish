"""build-stamps reads dirty_/sr_/hr_ (no model) and writes 4-band Lupton RGB
stamps + catalog. Tiny synthetic records; no SR model, no torch."""

from __future__ import annotations

import importlib.util
import os

import numpy as np
from PIL import Image

from euclid_polish.config import Config
from euclid_polish.sky.generation.source_catalog import SOURCE_COLS
from euclid_polish.image.tfio import open_writer
from euclid_polish.image import Image as MultiBandSkyImage


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
    with open_writer(name, records_dir=rdir) as w:
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


def _write_multi(rdir, name, shape, bands, n):
    with open_writer(name, records_dir=rdir) as w:
        for i in range(n):
            data = np.full(shape, float(i + 1), np.float32)
            w.write(MultiBandSkyImage(
                data=data, pixel_scale_arcsec=0.05, band_names=bands,
                is_clean=("dirty" not in name), index=i, subset="train"), index=i)


def test_build_stamps_streams_multiple_fields(tmp_path):
    # Several fields, written in index order to dirty_/sr_/hr_ — exercises the
    # lockstep zip stream (the memory-safe path), not the old all-in-RAM load.
    rdir = str(tmp_path / "rec")
    os.makedirs(rdir, exist_ok=True)
    n = 3
    _write_multi(rdir, "dirty_train", (64, 64, 4), Config.LR_INPUT_BAND_NAMES, n)
    _write_multi(rdir, "sr_train", (128, 128, 4), Config.HR_TARGET_BAND_NAMES, n)
    _write_multi(rdir, "hr_train", (128, 128, 4), Config.HR_TARGET_BAND_NAMES, n)
    with open(os.path.join(rdir, "sources_train.csv"), "w", newline="") as f:
        f.write(",".join(SOURCE_COLS) + "\n")
        for fi in range(n):                       # one lens + one galaxy per field
            for typ, x, y, te in (("lens", 64.0, 64.0, 1.0), ("galaxy", 70.0, 60.0, "")):
                row = {"field_index": fi, "type": typ, "x_pix": x, "y_pix": y,
                       "theta_E_arcsec": te, "flux_vis_e": 300}
                f.write(",".join(str(row.get(c, "")) for c in SOURCE_COLS) + "\n")
    out = str(tmp_path / "stamps")

    rc = bs.main(["--records-dir", rdir, "--subset", "train", "--out-dir", out,
                  "--stamp-m", "106"])
    assert rc == 0
    rows = open(os.path.join(out, "catalog.csv")).read().strip().splitlines()
    # n fields × (1 lens + 1 galaxy) × 3 recons + header.
    assert len(rows) == 1 + n * 2 * 3
    assert os.path.exists(os.path.join(out, "train", "hr", "00002_lens_0.png"))


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
