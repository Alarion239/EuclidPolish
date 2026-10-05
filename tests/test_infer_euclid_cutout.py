"""scripts/infer_euclid_cutout.py reconstructs through the production model.

The ensemble is the model: the script loads ``load_eval_ensemble`` (never a
single checkpoint from ./ckpt/wdsr) and writes a 4-band SR cube whose header
names the combiner and members. The archive download is faked.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

import scripts.infer_euclid_cutout as infer
from euclid_polish.config import Config

SIDE = 8


class _FakeEnsemble:
    member_labels = ["170·psnr", "171·psnr", "195·psnr"]
    run_labels = ["170·psnr", "195·psnr"]
    combiner_kind = "spatial_gate"
    n_members, n_run = 3, 2
    label = "spatial gate (convolutional, convex) over 2 of 3 STARFULL models"

    def member_arrays(self, lr):
        up = np.repeat(np.repeat(np.asarray(lr, np.float32), 2, 0), 2, 1)
        return np.stack([up, 3.0 * up])

    def combine(self, members, lr):
        return members.mean(axis=0)


def _fake_fetch(*, ra, dec, band_name, output_file, cutout_size_vis_pixels):
    header = fits.Header()
    header["MAGZERO"] = 24.5
    data = np.full((SIDE, SIDE), 1.0 + Config.LR_INPUT_BAND_NAMES.index(band_name),
                   dtype=np.float32)
    fits.PrimaryHDU(data, header=header).writeto(output_file, overwrite=True)
    return True, None


@pytest.fixture
def loads(monkeypatch):
    calls: list[tuple] = []

    def fake_load(base_dir=None, num_res_blocks=None, *, log=None, **_kw):
        calls.append((base_dir, num_res_blocks))
        return _FakeEnsemble()

    monkeypatch.setattr(infer, "fetch_cutout_at", _fake_fetch)
    monkeypatch.setattr(infer, "load_eval_ensemble", fake_load)
    return calls


def test_reconstructs_with_the_production_ensemble(loads, tmp_path):
    rc = infer.main(["--ra", "150.1", "--dec", "2.2", "--vis-pixels", str(SIDE),
                     "--ensemble-dir", "/x/ensemble", "--out-dir", str(tmp_path)])

    assert rc == 0
    assert loads == [("/x/ensemble", Config.DEFAULT_NUM_RES_BLOCKS)]
    with fits.open(tmp_path / "SR.fits") as hdul:
        sr, header = hdul[0].data, hdul[0].header
    # A 4-band cube, one plane per band (band 0 = VIS), at 2× the LR side;
    # the production combine of the two members that ran (mean of x and 3x).
    assert sr.shape == (len(Config.LR_INPUT_BAND_NAMES), 2 * SIDE, 2 * SIDE)
    with fits.open(tmp_path / "original_stack.fits") as hdul:
        stack = hdul[0].data
    assert np.allclose(sr[:, ::2, ::2], 2.0 * stack)
    assert header["BANDS"] == ",".join(Config.LR_INPUT_BAND_NAMES)
    assert header["COMBINER"] == "spatial_gate"
    assert header["MEMBERS"] == "170.psnr,171.psnr,195.psnr"
    assert header["RUNMEMB"] == "170.psnr,195.psnr"
    assert header["MODEL"].startswith("spatial gate")


def test_default_ensemble_location_and_no_checkpoint_flag(loads, tmp_path):
    assert infer.main(["--ra", "1", "--dec", "2", "--vis-pixels", str(SIDE),
                       "--out-dir", str(tmp_path)]) == 0
    assert loads == [(None, Config.DEFAULT_NUM_RES_BLOCKS)]   # registry default
    with pytest.raises(SystemExit):
        infer.parse_args(["--ra", "1", "--dec", "2", "--ckpt-dir", "ckpt/wdsr"])


def test_plain_mean_fallback_is_recorded(loads, tmp_path, monkeypatch):
    monkeypatch.setattr(_FakeEnsemble, "combiner_kind", None)
    assert infer.main(["--ra", "1", "--dec", "2", "--vis-pixels", str(SIDE),
                       "--out-dir", str(tmp_path)]) == 0
    assert fits.getheader(tmp_path / "SR.fits")["COMBINER"] == "member_mean"
