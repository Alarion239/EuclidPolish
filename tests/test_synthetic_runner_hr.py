"""HR.fits is persisted 4-band (heavy deps monkeypatched; no torch/TF)."""

from __future__ import annotations

import numpy as np
from astropy.io import fits

from euclid_polish.eval import synthetic_runner as sr


class _Img:
    def __init__(self, index, data):
        self.index = index
        self.data = data


def test_hr_fits_written_four_band(tmp_path, monkeypatch):
    # dirty_* records are LR half-grid (64²×4); hr_* are HR-grid (128²×4).
    def fake_read(path, num_images=0):
        if "dirty" in str(path):
            return [_Img(0, np.zeros((64, 64, 4), np.float32))]
        return [_Img(0, np.ones((128, 128, 4), np.float32))]

    monkeypatch.setattr("euclid_polish.sky.tfrecord.read_multiband_skyimages",
                        fake_read)
    monkeypatch.setattr("euclid_polish.sky.source_catalog.read_sources",
                        lambda p: {0: [{"type": "lens", "x_pix": 64.0,
                                        "y_pix": 64.0, "flux_vis_e": 1.0}]})
    monkeypatch.setattr("euclid_polish.training.inference.reconstruct",
                        lambda model, lr: (None, np.ones((128, 128, 4), np.float32)))

    out_dir = str(tmp_path / "eval")
    res = sr.run_synthetic_eval(
        out_dir, n=1, model=object(), records_dir=str(tmp_path),
        on_progress=lambda *a: None, log=lambda *a: None)
    assert res["n_ok"] == 1

    with fits.open(f"{out_dir}/syn-lens_0000/HR.fits") as hdul:
        hr = np.asarray(hdul[0].data)
        bands = hdul[0].header.get("BANDS", "")
    assert hr.shape == (4, 64, 64)          # 4-band, HR grid (m=64 here)
    assert "VIS" in bands
