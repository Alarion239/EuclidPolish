"""HR.fits is persisted 4-band (heavy deps monkeypatched; no torch/TF)."""

from __future__ import annotations

import numpy as np
from astropy.io import fits

from euclid_polish.eval import synthetic_runner as sr
from euclid_polish.eval.catalog_runner import EVAL_HR_SIZE, EVAL_LR_SIZE


class _Img:
    def __init__(self, index, data):
        self.index = index
        self.data = data


def test_hr_fits_written_four_band(tmp_path, monkeypatch):
    # dirty_* records are LR half-grid (64²×4); hr_* are HR-grid (128²×4) — large
    # enough to crop the canonical 53² LR / 106² SR·HR stamp centered at (64,64).
    def fake_read(path, num_images=0):
        if "dirty" in str(path):
            return [_Img(0, np.zeros((64, 64, 4), np.float32))]
        return [_Img(0, np.ones((128, 128, 4), np.float32))]

    monkeypatch.setattr("euclid_polish.image.tfio.read_multiband_skyimages",
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

    base = f"{out_dir}/syn-lens_0000"
    # Canonical geometry: LR EVAL_LR_SIZE², SR/HR EVAL_HR_SIZE², all 4-band.
    with fits.open(f"{base}/HR.fits") as hdul:
        assert np.asarray(hdul[0].data).shape == (4, EVAL_HR_SIZE, EVAL_HR_SIZE)
        assert "VIS" in hdul[0].header.get("BANDS", "")
    with fits.open(f"{base}/SR.fits") as hdul:
        assert np.asarray(hdul[0].data).shape == (4, EVAL_HR_SIZE, EVAL_HR_SIZE)
    with fits.open(f"{base}/original_stack.fits") as hdul:
        assert np.asarray(hdul[0].data).shape == (4, EVAL_LR_SIZE, EVAL_LR_SIZE)
