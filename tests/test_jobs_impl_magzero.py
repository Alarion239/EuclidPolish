"""The per-object real-cutout reconstruction converts archive ADU/s to
electrons with the header's own MAGZERO, and refuses a cutout without one
(a fallback to the band's simulator zeropoint would make the factor exactly
one and pass ADU/s pixels off as electrons, ~7.6e3× too faint in VIS)."""

from __future__ import annotations

import os

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.photometry import adu_per_s_to_electrons_factor
from euclid_polish.web.helpers import jobs_impl

H = W = 8


def _fake_sr(calls):
    def fake_sr_from_model(model, lr_cube):
        calls.append(lr_cube)
        sr = np.ones((2 * H, 2 * W, lr_cube.shape[-1]), np.float32)
        return lr_cube[..., 0], sr, None
    return fake_sr_from_model


def _fetch(header_cards):
    def fake_fetch(*, ra, dec, band_name, output_file, cutout_size_vis_pixels):
        hdr = fits.Header()
        for key, value in header_cards.items():
            hdr[key] = value
        fits.PrimaryHDU(np.ones((H, W), np.float32), header=hdr).writeto(output_file, overwrite=True)
        return True, None
    return fake_fetch


def _run(out_dir):
    return jobs_impl.reconstruct_cutout_at(
        model=None, ra=1.0, dec=2.0, cutout_size_vis_pixels=H,
        out_dir=str(out_dir), render=False, checkpoint_dir="ckpt-x")


def test_a_downloaded_cutout_without_magzero_is_refused(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setattr(jobs_impl, "fetch_cutout_at", _fetch({}))
    monkeypatch.setattr(jobs_impl, "sr_from_model", _fake_sr(calls))
    with pytest.raises(ValueError, match="MAGZERO"):
        _run(tmp_path / "obj")
    assert calls == []                                   # the model never sees ADU/s


def test_a_cached_cutout_with_a_non_finite_magzero_is_refused(tmp_path, monkeypatch):
    out_dir = tmp_path / "obj"
    os.makedirs(out_dir)
    for band_name in Config.LR_INPUT_BAND_NAMES:
        hdr = fits.Header()
        hdr["MAGZERO"] = "nan"
        fits.PrimaryHDU(np.ones((H, W), np.float32), header=hdr).writeto(
            os.path.join(out_dir, f"{band_name}.fits"))
    calls: list = []
    monkeypatch.setattr(jobs_impl, "fetch_cutout_at",
                        lambda *a, **k: pytest.fail("cached band FITS should be reused"))
    monkeypatch.setattr(jobs_impl, "sr_from_model", _fake_sr(calls))
    with pytest.raises(ValueError, match="no finite MAGZERO"):
        _run(out_dir)
    assert calls == []


def test_the_header_magzero_sets_the_electron_scale(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setattr(jobs_impl, "fetch_cutout_at", _fetch({"MAGZERO": 24.6}))
    monkeypatch.setattr(jobs_impl, "sr_from_model", _fake_sr(calls))
    res = _run(tmp_path / "obj")
    for band_name in Config.LR_INPUT_BAND_NAMES:
        factor = adu_per_s_to_electrons_factor(24.6, Config.get_band(band_name))
        assert factor > 1.0
        assert res["bands"][band_name]["magzero"] == pytest.approx(24.6)
        assert res["bands"][band_name]["adu_to_e"] == pytest.approx(factor)
    np.testing.assert_allclose(calls[0][..., 0],
                               adu_per_s_to_electrons_factor(24.6, Config.get_band("VIS")), rtol=1e-6)
