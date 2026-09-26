from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.web.helpers import fits_render, viewer_data


def _write_psf(path, *, side=8):
    data = np.zeros((side, side), dtype=np.float32)
    data[side // 2, side // 2] = 1.0
    hdu = fits.PrimaryHDU(data)
    hdu.header["PXSCALE"] = 0.05
    hdu.header["FWHM"] = 0.2
    hdu.header["NPSF"] = 3
    hdu.writeto(path)


def test_psf_viewer_warp_is_replayable_and_flux_preserving(tmp_path, monkeypatch):
    side = 31
    yy, xx = np.indices((side, side), dtype=np.float32)
    data = np.exp(-((yy - 15) ** 2 + (xx - 15) ** 2) / (2 * 2.2 ** 2))
    data /= data.sum()
    path = tmp_path / "vis_psf.fits"
    fits.PrimaryHDU(data.astype(np.float32)).writeto(path)
    monkeypatch.setattr(viewer_data, "_psf_paths", lambda: {"VIS": str(path)})
    monkeypatch.setattr(
        viewer_data, "_psf_preview_warp_settings", lambda: (20.0, 3.0),
    )

    nominal, _ = viewer_data._psf_cube(0, "VIS", {})
    warped_a, info_a = viewer_data._psf_cube(
        0, "VIS", {"psf_warp": "1", "psf_warp_seed": "17"},
    )
    warped_b, _ = viewer_data._psf_cube(
        0, "VIS", {"psf_warp": "1", "psf_warp_seed": "17"},
    )

    assert np.array_equal(warped_a, warped_b)
    assert not np.array_equal(warped_a, nominal)
    assert float(warped_a.sum()) == pytest.approx(float(nominal.sum()), rel=1e-6)
    assert "warped alpha=" in info_a["label"]
    assert info_a["label"].encode("latin-1")
