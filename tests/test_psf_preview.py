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


def test_psf_preview_payload_is_bounded_and_cache_first(tmp_path, monkeypatch):
    band = Config.BANDS[0]
    _write_psf(tmp_path / band.psf_fits_filename, side=8)
    monkeypatch.setattr(fits_render, "_cached_fasrc_psf_dir", lambda: str(tmp_path))

    payload = fits_render._psf_preview_payload("all", max_side=4)

    assert payload["available"] is True
    assert payload["source"] == "FASRC cache"
    assert len(payload["bands"]) == 1
    preview = payload["bands"][0]
    assert preview["name"] == band.name
    assert preview["shape"] == [8, 8]
    assert len(preview["values"]) == 4
    assert len(preview["values"][0]) == 4
    assert preview["n_psf"] == 3


def test_psf_preview_route_returns_json(monkeypatch):
    from euclid_polish.web.app import create_app

    app = create_app()
    app.config["TESTING"] = True
    monkeypatch.setattr(
        "euclid_polish.web.routes.psfs._psf_preview_payload",
        lambda band: {"available": False, "source": "FASRC cache", "bands": []},
    )

    response = app.test_client().get("/api/euclid-psf/preview?band=VIS")

    assert response.status_code == 200
    assert response.get_json() == {
        "available": False, "source": "FASRC cache", "bands": [],
    }


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
