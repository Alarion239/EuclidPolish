"""``scripts/show_band_psfs.py`` labels every band's FWHM, including an
empirical kernel loaded without a cached ``fwhm_arcsec``."""

from __future__ import annotations

import importlib
import os

import numpy as np

from euclid_polish.config import Config
from euclid_polish.psf import PSF

show = importlib.import_module("scripts.show_band_psfs")


def _gauss(side: int, fwhm_pix: float, fwhm_arcsec: float | None) -> PSF:
    x = np.arange(side) - side // 2
    xx, yy = np.meshgrid(x, x)
    s = fwhm_pix / 2.355
    g = np.exp(-(xx * xx + yy * yy) / (2 * s * s)).astype(np.float32)
    return PSF(data=g / g.sum(), pixel_scale=0.05, fwhm_arcsec=fwhm_arcsec)


def test_renders_an_empirical_psf_without_a_cached_fwhm(tmp_path, monkeypatch):
    names = [band.name for band in Config.BANDS]
    # VIS: an empirical ePSF read from a file whose HDU0 has no FWHM card.
    psfs = {name: _gauss(41, 4.0, None if name == "VIS" else 0.5) for name in names}
    out = tmp_path / "band_psfs.png"
    monkeypatch.setattr(show, "OUT", str(out))
    monkeypatch.setattr(show, "load_all_band_psfs", lambda **_kw: psfs)
    monkeypatch.setattr(show, "psf_inventory",
                        lambda: {name: ("x.fits" if name == "VIS" else None) for name in names})
    assert show.main() == 0
    assert os.path.getsize(out) > 0
