"""The sky-records BHR tier hint in ``euclid_polish/web/helpers/viewer_data.py``
describes what ``_sky_cube`` serves: HR blurred by the Gaussian target PSF."""
from __future__ import annotations

from euclid_polish.config import Config
from euclid_polish.image.tfio import tfrecord_path
from euclid_polish.web.helpers import sky_records
from euclid_polish.web.helpers import viewer_data as vd


def test_sky_bhr_hint_names_the_gaussian_target_psf(tmp_path, monkeypatch):
    monkeypatch.setattr(vd, "_sky_records_local_dir", lambda: str(tmp_path))
    monkeypatch.setattr(vd, "_record_count", lambda _name, _dir: 3)
    monkeypatch.setattr(sky_records, "sr_count", lambda _subset: 0)
    with open(tfrecord_path(str(tmp_path), "hr_test"), "wb"):
        pass

    tiers = {tier["key"]: tier for tier in vd._sky_meta({})["tiers"]}
    hint = tiers["bhr"]["hint"]

    # BHR is blur_target_array(HR, target FWHM): never the LR (VIS/NISP) PSF.
    assert "LR PSF" not in hint and "perfect LR" not in hint
    assert "Gaussian target PSF" in hint
    assert f"{Config.TARGET_PSF_FWHM_ARCSEC:g}″" in hint
