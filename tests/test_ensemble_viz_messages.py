"""User-facing hints in ``euclid_polish/web/helpers/ensemble_viz.py`` name
console places that exist: records sync in Synthetic › Records, campaigns in
Notebook › Log."""
from __future__ import annotations

import os

import pytest

from euclid_polish.config import Config
from euclid_polish.web.helpers import ensemble_viz as ev


class _Cap:
    def tick(self, *a):
        pass

    def write(self, *a):
        pass


def _assert_records_hint(message: str) -> None:
    assert "Synthetic › Records" in message
    assert "/sky page" not in message
    assert "Include validate" not in message


@pytest.fixture
def no_records(monkeypatch):
    monkeypatch.setattr(ev, "_sky_records_local_dir", lambda: None)


def test_member_psnr_without_records_points_at_records_tab(no_records):
    with pytest.raises(RuntimeError) as excinfo:
        ev.job_member_psnr(_Cap())
    _assert_records_hint(str(excinfo.value))


def test_ensemble_evaluate_without_records_points_at_records_tab(no_records):
    with pytest.raises(RuntimeError) as excinfo:
        ev.job_ensemble_evaluate(_Cap(), num_images=1, starless=False)
    _assert_records_hint(str(excinfo.value))


def test_validate_cubes_without_records_points_at_records_tab(no_records):
    with pytest.raises(RuntimeError) as excinfo:
        ev._prepare_validate_cubes(_Cap(), starless=False, num_images=1,
                                   target_fwhm=Config.TARGET_PSF_FWHM_ARCSEC)
    _assert_records_hint(str(excinfo.value))


def test_missing_validate_split_points_at_records_sync(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "_sky_records_local_dir", lambda: str(tmp_path))
    with pytest.raises(RuntimeError) as excinfo:
        ev._prepare_validate_cubes(_Cap(), starless=False, num_images=1,
                                   target_fwhm=Config.TARGET_PSF_FWHM_ARCSEC)
    message = str(excinfo.value)
    _assert_records_hint(message)
    assert "validate" in message and "dirty_validate + hr_validate" in message


def test_archive_without_campaign_points_at_notebook(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR", str(tmp_path / "ckpt/wdsr"))
    monkeypatch.setattr(Config, "VIS_DIR", str(tmp_path / "vis"))
    monkeypatch.setattr(Config, "TRACKING_DIR", str(tmp_path / "tracking"))
    member = os.path.join(ev.ensemble_dir(), "member_03")
    os.makedirs(member)
    with open(os.path.join(member, "checkpoint"), "w") as f:
        f.write("weights")
    with pytest.raises(RuntimeError) as excinfo:
        ev.job_archive_member(_Cap(), name="member_03")
    message = str(excinfo.value)
    assert "Notebook › Log" in message
    assert "/tracking" not in message
