"""``scan_cutouts`` must find cutouts for *all* star IDs.

Regression: the on-disk scan globbed ``star_[0-9][0-9][0-9][0-9]_*.fits`` —
exactly four digits — but filenames come from ``f"star_{id:04d}_{size}.fits"``,
where ``:04d`` is only a *minimum* width. So every star with id ≥ 10000 rendered
a 5-digit name the glob silently skipped: those cutouts were never recognised on
disk and got re-downloaded on every run (and could never be PSF-extracted).
"""

from __future__ import annotations

import os

import numpy as np
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.catalog.downloader import (
    DownloadConfig, positions_match, scan_cutouts,
)


def _write_cutout(path: str, ra: float, dec: float, size: int = 16) -> None:
    hdu = fits.PrimaryHDU(np.ones((size, size), dtype=np.float32))
    hdu.header["CRVAL1"] = ra
    hdu.header["CRVAL2"] = dec
    hdu.writeto(path, overwrite=True)


def _cutout_dir(tmp_path) -> str:
    d = Config.cutout_dir_for_band(
        "VIS", root=os.path.join(str(tmp_path), Config.CUTOUTS_SUBDIR))
    os.makedirs(d, exist_ok=True)
    return d


def test_finds_four_and_five_digit_ids(tmp_path):
    cutout_dir = _cutout_dir(tmp_path)
    # One 4-digit id, one 5-digit id (the previously-skipped case), one 6-digit.
    ids = [42, 10002, 123456]
    for sid in ids:
        _write_cutout(
            os.path.join(cutout_dir, f"star_{sid:04d}_16.fits"),
            150.0 + sid * 1e-5, 2.0 + sid * 1e-5)

    existing, corrupted = scan_cutouts(cutout_dir)
    assert set(existing) == set(ids), (
        f"missed ids {set(ids) - set(existing)} — 5-digit glob regression")
    assert corrupted == []
    for sid in ids:
        (ra, dec, size, path), = existing[sid]
        assert size == 16
        assert f"star_{sid:04d}_16.fits" in path


def test_five_digit_cutout_is_matched_not_redownloaded(tmp_path):
    cutout_dir = _cutout_dir(tmp_path)
    sid, ra, dec = 10005, 187.5, -3.25
    _write_cutout(os.path.join(cutout_dir, f"star_{sid:04d}_16.fits"), ra, dec)

    existing, _ = scan_cutouts(cutout_dir)
    assert sid in existing
    cfg = DownloadConfig.for_band("VIS", cutout_size=16)
    fra, fdec, fsize, _ = existing[sid][0]
    assert positions_match(fra, fdec, ra, dec, cfg.position_tolerance)
