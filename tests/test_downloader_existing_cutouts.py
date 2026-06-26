"""``get_existing_cutouts`` must find cutouts for *all* star IDs.

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

from euclid_polish.catalog.star_catalog import StarCatalog
from euclid_polish.catalog.downloader import DownloadConfig, EuclidCutoutDownloader


def _write_cutout(path: str, ra: float, dec: float, size: int = 16) -> None:
    hdu = fits.PrimaryHDU(np.ones((size, size), dtype=np.float32))
    hdu.header["CRVAL1"] = ra
    hdu.header["CRVAL2"] = dec
    hdu.writeto(path, overwrite=True)


def _downloader(tmp_path, size_px):
    cat = StarCatalog(str(tmp_path))
    cfg = DownloadConfig.for_band("VIS", cutout_size=size_px)
    return EuclidCutoutDownloader(cat, cfg)


def test_finds_four_and_five_digit_ids(tmp_path):
    dl = _downloader(tmp_path, 16)
    # One 4-digit id, one 5-digit id (the previously-skipped case), one 6-digit.
    ids = [42, 10002, 123456]
    for sid in ids:
        _write_cutout(
            os.path.join(dl.cutout_dir, f"star_{sid:04d}_16.fits"),
            150.0 + sid * 1e-5, 2.0 + sid * 1e-5)

    existing, corrupted = dl.get_existing_cutouts()
    assert set(existing) == set(ids), (
        f"missed ids {set(ids) - set(existing)} — 5-digit glob regression")
    assert corrupted == []
    # Each maps to one (ra, dec, size, path) tuple at the right size.
    for sid in ids:
        (ra, dec, size, path), = existing[sid]
        assert size == 16
        assert f"star_{sid:04d}_16.fits" in path


def test_five_digit_cutout_is_matched_not_redownloaded(tmp_path):
    # End-to-end of the reconciliation: a 5-digit star whose cutout is on disk
    # (position-matching the catalog) must NOT end up in stars_needing_download.
    dl = _downloader(tmp_path, 16)
    sid, ra, dec = 10005, 187.5, -3.25
    _write_cutout(os.path.join(dl.cutout_dir, f"star_{sid:04d}_16.fits"), ra, dec)

    existing, _ = dl.get_existing_cutouts()
    assert sid in existing
    star = {"id": sid, "ra": ra, "dec": dec}
    fra, fdec, fsize, _ = existing[sid][0]
    assert dl.positions_match(fra, fdec, star["ra"], star["dec"])
