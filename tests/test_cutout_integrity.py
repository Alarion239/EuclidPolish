"""The cutout integrity pass (``euclid_polish.catalog.cutout_integrity``).

Opening every ``star_<id>_<size>.fits`` and (re)deriving the catalog's
per-(band, size) validity, so un-openable cutouts are flagged corrupted and
downstream can trust "valid in all 4 bands".
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.catalog.catalog_object import CatalogObject
from euclid_polish.catalog.cutout_integrity import (
    cutout_openable,
    purge_incomplete_cutouts,
    rebuild_catalog_from_cutouts,
    validate_all_cutouts,
)
from euclid_polish.config import Config

_SIZE = 16


def _write_good(path: str) -> None:
    fits.PrimaryHDU(np.ones((_SIZE, _SIZE), dtype=np.float32)).writeto(
        path, overwrite=True)


def _write_star(path: str, ra: float, dec: float) -> None:
    hdu = fits.PrimaryHDU(np.ones((_SIZE, _SIZE), dtype=np.float32))
    hdu.header["CRVAL1"] = ra
    hdu.header["CRVAL2"] = dec
    hdu.writeto(path, overwrite=True)


def _seed_cutouts(cutouts: str, n: int) -> None:
    for bn in [b.name for b in Config.BANDS]:
        d = Config.cutout_dir_for_band(bn, root=cutouts)
        os.makedirs(d, exist_ok=True)
        for sid in range(n):
            _write_star(os.path.join(d, f"star_{sid:04d}_{_SIZE}.fits"),
                        150.0 + sid * 0.01, 2.0 + sid * 0.01)


def _save_catalog(output_dir, objects):
    CatalogObject.write(objects, os.path.join(output_dir, Config.CATALOG_FILE))


def _load_by_id(output_dir):
    objs = CatalogObject.read(os.path.join(output_dir, Config.CATALOG_FILE))
    return {int(o.id): o for o in objs if o.id is not None}


def test_cutout_openable(tmp_path):
    good = tmp_path / "good.fits"
    _write_good(str(good))
    assert cutout_openable(str(good))
    (tmp_path / "bad.fits").write_bytes(b"not a fits file")
    assert not cutout_openable(str(tmp_path / "bad.fits"))
    assert not cutout_openable(str(tmp_path / "missing.fits"))


def test_validate_flags_openable_and_corrupt(tmp_path):
    root = str(tmp_path / "euclid_stars")
    cutouts = os.path.join(root, "cutouts")
    band_names = [b.name for b in Config.BANDS]

    # 4 stars × 4 bands of good cutouts.
    for bn in band_names:
        d = Config.cutout_dir_for_band(bn, root=cutouts)
        os.makedirs(d, exist_ok=True)
        for sid in range(4):
            _write_good(os.path.join(d, f"star_{sid:04d}_{_SIZE}.fits"))
    # Corrupt star 2's VIS cutout (garbage bytes, not a FITS).
    vis_dir = Config.cutout_dir_for_band("VIS", root=cutouts)
    (open(os.path.join(vis_dir, f"star_0002_{_SIZE}.fits"), "wb")
     .write(b"truncated garbage, not openable"))

    _save_catalog(root, [CatalogObject(ra=150.0 + i * 1e-3, dec=2.0 + i * 1e-3,
                                       id=i) for i in range(4)])

    summary = validate_all_cutouts(root, band_names)

    assert summary["checked"] == 16          # 4 stars × 4 bands
    assert summary["unopenable"] == 1        # the garbage VIS file
    assert summary["valid_all_bands"] == 3   # stars 0, 1, 3 (not 2)
    assert summary["n_bands"] == len(band_names)

    by_id = _load_by_id(root)
    assert set(by_id[0].valid_bands()) >= set(band_names)
    assert "VIS" not in by_id[2].valid_bands()
    assert set(by_id[2].valid_bands()) == set(band_names) - {"VIS"}


# ---------------------------------------------------------------------------
# Cleanup: delete cutouts of any star not valid in all bands
# ---------------------------------------------------------------------------

def _all_paths(cutouts: str) -> set:
    import glob as _glob
    out = set()
    for bn in [b.name for b in Config.BANDS]:
        d = Config.cutout_dir_for_band(bn, root=cutouts)
        out |= set(_glob.glob(os.path.join(d, "star_*.fits")))
    return out


def test_purge_deletes_incomplete_keeps_complete(tmp_path):
    root = str(tmp_path / "euclid_stars")
    cutouts = os.path.join(root, "cutouts")
    band_names = [b.name for b in Config.BANDS]

    # 3 stars: 0 & 2 complete in all bands; star 1 has a corrupt VIS cutout.
    for bn in band_names:
        d = Config.cutout_dir_for_band(bn, root=cutouts)
        os.makedirs(d, exist_ok=True)
        for sid in range(3):
            _write_good(os.path.join(d, f"star_{sid:04d}_{_SIZE}.fits"))
    vis_dir = Config.cutout_dir_for_band("VIS", root=cutouts)
    with open(os.path.join(vis_dir, f"star_0001_{_SIZE}.fits"), "wb") as fh:
        fh.write(b"garbage, not a fits")

    _save_catalog(root, [CatalogObject(ra=150.0 + i * 1e-3, dec=2.0 + i * 1e-3,
                                       id=i) for i in range(3)])
    validate_all_cutouts(root, band_names)   # set flags from disk

    s = purge_incomplete_cutouts(root, band_names)
    assert s["complete_stars"] == 2          # stars 0 and 2
    assert s["incomplete_stars"] == 1        # star 1
    assert s["deleted_files"] == len(band_names)   # all 4 of star 1's cutouts
    assert s["dropped_rows"] == 1

    remaining = _all_paths(cutouts)
    assert not any("star_0001_" in p for p in remaining)
    assert sum("star_0000_" in p for p in remaining) == len(band_names)
    assert sum("star_0002_" in p for p in remaining) == len(band_names)
    assert set(_load_by_id(root)) == {0, 2}


def test_purge_missing_band_is_incomplete(tmp_path):
    root = str(tmp_path / "euclid_stars")
    cutouts = os.path.join(root, "cutouts")
    band_names = [b.name for b in Config.BANDS]

    # Star 0 complete; star 1 only ever downloaded in VIS (others never arrived).
    for bn in band_names:
        d = Config.cutout_dir_for_band(bn, root=cutouts)
        os.makedirs(d, exist_ok=True)
        _write_good(os.path.join(d, f"star_0000_{_SIZE}.fits"))
    _write_good(os.path.join(Config.cutout_dir_for_band("VIS", root=cutouts),
                             f"star_0001_{_SIZE}.fits"))

    _save_catalog(root, [CatalogObject(ra=150.0, dec=2.0, id=0),
                         CatalogObject(ra=150.1, dec=2.1, id=1)])
    validate_all_cutouts(root, band_names)

    s = purge_incomplete_cutouts(root, band_names)
    assert s["complete_stars"] == 1 and s["incomplete_stars"] == 1
    assert s["deleted_files"] == 1           # star 1's lone VIS cutout
    remaining = _all_paths(cutouts)
    assert not any("star_0001_" in p for p in remaining)
    assert set(_load_by_id(root)) == {0}


def test_purge_dry_run_touches_nothing(tmp_path):
    root = str(tmp_path / "euclid_stars")
    cutouts = os.path.join(root, "cutouts")
    band_names = [b.name for b in Config.BANDS]
    for bn in band_names:
        d = Config.cutout_dir_for_band(bn, root=cutouts)
        os.makedirs(d, exist_ok=True)
        _write_good(os.path.join(d, f"star_0000_{_SIZE}.fits"))
    _write_good(os.path.join(Config.cutout_dir_for_band("VIS", root=cutouts),
                             f"star_0001_{_SIZE}.fits"))
    _save_catalog(root, [CatalogObject(ra=150.0, dec=2.0, id=0),
                         CatalogObject(ra=150.1, dec=2.1, id=1)])
    validate_all_cutouts(root, band_names)

    before = _all_paths(cutouts)
    s = purge_incomplete_cutouts(root, band_names, dry_run=True)
    assert s["dry_run"] is True and s["deleted_files"] == 1 and s["dropped_rows"] == 0
    assert _all_paths(cutouts) == before                       # nothing deleted
    assert len(_load_by_id(root)) == 2                         # catalog intact


def test_purge_keep_catalog_rows(tmp_path):
    root = str(tmp_path / "euclid_stars")
    cutouts = os.path.join(root, "cutouts")
    band_names = [b.name for b in Config.BANDS]
    for bn in band_names:
        d = Config.cutout_dir_for_band(bn, root=cutouts)
        os.makedirs(d, exist_ok=True)
        _write_good(os.path.join(d, f"star_0000_{_SIZE}.fits"))
    _write_good(os.path.join(Config.cutout_dir_for_band("VIS", root=cutouts),
                             f"star_0001_{_SIZE}.fits"))
    _save_catalog(root, [CatalogObject(ra=150.0, dec=2.0, id=0),
                         CatalogObject(ra=150.1, dec=2.1, id=1)])
    validate_all_cutouts(root, band_names)

    s = purge_incomplete_cutouts(root, band_names, drop_catalog_rows=False)
    assert s["deleted_files"] == 1 and s["dropped_rows"] == 0
    assert not any("star_0001_" in p for p in _all_paths(cutouts))   # files gone
    assert set(_load_by_id(root)) == {0, 1}                          # rows kept


# ---------------------------------------------------------------------------
# Recovery: rebuild the catalog from cutouts after a corrupt/truncated stars.csv
# ---------------------------------------------------------------------------

def test_rebuild_recovers_missing_stars(tmp_path):
    root = str(tmp_path / "euclid_stars")
    _seed_cutouts(os.path.join(root, "cutouts"), n=6)     # 6 stars on disk
    band_names = [b.name for b in Config.BANDS]
    # A surviving PARTIAL catalog (the OOM symptom): only ids 0,1 remain,
    # carrying magnitude metadata; ids 2–5 are orphaned cutouts.
    _save_catalog(root, [CatalogObject(ra=150.0, dec=2.0, id=0, magnitude=18.5),
                         CatalogObject(ra=150.01, dec=2.01, id=1, magnitude=19.0)])

    s = rebuild_catalog_from_cutouts(root)
    assert s["ids_on_disk"] == 6 and s["recovered"] == 4
    assert s["catalog_after"] == 6 and s["missing_radec"] == 0

    by_id = _load_by_id(root)
    assert set(by_id) == {0, 1, 2, 3, 4, 5}
    # Recovered star: ra/dec from the FITS WCS header, valid in every band.
    assert by_id[3].ra == pytest.approx(150.03)
    assert by_id[3].dec == pytest.approx(2.03)
    assert set(by_id[3].valid_bands()) == set(band_names)
    # Surviving star kept its magnitude.
    assert by_id[0].magnitude == pytest.approx(18.5)


def test_rebuild_dry_run_writes_nothing(tmp_path):
    root = str(tmp_path / "euclid_stars")
    _seed_cutouts(os.path.join(root, "cutouts"), n=3)
    _save_catalog(root, [CatalogObject(ra=150.0, dec=2.0, id=0)])
    s = rebuild_catalog_from_cutouts(root, dry_run=True)
    assert s["recovered"] == 2 and s["dry_run"] is True
    assert len(_load_by_id(root)) == 1                     # unchanged on disk
