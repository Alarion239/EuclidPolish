"""The cutout integrity pass (``euclid.cutout_integrity``).

Opening every ``star_<id>_<size>.fits`` and (re)deriving the catalog's
per-(band, size) validity, so un-openable cutouts are flagged corrupted and
downstream can trust "valid in all 4 bands".
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.euclid.catalog import StarCatalog
from euclid_polish.euclid.cutout_integrity import (
    cutout_openable, rebuild_catalog_from_cutouts, validate_all_cutouts,
)

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


def test_cutout_openable(tmp_path):
    good = tmp_path / "good.fits"
    _write_good(str(good))
    assert cutout_openable(str(good))
    (tmp_path / "bad.fits").write_bytes(b"not a fits file")
    assert not cutout_openable(str(tmp_path / "bad.fits"))
    assert not cutout_openable(str(tmp_path / "missing.fits"))


def test_validate_flags_openable_and_corrupt(tmp_path):
    root = tmp_path / "euclid_stars"
    cutouts = str(root / "cutouts")
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

    cat = StarCatalog(str(root))
    cat.save({"stars": [{"id": i, "ra": 150.0 + i * 1e-3, "dec": 2.0 + i * 1e-3}
                        for i in range(4)],
              "next_id": 4})

    summary = validate_all_cutouts(cat, cat.load(), band_names)

    assert summary["checked"] == 16          # 4 stars × 4 bands
    assert summary["unopenable"] == 1        # the garbage VIS file
    assert summary["valid_all_bands"] == 3   # stars 0, 1, 3 (not 2)
    assert summary["n_bands"] == len(band_names)

    # Flags persisted to the CSV: star 2 lost VIS; the others keep all bands.
    by_id = {int(s["id"]): s for s in cat.load()["stars"]}
    assert set(StarCatalog.valid_bands(by_id[0])) >= set(band_names)
    assert "VIS" not in StarCatalog.valid_bands(by_id[2])
    assert set(StarCatalog.valid_bands(by_id[2])) == set(band_names) - {"VIS"}


# ---------------------------------------------------------------------------
# Recovery: rebuild the catalog from cutouts after a corrupt/truncated stars.csv
# ---------------------------------------------------------------------------

def test_rebuild_recovers_missing_stars(tmp_path):
    root = tmp_path / "euclid_stars"
    _seed_cutouts(str(root / "cutouts"), n=6)             # 6 stars on disk
    band_names = [b.name for b in Config.BANDS]
    cat = StarCatalog(str(root))
    # A surviving PARTIAL catalog (the OOM symptom): only ids 0,1 remain,
    # carrying magnitude metadata; ids 2–5 are orphaned cutouts.
    cat.save({"stars": [{"id": 0, "ra": 150.0, "dec": 2.0, "magnitude": 18.5},
                        {"id": 1, "ra": 150.01, "dec": 2.01, "magnitude": 19.0}],
              "next_id": 2})

    s = rebuild_catalog_from_cutouts(cat)
    assert s["ids_on_disk"] == 6 and s["recovered"] == 4
    assert s["catalog_after"] == 6 and s["missing_radec"] == 0

    by_id = {int(st["id"]): st for st in cat.load()["stars"]}
    assert set(by_id) == {0, 1, 2, 3, 4, 5}
    # Recovered star: ra/dec from the FITS WCS header, valid in every band.
    assert by_id[3]["ra"] == pytest.approx(150.03)
    assert by_id[3]["dec"] == pytest.approx(2.03)
    assert set(StarCatalog.valid_bands(by_id[3])) == set(band_names)
    # Surviving star kept its magnitude.
    assert by_id[0]["magnitude"] == pytest.approx(18.5)
    assert cat.load()["next_id"] == 6


def test_rebuild_dry_run_writes_nothing(tmp_path):
    root = tmp_path / "euclid_stars"
    _seed_cutouts(str(root / "cutouts"), n=3)
    cat = StarCatalog(str(root))
    cat.save({"stars": [{"id": 0, "ra": 150.0, "dec": 2.0}], "next_id": 1})
    s = rebuild_catalog_from_cutouts(cat, dry_run=True)
    assert s["recovered"] == 2 and s["dry_run"] is True
    assert len(cat.load()["stars"]) == 1                   # unchanged on disk


def test_save_is_atomic_and_keeps_backup(tmp_path):
    import pandas as pd
    cat = StarCatalog(str(tmp_path / "euclid_stars"))
    cat.save({"stars": [{"id": 0, "ra": 1.0, "dec": 2.0}], "next_id": 1})
    assert os.path.isfile(cat.catalog_path)
    assert not os.path.exists(cat.catalog_path + ".tmp")   # temp cleaned up
    cat.save({"stars": [{"id": 0, "ra": 1.0, "dec": 2.0},
                        {"id": 1, "ra": 3.0, "dec": 4.0}], "next_id": 2})
    assert os.path.isfile(cat.catalog_path + ".bak")       # prior version kept
    assert len(pd.read_csv(cat.catalog_path + ".bak")) == 1
    assert len(cat.load()["stars"]) == 2


def test_load_backs_up_corrupt_nonempty_catalog(tmp_path):
    cat = StarCatalog(str(tmp_path / "euclid_stars"))
    os.makedirs(cat.output_dir, exist_ok=True)
    # A non-empty but column-less file (the realistic OOM-truncation: a
    # non-atomic write killed before any header/rows reached disk).
    with open(cat.catalog_path, "wb") as fh:
        fh.write(b"\n   \n\n")
    out = cat.load()
    assert out == {"stars": [], "next_id": 0}              # safe empty fallback
    assert os.path.isfile(cat.catalog_path + ".corrupt")   # bytes preserved
