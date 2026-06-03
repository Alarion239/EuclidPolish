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
    cutout_openable, validate_all_cutouts,
)

_SIZE = 16


def _write_good(path: str) -> None:
    fits.PrimaryHDU(np.ones((_SIZE, _SIZE), dtype=np.float32)).writeto(
        path, overwrite=True)


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
