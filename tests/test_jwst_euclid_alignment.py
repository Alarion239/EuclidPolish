from __future__ import annotations

import json

import numpy as np

from euclid_polish.config import Config
from euclid_polish.web.helpers import jwst_euclid


def test_field_id_is_stable_and_path_safe():
    identifier = jwst_euclid.field_id("ESA", "123/456", "jw-obs:1", 30.0)
    assert identifier == "ESA-123-456-jw-obs-1-s30"
    assert "/" not in identifier
    assert " " not in identifier


def test_euclid_product_path_joins_directory_and_filename():
    assert jwst_euclid.euclid_product_path({
        "file_path": "/archive/VIS",
        "file_name": "tile.fits",
    }) == "/archive/VIS/tile.fits"


def test_field_coordinates_prefer_jwst_footprint_center():
    assert jwst_euclid.field_coordinates({
        "jwst_ra_deg": "58.91",
        "jwst_dec_deg": "-46.29",
        "euclid_ra_deg": "59.02",
        "euclid_dec_deg": "-46.50",
    }) == (58.91, -46.29)


def test_aligned_header_removes_invalid_extension_cards(tmp_path):
    from astropy.io import fits

    header = fits.Header()
    header["EXTNAME"] = 1
    header["EXTVER"] = 1
    safe = jwst_euclid._aligned_primary_header(header, "jwst_i2d.fits")
    assert "EXTNAME" not in safe
    assert safe["ALIGN"] == "JWST-EUCLID"
    fits.PrimaryHDU(data=np.zeros((2, 2), dtype=np.float32), header=safe).writeto(
        tmp_path / "aligned.fits",
    )
    assert jwst_euclid.euclid_product_path({
        "file_path": "/archive/VIS/tile.fits",
        "file_name": "tile.fits",
    }) == "/archive/VIS/tile.fits"


def test_jwst_product_selection_prefers_resampled_image():
    rows = [
        {"filename": "jw0001_rate.fits"},
        {"filename": "jw0001_i2d.fits"},
        {"filename": "jw0001_uncal.fits"},
    ]
    assert jwst_euclid._choose_jwst_product(rows) == "jw0001_i2d.fits"


def test_readable_fits_rejects_empty_archive_placeholder(tmp_path):
    from astropy.io import fits

    empty = tmp_path / "empty.fits"
    empty.touch()
    valid = tmp_path / "valid.fits"
    fits.PrimaryHDU(data=np.zeros((2, 2), dtype=np.float32)).writeto(valid)
    assert not jwst_euclid._is_readable_fits(empty)
    assert jwst_euclid._is_readable_fits(valid)


def test_euclid_download_recovers_valid_file_from_placeholder(tmp_path):
    from astropy.io import fits

    extracted = tmp_path / "extracted.fits"
    fits.PrimaryHDU(data=np.ones((2, 2), dtype=np.float32)).writeto(extracted)
    destination = tmp_path / "requested.fits"

    class FakeEuclid:
        @staticmethod
        def get_cutout(**kwargs):
            destination.touch()
            return [str(extracted)]

    jwst_euclid._download_euclid_cutout(
        FakeEuclid,
        file_path="archive/file",
        tile_index="T123",
        coordinate=None,
        radius=None,
        destination=destination,
    )
    assert jwst_euclid._is_readable_fits(destination)


def test_align_to_target_preserves_an_identity_grid():
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [2.0, 2.0]
    wcs.wcs.crval = [10.0, 20.0]
    wcs.wcs.cdelt = [-0.001, 0.001]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    aligned = jwst_euclid.align_to_target(data, wcs, wcs, data.shape)
    np.testing.assert_allclose(aligned[1:-1, 1:-1], data[1:-1, 1:-1], atol=1e-4)


def test_overlap_rows_reads_cached_csv_and_marks_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path / "data"))
    root = jwst_euclid.overlap_root()
    root.mkdir(parents=True)
    (root / "esa_partial.csv").write_text(
        "euclid_tile_index,euclid_ra_deg,euclid_dec_deg,jwst_archive,jwst_observation_id,jwst_target_name\n"
        "T123,10,20,esa,jwobs,Target\n",
        encoding="utf-8",
    )
    rows, status = jwst_euclid.overlap_rows()
    assert status["partial"] is False
    assert rows[0]["field_id"] == jwst_euclid.field_id("esa", "T123", "jwobs", 30.0)
    assert rows[0]["available"] is False

    pair = jwst_euclid.pair_root() / rows[0]["field_id"]
    pair.mkdir(parents=True)
    (pair / "manifest.json").write_text(json.dumps({"field_id": rows[0]["field_id"]}), encoding="utf-8")
    assert jwst_euclid.overlap_rows()[0][0]["available"] is False


def test_scan_euclid_coverage_caches_unique_field_centers(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path / "data"))
    root = jwst_euclid.overlap_root()
    root.mkdir(parents=True)
    (root / "esa_partial.csv").write_text(
        "euclid_tile_index,euclid_ra_deg,euclid_dec_deg,jwst_archive,jwst_observation_id,jwst_target_name,jwst_ra_deg,jwst_dec_deg\n"
        "T123,10,20,esa,jwobs-1,Covered,10.0,20.0\n"
        "T123,10,20,esa,jwobs-2,Same center,10.0,20.0\n"
        "T124,11,21,esa,jwobs-3,Uncovered,11.0,21.0\n",
        encoding="utf-8",
    )

    calls = []
    probes = []

    def fake_coverage(ra, dec, *, strict=False):
        calls.append((ra, dec, strict))
        return [{"tile_index": "VIS-T123", "file_name": "vis.fits", "file_path": "/archive"}] if ra == 10.0 else []

    monkeypatch.setattr(jwst_euclid, "euclid_tiles_covering", fake_coverage)
    def fake_probe(client, tiles, **kwargs):
        probes.append(list(tiles))
        return (tiles[0], 0, []) if tiles else (None, 1, [])

    monkeypatch.setattr(jwst_euclid, "_probe_euclid_tiles", fake_probe)
    summary = jwst_euclid.scan_euclid_coverage()

    assert summary["unique_count"] == 2
    assert summary["covered_count"] == 1
    assert summary["not_covered_count"] == 1
    assert len(calls) == 2
    assert len(probes) == 1
    rows, _ = jwst_euclid.overlap_rows()
    assert [row["euclid_coverage_status"] for row in rows] == ["covered", "covered", "not_covered"]

    jwst_euclid.scan_euclid_coverage()
    assert len(calls) == 2
