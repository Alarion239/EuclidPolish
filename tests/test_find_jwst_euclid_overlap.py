"""Pure-function checks for the JWST–Euclid overlap discovery script."""

from __future__ import annotations

import csv

import pytest

from scripts.find_jwst_euclid_overlap import (
    _distance_deg,
    _is_direct_imaging,
    _polygon_from_s_region,
    _write_csv,
)


def test_direct_imaging_filter_keeps_nircam_and_excludes_spectroscopy():
    assert _is_direct_imaging({"instrument_name": "NIRCAM/IMAGE"})
    assert _is_direct_imaging({"instrument_name": "MIRI/IMAGE"})
    assert not _is_direct_imaging({"instrument_name": "NIRSPEC/IFU"})
    assert not _is_direct_imaging({"instrument_name": "NIRCAM/WFSS"})


def test_polygon_parser_accepts_mast_stc_s_polygon():
    polygon = _polygon_from_s_region("POLYGON ICRS 10 20 11 20 11 21 10 21")
    assert polygon == [(10.0, 20.0), (11.0, 20.0), (11.0, 21.0), (10.0, 21.0)]
    esa_polygon = _polygon_from_s_region("Polygon 10 20 11 20 11 21 10 21")
    assert esa_polygon == polygon
    assert _polygon_from_s_region("CIRCLE ICRS 10 20 1") is None


def test_distance_handles_ra_wrap():
    distance = _distance_deg(
        {"ra": 359.5, "dec": 0.0},
        {"target_ra": 0.5, "target_dec": 0.0},
    )
    assert distance == pytest.approx(1.0)


def test_write_csv_has_stable_schema(tmp_path):
    path = tmp_path / "overlap.csv"
    _write_csv(path, [{"euclid_tile_index": "123", "jwst_archive": "mast"}])
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["euclid_tile_index"] == "123"
    assert rows[0]["jwst_archive"] == "mast"
    assert "jwst_s_region" in rows[0]
