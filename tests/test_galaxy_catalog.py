"""Unit tests for the real-galaxy eval catalog builder (archive mocked)."""
import csv
import math
import os

import pytest

from euclid_polish.euclid import galaxy_catalog as gc
from euclid_polish.config import Config


def test_diam_to_area_px_matches_circle():
    # diameter 5" at 0.1"/px → radius 25 px → area = pi*25^2.
    area = gc._diam_to_area_px(5.0)
    r_px = (5.0 / 2.0) / Config.VIS_PIXEL_SCALE_ARCSEC
    assert area == pytest.approx(math.pi * r_px * r_px)


def test_galaxy_adql_has_cuts():
    q = gc.galaxy_adql(10.0, -5.0, 0.05)
    assert "catalogue.mer_catalogue" in q
    assert f"{gc._POINTLIKE_COL} = 0" in q          # extended, not a star
    assert f"{gc._SPURIOUS_COL} = 0" in q           # not an artifact
    assert f"{gc._QUALITY_COL} = 0" in q            # clean detection
    assert f"{gc._SIZE_COL} BETWEEN" in q           # size window
    assert "CIRCLE('ICRS', 10.0, -5.0, 0.05)" in q  # the cone


def test_candidates_parse_and_mag_floor():
    rows = [
        # flux 50 µJy → mag ~19.65 → kept
        {"object_id": 1, "right_ascension": 10.01, "declination": -5.0,
         "segmentation_area": 800.0, "flux_vis_psf": 50.0},
        # flux 0.01 µJy → mag ~28.9 → too faint → dropped
        {"object_id": 2, "right_ascension": 10.02, "declination": -5.0,
         "segmentation_area": 800.0, "flux_vis_psf": 0.01},
        # non-finite flux → dropped
        {"object_id": 3, "right_ascension": 10.03, "declination": -5.0,
         "segmentation_area": 800.0, "flux_vis_psf": float("nan")},
    ]
    cands = gc._candidates_from_results(rows)
    assert [c["id"] for c in cands] == ["gal_1"]
    assert cands[0]["ra"] == 10.01 and cands[0]["dec"] == -5.0


def test_candidates_handle_none_results():
    assert gc._candidates_from_results(None) == []
