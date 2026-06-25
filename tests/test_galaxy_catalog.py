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
