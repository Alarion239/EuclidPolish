"""The WCS orientation read-out in ``scripts/inspect_cutout_orientation.py``
recovers the position angle from the header (CD/PC matrix)."""

from __future__ import annotations

import importlib

import numpy as np
from astropy.wcs import WCS

mod = importlib.import_module("scripts.inspect_cutout_orientation")


def _header(rot_deg: float, n: int = 32):
    w = WCS(naxis=2)
    w.wcs.crpix = [n / 2 + 0.5, n / 2 + 0.5]
    w.wcs.crval = [150.0, 2.0]
    s = 0.10 / 3600.0
    th = np.radians(rot_deg)
    w.wcs.cd = [[-s * np.cos(th), s * np.sin(th)],
                [s * np.sin(th),  s * np.cos(th)]]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    h = w.to_header()
    h["NAXIS1"] = n
    h["NAXIS2"] = n
    return h


def test_orientation_recovers_position_angle():
    assert abs(mod._orientation(_header(0.0))["pa_y_deg"]) < 1e-6
    assert abs(mod._orientation(_header(35.0))["pa_y_deg"] - 35.0) < 1e-3
    assert abs(mod._orientation(_header(-12.0))["pa_y_deg"] + 12.0) < 1e-3


def test_orientation_reports_pixscale_and_centre():
    o = mod._orientation(_header(0.0))
    assert abs(o["pixscale_y"] - 0.10) < 1e-4
    assert abs(o["ra"] - 150.0) < 1e-3 and abs(o["dec"] - 2.0) < 1e-3
