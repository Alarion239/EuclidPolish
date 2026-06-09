"""Unit tests for scripts/measure_star_saturation.py (the FASRC diagnostic)."""

from __future__ import annotations

import csv
import os

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.config import Config

from scripts.measure_star_saturation import (
    _adu_to_e_factor, _summarize_band, load_cutout_electrons,
    measure_core_saturation, scan_stars,
)

_SIZE = 64


def _write_cutout(path: str, arr: np.ndarray, magzero: float) -> None:
    hdu = fits.PrimaryHDU(np.asarray(arr, dtype=np.float32))
    hdu.header["MAGZERO"] = magzero
    hdu.header["CRVAL1"] = 150.0
    hdu.header["CRVAL2"] = 2.0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdu.writeto(path, overwrite=True)


def test_measure_core_saturation_flattop_and_nan():
    img = np.zeros((40, 40), dtype=np.float64)
    img[18:23, 18:23] = 500.0                  # 5×5 flat-topped saturated core
    m = measure_core_saturation(img)
    assert m["peak_e"] == pytest.approx(500.0)
    assert m["flattop_px"] == 25               # the whole flat core
    assert m["nan_core_px"] == 0

    img[20, 20] = np.nan                        # MER-masked centre
    m2 = measure_core_saturation(img)
    assert m2["nan_core_px"] >= 1


def test_load_cutout_applies_magzero_conversion(tmp_path):
    band = Config.BAND_VIS
    # MAGZERO 2.5 below the stack zeropoint → factor 10× (ADU/s → e⁻).
    magzero = band.sim_zeropoint_e - 2.5
    assert _adu_to_e_factor(magzero, band) == pytest.approx(10.0)
    arr = np.zeros((16, 16), dtype=np.float32)
    arr[8, 8] = 7.0
    p = str(tmp_path / "c.fits")
    _write_cutout(p, arr, magzero)
    img_e, mz = load_cutout_electrons(p, band)
    assert mz == pytest.approx(magzero)
    assert img_e[8, 8] == pytest.approx(70.0)  # 7 ADU/s × 10


def test_scan_stars_reads_per_band(tmp_path):
    root = str(tmp_path / "euclid_stars")
    cutouts = os.path.join(root, Config.CUTOUTS_SUBDIR)
    band = Config.BAND_VIS
    magzero = band.sim_zeropoint_e            # factor 1 → e⁻ == ADU/s value
    vis_dir = Config.cutout_dir_for_band("VIS", root=cutouts)

    # Bright saturated star (flat core 5×5 @ 8e5) and a faint star (single px).
    bright = np.zeros((_SIZE, _SIZE), np.float32); bright[30:35, 30:35] = 8.0e5
    faint = np.zeros((_SIZE, _SIZE), np.float32); faint[_SIZE // 2, _SIZE // 2] = 3.0e2
    _write_cutout(os.path.join(vis_dir, f"star_0001_{_SIZE}.fits"), bright, magzero)
    _write_cutout(os.path.join(vis_dir, f"star_0002_{_SIZE}.fits"), faint, magzero)

    stars_csv = os.path.join(root, Config.CATALOG_FILE)
    with open(stars_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "ra", "dec", "magnitude"])
        w.writerow([1, 150.0, 2.0, 12.0])      # bright
        w.writerow([2, 150.1, 2.1, 21.0])      # faint

    data = scan_stars(stars_csv, root, size=_SIZE, n=10)
    vis = data["VIS"]
    assert len(vis) == 2
    by_mag = {round(r["mag"]): r for r in vis}
    assert by_mag[12]["peak_e"] == pytest.approx(8.0e5)   # saturated ceiling
    assert by_mag[12]["flattop_px"] == 25
    assert by_mag[21]["peak_e"] == pytest.approx(3.0e2)   # faint, unsaturated
    assert by_mag[21]["flattop_px"] == 1
    _summarize_band(vis)                                  # runs without error
    # NISP bands have no cutouts here → no records, no crash
    assert data["H_E"] == []


def test_scan_is_cutout_driven_despite_catalog_id_mismatch(tmp_path):
    """The real failure mode: stars.csv ids don't match the cutout ids (the OOM
    reset the catalog). The scan must still read the cutout FILES that exist —
    mag annotated NaN — so the saturation ceiling is still measured."""
    root = str(tmp_path / "euclid_stars")
    cutouts = os.path.join(root, Config.CUTOUTS_SUBDIR)
    band = Config.BAND_VIS
    vis_dir = Config.cutout_dir_for_band("VIS", root=cutouts)
    sat = np.zeros((_SIZE, _SIZE), np.float32); sat[30:35, 30:35] = 8.0e5
    faint = np.zeros((_SIZE, _SIZE), np.float32); faint[_SIZE // 2, _SIZE // 2] = 3.0e2
    _write_cutout(os.path.join(vis_dir, f"star_1000_{_SIZE}.fits"), sat, band.sim_zeropoint_e)
    _write_cutout(os.path.join(vis_dir, f"star_1001_{_SIZE}.fits"), faint, band.sim_zeropoint_e)
    # stars.csv points at DIFFERENT ids (1, 2) — desynced catalog.
    stars_csv = os.path.join(root, Config.CATALOG_FILE)
    with open(stars_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "ra", "dec", "magnitude"])
        w.writerow([1, 150.0, 2.0, 12.0]); w.writerow([2, 150.1, 2.1, 21.0])

    vis = scan_stars(stars_csv, root, size=_SIZE, n=10)["VIS"]
    assert len(vis) == 2                                  # cutouts read despite mismatch
    assert max(r["peak_e"] for r in vis) == pytest.approx(8.0e5)   # ceiling found
    assert all(np.isnan(r["mag"]) for r in vis)          # no catalog mag matched
