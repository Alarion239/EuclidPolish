"""Flux conservation against the MER catalogue (eval/flux_conservation.py):
the Kron ellipse follows MER's angle convention, the input split onto the SR
grid measures exactly like the input, a flux-keeping sharpening keeps the
Kron total, and the summary reads pulls in catalogue σ."""
from __future__ import annotations

import numpy as np

from euclid_polish.eval import flux_conservation as fc


def _galaxy(n=96, x0=47.3, y0=48.6, a=6.0, q=0.4, pa_deg=30.0, flux=1e4):
    """An elliptical Gaussian whose major axis lies at ``pa_deg + 90°`` from x
    (MER's POSITION_ANGLE convention)."""
    yy, xx = np.mgrid[:n, :n].astype(float)
    t = np.deg2rad(pa_deg + fc.KRON_ANGLE_OFFSET_DEG)
    u = (xx - x0) * np.cos(t) + (yy - y0) * np.sin(t)
    v = -(xx - x0) * np.sin(t) + (yy - y0) * np.cos(t)
    image = np.exp(-0.5 * ((u / a) ** 2 + (v / (a * q)) ** 2))
    return flux * image / image.sum()


def test_kron_ellipse_follows_mer_position_angle():
    image = _galaxy()
    along = fc.kron_flux(image, 47.3, 48.6, 12.0, 0.6, 30.0)
    across = fc.kron_flux(image, 47.3, 48.6, 12.0, 0.6, 30.0 + 90.0)
    assert along > 0.8 * image.sum()
    assert across < 0.75 * along


def test_input_split_onto_sr_grid_measures_like_the_input():
    lr = np.stack([_galaxy(pa_deg=10.0 * b) for b in range(4)], -1)
    sr = np.repeat(np.repeat(lr, 2, 0), 2, 1) / 4.0
    row = {"kron_radius": 10.0, "ellipticity": 0.5, "position_angle": 15.0, "fwhm": 1.2}
    lo = fc.measure(lr, np.array([[47.3, 48.6]]), [row], 1.2, scale=1)[0]
    hi = fc.measure(sr, np.array([[2 * 47.3 + 0.5, 2 * 48.6 + 0.5]]), [row], 1.2, scale=2)[0]
    assert abs(hi["kron_vis"] / lo["kron_vis"] - 1) < 1e-6
    for band in fc.BANDS:
        assert abs(hi[f"ap4_{band}"] / lo[f"ap4_{band}"] - 1) < 2e-3


def test_flux_keeping_sharpening_keeps_the_kron_total():
    broad = _galaxy(a=5.0)
    sharp = _galaxy(a=3.0)
    assert abs(fc.kron_flux(sharp, 47.3, 48.6, 20.0, 0.5, 30.0)
               / fc.kron_flux(broad, 47.3, 48.6, 20.0, 0.5, 30.0) - 1) < 0.02


def test_summary_reads_pulls_in_catalogue_sigma():
    rng = np.random.default_rng(0)
    cat = rng.uniform(100, 1000, 400)
    err = np.full(400, 10.0)
    lr = cat.copy()
    model = cat - 20.0                       # 2σ low everywhere
    mag = np.full(400, 21.0)
    s = fc.summarize(model, lr, cat, err, mag)
    assert s["n"] == 400 and abs(s["pull_median"] + 2.0) < 1e-9
    assert s["within_1sigma"] == 0.0 and s["within_3sigma"] == 1.0
    assert s["input_pull_mad"] == 0.0
    assert s["by_mag"][1]["n"] == 400


def test_aperture_correction_is_the_median_input_ratio():
    assert fc.aperture_correction(np.array([1.1, 1.2, 1.3, np.nan]), np.ones(4)) == 1.2
