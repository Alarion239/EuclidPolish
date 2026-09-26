"""Real-data SR metrics (plan WP-B2 T4; definitions of the 2026-09-23/25 real
galaxy analysis): hole %, enclosed-flux R around bright locally dominant
peaks after the central-pixel-fraction artifact cut, total SR/LR flux."""

from __future__ import annotations

import json

import numpy as np
import pytest

from euclid_polish.web.helpers import real_metrics as rm

BANDS = ("VIS", "Y_E", "J_E", "H_E")


def _gaussian(shape, x0, y0, amplitude, sigma):
    yy, xx = np.indices(shape, dtype=np.float64)
    return amplitude * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))


def _scene(n=128, *, seed=3, stars=((30, 30), (90, 40), (60, 100)), sigma=2.0,
           amplitude=2.0e4, noise=10.0, background=20.0):
    rng = np.random.default_rng(seed)
    plane = np.full((n, n), background, np.float64) + rng.normal(0, noise, (n, n))
    for x, y in stars:
        plane += _gaussian((n, n), x, y, amplitude, sigma)
    return np.repeat(plane[..., None], 4, axis=-1).astype(np.float32)


def _flux_conserving_sr(lr):
    """Every LR pixel's flux split evenly over its 2x2 SR block."""
    return (np.kron(lr, np.ones((2, 2, 1))) / 4.0).astype(np.float32)


def test_flux_conserving_sr_has_no_holes_and_unit_r():
    lr = _scene()
    metrics = rm.tile_metrics(lr, _flux_conserving_sr(lr), BANDS)
    json.dumps(metrics, allow_nan=False)
    for band in BANDS:
        m = metrics["per_band"][band]
        assert m["hole_pct"] == 0.0
        assert m["n_bright_px"] > 0
        assert m["flux_ratio"] == pytest.approx(1.0, rel=1e-5)
        assert m["n_peaks"] == 3
        assert m["median_R"] == pytest.approx(1.0, abs=1e-4)
        assert m["pct_R_lt_0p8"] == 0.0 and m["pct_R_lt_0p5"] == 0.0
    assert metrics["summary"]["n_peaks"] == 12


def test_zeroed_cores_are_holes_and_break_enclosed_flux():
    lr = _scene()
    sr = _flux_conserving_sr(lr)
    threshold = np.percentile(lr[..., 0], 99)
    bright = np.kron((lr[..., 0] >= threshold).astype(np.float32), np.ones((2, 2))) > 0
    sr[bright] = 0.0
    metrics = rm.tile_metrics(lr, sr, BANDS)
    for band in BANDS:
        m = metrics["per_band"][band]
        assert m["hole_pct"] == pytest.approx(100.0)
        assert m["pct_R_lt_0p8"] == pytest.approx(100.0)
        assert m["median_R"] < 0.8
        assert m["flux_ratio"] < 1.0


def test_hole_threshold_is_half_the_lr_flux_per_sr_pixel():
    lr = _scene()
    sr = _flux_conserving_sr(lr)
    just_above = rm.tile_metrics(lr, sr * 0.51, BANDS)["per_band"]["VIS"]
    just_below = rm.tile_metrics(lr, sr * 0.49, BANDS)["per_band"]["VIS"]
    assert just_above["hole_pct"] == 0.0
    assert just_below["hole_pct"] == pytest.approx(100.0)


def test_hot_pixels_are_dropped_by_the_central_pixel_fraction():
    lr = _scene(stars=((30, 30),))
    lr[100, 100, :] += 5.0e4                 # a lone hot pixel (cpf ~ 1)
    m = rm.tile_metrics(lr, _flux_conserving_sr(lr), BANDS)["per_band"]
    assert m["VIS"]["n_peaks"] == 1 and m["VIS"]["n_artifacts"] == 1
    # the NISP cut is stricter (0.14): a sharp but real VIS-like source can
    # pass in VIS and be cut in NISP
    sharp = _scene(stars=((64, 64),), sigma=1.0)   # cpf ~ 0.20
    s = rm.tile_metrics(sharp, _flux_conserving_sr(sharp), BANDS)["per_band"]
    assert s["VIS"]["n_peaks"] == 1
    assert s["H_E"]["n_peaks"] == 0 and s["H_E"]["n_artifacts"] == 1


def test_only_locally_dominant_bright_peaks_count():
    # 10 px = 1.0" apart: only the brighter one is locally dominant (±1.5")
    close = _scene(stars=((40, 40), (50, 40)))
    close[..., :] += _gaussian(close.shape[:2], 40, 40, 1.0e4, 2.0)[..., None]
    m = rm.tile_metrics(close, _flux_conserving_sr(close), BANDS)["per_band"]["VIS"]
    assert m["n_peaks"] == 1
    # 20 px = 2.0" apart: both count
    far = _scene(stars=((40, 40), (60, 40)))
    assert rm.tile_metrics(far, _flux_conserving_sr(far), BANDS)["per_band"]["VIS"]["n_peaks"] == 2
    # a peak below 100 sigma is not bright
    faint = _scene(stars=((40, 40),), amplitude=500.0)
    assert rm.tile_metrics(faint, _flux_conserving_sr(faint), BANDS)["per_band"]["VIS"]["n_peaks"] == 0


def test_no_peaks_reports_nulls_not_nan():
    lr = _scene(stars=())
    metrics = rm.tile_metrics(lr, _flux_conserving_sr(lr), BANDS)
    m = metrics["per_band"]["VIS"]
    assert m["n_peaks"] == 0 and m["median_R"] is None and m["pct_R_lt_0p8"] is None
    json.dumps(metrics, allow_nan=False)


def test_shape_mismatch_is_rejected():
    lr = _scene()
    with pytest.raises(ValueError):
        rm.tile_metrics(lr, lr, BANDS)


def test_aggregate_pools_tiles():
    lr = _scene()
    good = rm.tile_metrics(lr, _flux_conserving_sr(lr), BANDS)
    sr = _flux_conserving_sr(lr) * 0.3
    bad = rm.tile_metrics(lr, sr, BANDS)
    pooled = rm.aggregate([good, bad])
    assert pooled["n_tiles"] == 2
    vis = pooled["per_band"]["VIS"]
    assert vis["n_peaks"] == 6
    assert vis["pct_R_lt_0p5"] == pytest.approx(50.0)
    assert vis["hole_pct"] == pytest.approx(50.0)


def test_hole_variant_restricted_to_pixels_above_100_sigma():
    faint = _scene(stars=())                       # pure noise: the top 1 % is noise
    sr = _flux_conserving_sr(faint) * 0.4
    m = rm.tile_metrics(faint, sr, BANDS)["per_band"]["VIS"]
    assert m["hole_pct"] == pytest.approx(100.0)
    assert m["hole_pct_100sigma"] is None and m["n_bright_100sigma_px"] == 0
    bright = _scene()
    b = rm.tile_metrics(bright, _flux_conserving_sr(bright) * 0.4, BANDS)["per_band"]["VIS"]
    assert b["hole_pct_100sigma"] == pytest.approx(100.0)
    assert 0 < b["n_bright_100sigma_px"] <= b["n_bright_px"]
    pooled = rm.aggregate([rm.tile_metrics(bright, _flux_conserving_sr(bright), BANDS)])
    assert pooled["per_band"]["VIS"]["hole_pct_100sigma"] == 0.0
