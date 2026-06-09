"""Bright-star detector saturation (euclid_polish.sky.saturation)."""

from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.sky.saturation import (
    StarSaturationModel, apply_star_saturation,
)

_BANDS = Config.LR_INPUT_BAND_NAMES


def test_well_depths_recover_detector_full_wells():
    m = StarSaturationModel()
    assert 1.7e5 < m.well_depth_e(Config.BAND_VIS) < 2.2e5      # VIS CCD ~198k e-
    for b in (Config.BAND_Y_E, Config.BAND_J_E, Config.BAND_H_E):
        assert 5e3 < m.well_depth_e(b) < 1.2e4                  # NISP H2RG ~8-10k


def test_probability_half_at_calibration_magnitudes():
    m = StarSaturationModel()
    assert m.saturation_probability(14.0, Config.BAND_VIS) == pytest.approx(0.5, abs=1e-6)
    assert m.saturation_probability(17.0, Config.BAND_Y_E) == pytest.approx(0.5, abs=1e-6)


def test_probability_monotonic_and_soft():
    m = StarSaturationModel()
    b = Config.BAND_VIS
    # brighter ⇒ higher saturation probability
    p = [m.saturation_probability(mv, b) for mv in (12, 13, 14, 15, 16)]
    assert p[0] >= p[1] >= p[2] >= p[3] >= p[4]
    assert p[0] > 0.99 and p[-1] < 0.01
    # SOFT onset: ±0.3 mag around the 50% point is strictly between 0 and 1.
    assert 0.6 < m.saturation_probability(13.7, b) < 0.95
    assert 0.05 < m.saturation_probability(14.3, b) < 0.4


def test_saturates_empirical_rate_matches_probability():
    m = StarSaturationModel()
    rng = np.random.default_rng(0)
    frac = np.mean([m.saturates(14.0, Config.BAND_VIS, rng) for _ in range(6000)])
    assert frac == pytest.approx(0.5, abs=0.03)


def test_bands_saturate_independently():
    # A mag-15 star: well above NISP onset, well below VIS onset.
    m = StarSaturationModel()
    assert m.saturation_probability(15.0, Config.BAND_VIS) < 0.01
    assert m.saturation_probability(15.0, Config.BAND_Y_E) > 0.99


def test_sharp_onset_when_jitter_zero():
    m = StarSaturationModel(jitter_dex=0.0)
    b = Config.BAND_VIS
    assert m.saturation_probability(13.9, b) == 1.0
    assert m.saturation_probability(14.1, b) == 0.0


def _rects_overlap(a, b) -> bool:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    return (ax0 < bx0 + bw and bx0 < ax0 + aw
            and ay0 < by0 + bh and by0 < ay0 + ah)


def test_rectangles_shape_and_overlap():
    m = StarSaturationModel()
    rng = np.random.default_rng(3)
    for _ in range(400):
        rects = m.rectangles(rng)
        assert 1 <= len(rects) <= m.max_rects
        x0, y0, w, h = rects[0]
        for (_, _, rw, rh) in rects:
            assert m.rect_min_px <= rw <= m.rect_max_px
            assert m.rect_min_px <= rh <= m.rect_max_px
        # first rectangle contains the peak pixel (origin)
        assert x0 <= 0 < x0 + w and y0 <= 0 < y0 + h
        # every other rectangle overlaps the first
        for r in rects[1:]:
            assert _rects_overlap(rects[0], r)


def test_apply_saturation_stamps_bright_star_only():
    m = StarSaturationModel()
    H = W = 64
    lr = np.zeros((H, W, len(_BANDS)), dtype=np.float32)
    # mag-13 star at HR (64,64) → LR (32,32) with hr_to_lr=0.5; saturates all bands.
    stars = [{"type": "star", "x_pix": 64.0, "y_pix": 64.0, "mag_vis": 13.0}]
    n = apply_star_saturation(lr, stars, m, np.random.default_rng(0),
                              hr_to_lr_scale=0.5, band_names=_BANDS)
    assert n == len(_BANDS)                                     # all bands saturated
    for k, bn in enumerate(_BANDS):
        well = m.well_depth_e(Config.get_band(bn))
        assert lr[..., k].max() == pytest.approx(well, rel=1e-5)
        assert (lr[..., k] == np.float32(well)).sum() >= 9      # a rectangle's worth

    # A faint star leaves the image untouched.
    lr2 = np.zeros((H, W, len(_BANDS)), dtype=np.float32)
    faint = [{"type": "star", "x_pix": 64.0, "y_pix": 64.0, "mag_vis": 22.0}]
    n2 = apply_star_saturation(lr2, faint, m, np.random.default_rng(0),
                               hr_to_lr_scale=0.5, band_names=_BANDS)
    assert n2 == 0 and lr2.max() == 0.0


def test_apply_saturation_clips_out_of_bounds():
    m = StarSaturationModel()
    lr = np.zeros((16, 16, len(_BANDS)), dtype=np.float32)
    # star far off-canvas → nothing stamped, no crash
    stars = [{"type": "star", "x_pix": 9000.0, "y_pix": 9000.0, "mag_vis": 12.0}]
    apply_star_saturation(lr, stars, m, np.random.default_rng(0),
                          hr_to_lr_scale=0.5, band_names=_BANDS)
    assert lr.max() == 0.0


# ---------------------------------------------------------------------------
# Forward-model integration
# ---------------------------------------------------------------------------

def _hr_field_with_star(mag_vis: float, n: int = 48):
    from euclid_polish.sky.types import MultiBandSkyImage
    data = np.zeros((n, n, len(_BANDS)), dtype=np.float32)
    return MultiBandSkyImage(
        data=data, pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=_BANDS, is_clean=True,
        metadata={"stars": [{"type": "star", "x_pix": n / 2.0,
                             "y_pix": n / 2.0, "mag_vis": mag_vis}]})


def test_forward_bakes_saturation_into_dirty_not_target():
    from euclid_polish.sky.multiband_forward import (
        MultiBandForward, MultiBandForwardConfig,
    )
    fwd = MultiBandForward(config=MultiBandForwardConfig(
        add_noise=False, add_artifacts=False, add_saturation=True))
    lr, hr = fwd.process(_hr_field_with_star(13.0),
                         np.random.default_rng(0))
    # VIS dirty carries the clipped (well-depth) saturated core...
    well_vis = StarSaturationModel().well_depth_e(Config.BAND_VIS)
    assert lr.data[..., 0].max() == pytest.approx(well_vis, rel=1e-4)
    # ...but the clean HR target is untouched (all zero).
    assert hr.data.max() == 0.0


def test_forward_saturation_can_be_disabled():
    from euclid_polish.sky.multiband_forward import (
        MultiBandForward, MultiBandForwardConfig,
    )
    fwd = MultiBandForward(config=MultiBandForwardConfig(
        add_noise=False, add_artifacts=False, add_saturation=False))
    lr, _ = fwd.process(_hr_field_with_star(13.0), np.random.default_rng(0))
    assert lr.data.max() == 0.0          # no saturation stamped
