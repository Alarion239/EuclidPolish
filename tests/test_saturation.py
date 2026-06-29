"""Bright-star detector saturation (euclid_polish.sky.observation.saturation)."""

from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.sky.observation.saturation import (
    StarSaturationModel,
    apply_saturation_masking,
)

_BANDS = Config.LR_INPUT_BAND_NAMES


def test_well_depths_recover_detector_full_wells():
    m = StarSaturationModel()
    assert 1.7e5 < m.well_depth_e(Config.BAND_VIS) < 2.2e5      # VIS CCD ~198k e-
    for b in (Config.BAND_Y_E, Config.BAND_J_E, Config.BAND_H_E):
        # Effective stack-referred clip level (calibrated from the observed
        # 50%-saturation mags, so it scales with t_total — 4×87.2 s MACC
        # integration), not the physical H2RG full well. Y/J/H ≈ 7.8/4.6/9.5k.
        assert 4e3 < m.well_depth_e(b) < 1.5e4


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


def test_apply_saturation_masking_zeros_over_well_regions():
    """Pixels at/above the well depth (any source) are masked to ~0 over a
    rectangular patch — nothing stays above the well."""
    m = StarSaturationModel()
    H = W = 64
    lr = np.zeros((H, W, len(_BANDS)), dtype=np.float32)
    for k, bn in enumerate(_BANDS):
        well = m.well_depth_e(Config.get_band(bn))
        lr[30:34, 30:34, k] = np.float32(well * 50.0)      # 50× over the well
    apply_saturation_masking(lr, m, np.random.default_rng(0), band_names=_BANDS)
    for k, bn in enumerate(_BANDS):
        well = m.well_depth_e(Config.get_band(bn))
        assert lr[..., k].max() < well                     # nothing above well
        assert lr[30:34, 30:34, k].max() == 0.0            # core masked to 0
        assert (lr[..., k] == 0.0).sum() >= 16             # a patch's worth


def test_apply_saturation_masking_leaves_subwell_untouched():
    m = StarSaturationModel()
    lr = np.zeros((32, 32, len(_BANDS)), dtype=np.float32)
    for k, bn in enumerate(_BANDS):
        well = m.well_depth_e(Config.get_band(bn))
        lr[10:14, 10:14, k] = np.float32(well * 0.5)       # below the well
    before = lr.copy()
    apply_saturation_masking(lr, m, np.random.default_rng(0), band_names=_BANDS)
    np.testing.assert_array_equal(lr, before)              # no saturation → no-op


def test_apply_saturation_masking_galaxy_core_not_just_stars():
    """A bright EXTENDED source (no star metadata) still saturates and is
    masked — the trigger is the pixel value, not the source type."""
    m = StarSaturationModel()
    lr = np.zeros((40, 40, len(_BANDS)), dtype=np.float32)
    well = m.well_depth_e(Config.get_band(_BANDS[0]))
    lr[15:25, 15:25, 0] = np.float32(well * 5.0)           # 10×10 bright core
    apply_saturation_masking(lr, m, np.random.default_rng(1), band_names=_BANDS)
    assert lr[..., 0].max() < well                         # whole core masked


# ---------------------------------------------------------------------------
# Forward-model integration
# ---------------------------------------------------------------------------

def _hr_field_with_bright_source(flux_e: float, n: int = 48):
    """An HR field with a bright core that drives the LR past the well."""
    from euclid_polish.image import Image
    data = np.zeros((n, n, len(_BANDS)), dtype=np.float32)
    data[n // 2 - 4:n // 2 + 4, n // 2 - 4:n // 2 + 4, :] = np.float32(flux_e)
    return Image(
        data=data, pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=_BANDS, is_clean=True, metadata={"stars": []})


def test_forward_masks_saturation_in_dirty_not_target():
    from euclid_polish.sky.observation.observation_simulator import (
        ObservationSimulator,
        ObservationSimulatorConfig,
    )
    fwd = ObservationSimulator(config=ObservationSimulatorConfig(
        add_noise=False, add_artifacts=False, add_saturation=True))
    lr, hr = fwd.process(_hr_field_with_bright_source(1e6),
                         np.random.default_rng(0))
    well_vis = StarSaturationModel().well_depth_e(Config.BAND_VIS)
    # Dirty VIS: the saturated core is masked to ~0 — nothing above the well.
    assert lr.data[..., 0].max() < well_vis
    assert float(lr.data[..., 0].min()) <= 0.0
    # Clean HR target keeps the bright source (untouched).
    assert hr.data.max() == pytest.approx(1e6, rel=1e-4)


def test_forward_saturation_can_be_disabled():
    from euclid_polish.sky.observation.observation_simulator import (
        ObservationSimulator,
        ObservationSimulatorConfig,
    )
    well_vis = StarSaturationModel().well_depth_e(Config.BAND_VIS)
    fwd = ObservationSimulator(config=ObservationSimulatorConfig(
        add_noise=False, add_artifacts=False, add_saturation=False))
    lr, _ = fwd.process(_hr_field_with_bright_source(1e6), np.random.default_rng(0))
    # Saturation off → the over-well core is NOT masked (stays far above well).
    assert lr.data[..., 0].max() > well_vis
