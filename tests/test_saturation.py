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


def test_vis_well_is_stack_referred_mccracken_blooming():
    """VIS well = McCracken+25 Q1 per-readout blooming (51 kADU × 3.48 e⁻/ADU)
    × the 4 VIS exposures our forward model co-adds = stack-referred ≈710 ke⁻.
    The per-readout value alone (177 ke⁻) would saturate ~1.5 mag too faint."""
    m = StarSaturationModel()
    vis = m.well_depth_e(Config.BAND_VIS)
    n_exp = Config.BAND_VIS.n_exposures
    assert vis == pytest.approx(n_exp * 51_000.0 * 3.48, rel=1e-6)   # 709 920 e-
    assert vis == pytest.approx(n_exp * 177_480.0, rel=1e-6)


def test_nisp_wells_are_effective_stack_referred_levels():
    m = StarSaturationModel()
    for b in (Config.BAND_Y_E, Config.BAND_J_E, Config.BAND_H_E):
        # Effective stack-referred clip level for the 4×87.2 s MACC integration
        # (not the physical H2RG full well). Y/J/H ≈ 7.8/4.6/9.5k e-.
        assert 4e3 < m.well_depth_e(b) < 1.5e4


def test_well_override_is_honoured():
    m = StarSaturationModel(well_e={"VIS": 1234.0})
    assert m.well_depth_e(Config.BAND_VIS) == 1234.0


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
