"""Bright-star detector saturation (euclid_polish.sky.observation.saturation)."""

from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.sky.observation.saturation import (
    StarSaturationModel,
    apply_saturation_masking,
    saturation_mask_probability,
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
    """Without a star plane the trigger is source-agnostic: a bright EXTENDED
    source still saturates at the star-derived well and is masked."""
    m = StarSaturationModel()
    lr = np.zeros((40, 40, len(_BANDS)), dtype=np.float32)
    well = m.well_depth_e(Config.get_band(_BANDS[0]))
    lr[15:25, 15:25, 0] = np.float32(well * 5.0)           # 10×10 bright core
    apply_saturation_masking(lr, m, np.random.default_rng(1), band_names=_BANDS)
    assert lr[..., 0].max() < well                         # whole core masked


def test_separate_trigger_does_not_expand_detector_artifacts():
    """Only the optical scene triggers blackout geometry.

    A post-noise hot/CR-like spike can stay above the effective well, while a
    genuinely saturated source in the trigger is zeroed in the final image.
    """
    m = StarSaturationModel()
    lr = np.zeros((40, 40, len(_BANDS)), dtype=np.float32)
    trigger = np.zeros_like(lr)
    band_index = 1
    well = m.well_depth_e(Config.get_band(_BANDS[band_index]))
    lr[5, 5, band_index] = np.float32(well * 20.0)       # detector artifact
    lr[18:22, 18:22, band_index] = np.float32(well * 5.0)
    trigger[18:22, 18:22, band_index] = np.float32(well * 5.0)

    apply_saturation_masking(
        lr,
        m,
        np.random.default_rng(2),
        band_names=_BANDS,
        trigger_4ch=trigger,
    )

    assert lr[5, 5, band_index] == pytest.approx(well * 20.0)
    assert lr[18:22, 18:22, band_index].max() == 0.0


def test_separate_trigger_shape_must_match_dirty_image():
    m = StarSaturationModel()
    lr = np.zeros((20, 20, len(_BANDS)), dtype=np.float32)
    with pytest.raises(ValueError, match="trigger_4ch shape"):
        apply_saturation_masking(
            lr,
            m,
            np.random.default_rng(0),
            band_names=_BANDS,
            trigger_4ch=np.zeros((19, 20, len(_BANDS)), dtype=np.float32),
        )


def test_zero_mask_probability_preserves_bright_core():
    m = StarSaturationModel()
    lr = np.zeros((24, 24, len(_BANDS)), dtype=np.float32)
    well = m.well_depth_e(Config.get_band(_BANDS[0]))
    lr[10:14, 10:14, 0] = np.float32(well * 3.0)

    apply_saturation_masking(
        lr,
        m,
        np.random.default_rng(0),
        band_names=_BANDS,
        mask_probability=0.0,
    )

    assert lr[10:14, 10:14, 0].min() == pytest.approx(well * 3.0)


def test_mask_probability_must_be_unit_interval():
    m = StarSaturationModel()
    lr = np.zeros((20, 20, len(_BANDS)), dtype=np.float32)
    with pytest.raises(ValueError, match="mask_probability"):
        apply_saturation_masking(
            lr,
            m,
            np.random.default_rng(0),
            band_names=_BANDS,
            mask_probability=1.01,
        )


def test_mask_probability_draw_is_shared_across_bands():
    m = StarSaturationModel()
    lr = np.zeros((28, 28, len(_BANDS)), dtype=np.float32)
    for k, name in enumerate(_BANDS):
        well = m.well_depth_e(Config.get_band(name))
        lr[12:16, 12:16, k] = np.float32(well * 2.0)

    apply_saturation_masking(
        lr,
        m,
        np.random.default_rng(3),
        band_names=_BANDS,
        mask_probability=0.5,
    )

    masked = [bool(np.all(lr[12:16, 12:16, k] == 0.0))
              for k in range(len(_BANDS))]
    assert len(set(masked)) == 1


def test_blackout_probability_rises_with_peak_over_well():
    probability = saturation_mask_probability(
        np.array([1.0, 5.0, 10.0, 20.0, 100.0]),
        near_well=0.2, bright=0.9, ratio_start=5.0, ratio_full=20.0,
    )
    np.testing.assert_allclose(probability, [0.2, 0.2, 0.55, 0.9, 0.9])


def test_blackout_probability_never_drops_below_near_well_value():
    probability = saturation_mask_probability(
        np.array([1.0, 10.0, 100.0]),
        near_well=1.0, bright=0.5, ratio_start=5.0, ratio_full=20.0,
    )
    np.testing.assert_allclose(probability, 1.0)


@pytest.mark.parametrize("kwargs", [
    {"bright_mask_probability": 1.5},
    {"bright_well_ratios": (20.0, 5.0)},
    {"bright_well_ratios": (0.0, 5.0)},
])
def test_bright_blackout_settings_are_validated(kwargs):
    m = StarSaturationModel()
    lr = np.zeros((20, 20, len(_BANDS)), dtype=np.float32)
    with pytest.raises(ValueError):
        apply_saturation_masking(
            lr, m, np.random.default_rng(0), band_names=_BANDS,
            mask_probability=0.2, **kwargs,
        )


def test_brightest_sources_are_blacked_out_more_often():
    """A core far above the well is masked almost always, while one just above
    it keeps the low near-well probability, as in real MER fields."""
    m = StarSaturationModel()
    k = _BANDS.index("J_E")
    well = m.well_depth_e(Config.get_band("J_E"))
    faint_masked = bright_masked = 0
    for seed in range(200):
        lr = np.zeros((64, 64, len(_BANDS)), dtype=np.float32)
        lr[8:12, 8:12, k] = np.float32(well * 1.5)
        lr[48:52, 48:52, k] = np.float32(well * 50.0)
        apply_saturation_masking(
            lr, m, np.random.default_rng(seed), band_names=_BANDS,
            mask_probability=0.05, bright_mask_probability=1.0,
            bright_well_ratios=(5.0, 20.0),
        )
        faint_masked += bool(np.all(lr[8:12, 8:12, k] == 0.0))
        bright_masked += bool(np.all(lr[48:52, 48:52, k] == 0.0))
    assert bright_masked == 200
    assert faint_masked < 30


def _core_with_halo(band_index: int, core_ratio: float, halo_ratio: float,
                    n: int = 48):
    """A 12x12 region at ``halo_ratio`` x the band well with a 4x4 core at
    ``core_ratio`` x the well, in one band."""
    m = StarSaturationModel()
    well = m.well_depth_e(Config.get_band(_BANDS[band_index]))
    img = np.zeros((n, n, len(_BANDS)), dtype=np.float32)
    img[18:30, 18:30, band_index] = np.float32(well * halo_ratio)
    img[22:26, 22:26, band_index] = np.float32(well * core_ratio)
    return m, well, img


def test_galaxy_light_below_extended_well_is_never_blanked():
    """With a star plane, galaxy light is recorded up to
    SATURATION_EXTENDED_WELL_FACTOR x the star-derived well."""
    k = _BANDS.index("J_E")
    ratio = 0.8 * Config.SATURATION_EXTENDED_WELL_FACTOR
    m, _well, lr = _core_with_halo(k, core_ratio=ratio, halo_ratio=2.0)
    before = lr.copy()
    apply_saturation_masking(
        lr, m, np.random.default_rng(0), band_names=_BANDS,
        trigger_4ch=before, star_trigger_4ch=np.zeros_like(lr),
        mask_probability=1.0)
    np.testing.assert_array_equal(lr, before)


def test_galaxy_core_loses_only_pixels_above_extended_well():
    """A selected galaxy source blanks the pixels above the extended well —
    no bounding box, no rectangles — and keeps the rest of its bright light."""
    k = _BANDS.index("J_E")
    factor = Config.SATURATION_EXTENDED_WELL_FACTOR
    m, well, lr = _core_with_halo(k, core_ratio=2.0 * factor, halo_ratio=2.0)
    trigger = lr.copy()
    apply_saturation_masking(
        lr, m, np.random.default_rng(0), band_names=_BANDS,
        trigger_4ch=trigger, star_trigger_4ch=np.zeros_like(lr),
        mask_probability=1.0)
    assert lr[22:26, 22:26, k].max() == 0.0                # core blanked
    halo = trigger[..., k] == np.float32(well * 2.0)
    np.testing.assert_array_equal(lr[..., k][halo], trigger[..., k][halo])
    assert int((lr[..., k] == 0.0).sum()) == int((trigger[..., k] == 0.0).sum()) + 16


def test_star_dominated_source_keeps_the_box_blackout():
    """The same light on the star plane is a star core: the star-derived well
    and the bounding-box + rectangle blackout still apply."""
    k = _BANDS.index("J_E")
    m, well, lr = _core_with_halo(k, core_ratio=3.0, halo_ratio=2.0)
    trigger = lr.copy()
    apply_saturation_masking(
        lr, m, np.random.default_rng(0), band_names=_BANDS,
        trigger_4ch=trigger, star_trigger_4ch=trigger.copy(),
        mask_probability=1.0)
    assert lr[18:30, 18:30, k].max() == 0.0                # whole box blanked
    assert lr[..., k].max() < well


def test_galaxy_and_star_share_one_probability_draw_order():
    """Without a star plane the draw sequence and result are the legacy ones."""
    k = _BANDS.index("J_E")
    m, _well, lr = _core_with_halo(k, core_ratio=3.0, halo_ratio=2.0)
    a, b = lr.copy(), lr.copy()
    apply_saturation_masking(a, m, np.random.default_rng(5), band_names=_BANDS,
                             mask_probability=0.5)
    apply_saturation_masking(b, m, np.random.default_rng(5), band_names=_BANDS,
                             trigger_4ch=lr, star_trigger_4ch=lr.copy(),
                             mask_probability=0.5)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("kwargs, match", [
    ({"star_trigger_4ch": np.zeros((19, 20, len(_BANDS)), np.float32)}, "star_trigger_4ch shape"),
    ({"star_trigger_4ch": np.zeros((20, 20, len(_BANDS)), np.float32),
      "extended_well_factor": 0.5}, "extended_well_factor"),
])
def test_star_plane_settings_are_validated(kwargs, match):
    m = StarSaturationModel()
    lr = np.zeros((20, 20, len(_BANDS)), dtype=np.float32)
    with pytest.raises(ValueError, match=match):
        apply_saturation_masking(lr, m, np.random.default_rng(0),
                                 band_names=_BANDS, **kwargs)


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


def test_forward_blacks_out_very_bright_sources_at_training_probability():
    """A core many times above every well (the test idx16 star) is blacked out
    in most exposures even at the low near-well training probability."""
    from euclid_polish.sky.observation.observation_simulator import (
        ObservationSimulator,
        ObservationSimulatorConfig,
    )
    fwd = ObservationSimulator(config=ObservationSimulatorConfig(
        add_noise=False, add_artifacts=False, add_saturation=True,
        saturation_mask_prob=Config.TRAIN_SATURATION_MASK_PROB))
    masked = 0
    for seed in range(20):
        lr, _ = fwd.process(_hr_field_with_bright_source(1e7),
                            np.random.default_rng(seed))
        masked += float(lr.data[11:13, 11:13, :].max()) == 0.0
    assert masked >= 14


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


def test_forward_hot_pixels_do_not_trigger_blackout_rectangles():
    from euclid_polish.sky.observation.artifacts import ArtifactConfig
    from euclid_polish.sky.observation.observation_simulator import (
        ObservationSimulator,
        ObservationSimulatorConfig,
    )

    artifact_config = ArtifactConfig(
        add_cosmic_rays=False,
        add_hot_pixels=True,
        hot_pixel_fraction=1.0,
        hot_pixel_charge_mean_e=1.0e7,
        add_dead_pixels=False,
        add_streaks=False,
    )
    fwd = ObservationSimulator(config=ObservationSimulatorConfig(
        add_noise=True,
        add_artifacts=True,
        artifact_config=artifact_config,
        add_saturation=True,
    ))
    lr, _ = fwd.process(
        _hr_field_with_bright_source(0.0),
        np.random.default_rng(4),
    )

    for band_index, band_name in enumerate(_BANDS):
        well = StarSaturationModel().well_depth_e(Config.get_band(band_name))
        assert lr.data[..., band_index].max() > well


def _j_peak_over_well(fwd, hr_img, star_plane) -> tuple[float, bool]:
    lr, _ = fwd.process(hr_img, np.random.default_rng(0), star_hr_4ch=star_plane)
    k = _BANDS.index("J_E")
    well = StarSaturationModel().well_depth_e(Config.get_band("J_E"))
    core = lr.data[20:28, 20:28, k]
    return float(core.max()) / well, bool((core == 0.0).any())


def test_forward_blanks_a_star_core_but_records_galaxy_light_at_the_same_level():
    """In the forward model the star plane decides the rule: the same
    above-well light is a blacked-out star core on the star plane and a
    recorded galaxy core in the scene (below the extended well)."""
    from euclid_polish.image import Image
    from euclid_polish.sky.observation.observation_simulator import (
        ObservationSimulator,
        ObservationSimulatorConfig,
    )
    plain = ObservationSimulator(config=ObservationSimulatorConfig(
        add_noise=False, add_artifacts=False, add_saturation=False))
    masking = ObservationSimulator(config=ObservationSimulatorConfig(
        add_noise=False, add_artifacts=False, add_saturation=True,
        saturation_mask_prob=1.0))
    unit = _hr_field_with_bright_source(1.0)
    zeros = np.zeros_like(unit.data)
    ratio_unit, _ = _j_peak_over_well(plain, unit, zeros)
    flux = 2.5 / ratio_unit                                  # J core at 2.5x the well
    galaxy = _hr_field_with_bright_source(flux)
    empty = Image(data=zeros, pixel_scale_arcsec=unit.pixel_scale_arcsec,
                  band_names=_BANDS, is_clean=True, metadata={"stars": []})

    ratio, blanked = _j_peak_over_well(masking, galaxy, zeros)
    assert 2.0 < ratio < Config.SATURATION_EXTENDED_WELL_FACTOR
    assert not blanked
    _, star_blanked = _j_peak_over_well(masking, empty, galaxy.data)
    assert star_blanked
