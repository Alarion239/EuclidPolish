"""Blank (exactly zero) NISP pixels of a MER mosaic: per-exposure invalid masks,
resampled with the bilinear stencil, blank only where every exposure is invalid."""

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.sky.observation.blank_pixels import (
    BAND_NAMES,
    BLANK_BAND_NAMES,
    BlankPixelModel,
    blank_where_all_invalid,
    draw_coverage_map,
    stencil_to_mosaic,
)

SHAPE = (384, 384)


def _blank_fraction(release: str, n_exposures: int, *, seeds: int, band: str = "H_E") -> float:
    model = BlankPixelModel(release)
    k = BAND_NAMES.index(band)
    total = 0.0
    for seed in range(seeds):
        coverage = np.full(SHAPE, n_exposures, np.int8)
        total += model.blank_mask(coverage, np.random.default_rng(seed))[..., k].mean()
    return total / seeds


def test_random_blanks_fall_on_the_three_nisp_bands_only():
    assert BAND_NAMES == Config.LR_INPUT_BAND_NAMES
    assert BLANK_BAND_NAMES == ("Y_E", "J_E", "H_E")
    blank = BlankPixelModel("dr1").blank_mask(np.full(SHAPE, 1, np.int8), np.random.default_rng(0))
    assert not blank[..., BAND_NAMES.index("VIS")].any()
    assert all(blank[..., BAND_NAMES.index(b)].any() for b in BLANK_BAND_NAMES)


def test_one_invalid_detector_pixel_removes_six_by_six_mosaic_pixels():
    """Bilinear resampling: every mosaic pixel whose 2x2 stencil touches the
    invalid 0.3" detector pixel loses it — 6x6 pixels on the 0.1" grid."""
    det = np.zeros((40, 40), np.uint8)
    det[20, 20] = 1
    mosaic = stencil_to_mosaic(det, (90, 90), theta_deg=0.0, offset=(0.0, 0.0))
    assert int(mosaic.sum()) == 36
    ys, xs = np.nonzero(mosaic)
    assert ys.max() - ys.min() == 5 and xs.max() - xs.min() == 5


def test_stencil_keeps_each_band_bit_separate():
    det = np.zeros((40, 40), np.uint8)
    det[10, 10] = 0b001
    det[25, 25] = 0b100
    mosaic = stencil_to_mosaic(det, (90, 90), theta_deg=0.0, offset=(0.0, 0.0))
    assert int((mosaic & 0b001 > 0).sum()) == 36
    assert int((mosaic & 0b100 > 0).sum()) == 36
    assert not np.any((mosaic & 0b001 > 0) & (mosaic & 0b100 > 0))


def test_pixel_is_blank_only_where_every_covering_exposure_is_invalid():
    a = np.zeros((4, 4), np.uint8)
    b = np.zeros((4, 4), np.uint8)
    a[0, :] = 1          # exposure 1 invalid along the top row
    b[:, 0] = 1          # exposure 2 invalid along the left column
    coverage = np.full((4, 4), 2, np.int8)
    coverage[:, 2:] = 1  # the right half is covered by exposure 1 only
    blank = blank_where_all_invalid([a, b], coverage)
    expected = np.zeros((4, 4), bool)
    expected[0, 0] = True        # both invalid
    expected[0, 2:] = True       # single coverage: exposure 1 alone decides
    np.testing.assert_array_equal(blank > 0, expected)


def test_uncovered_pixels_are_not_blanked():
    a = np.ones((3, 3), np.uint8)
    coverage = np.zeros((3, 3), np.int8)
    assert not blank_where_all_invalid([a], coverage).any()


def test_blank_mask_has_one_plane_per_band():
    blank = BlankPixelModel("dr1").blank_mask(np.full(SHAPE, 2, np.int8), np.random.default_rng(0))
    assert blank.shape == SHAPE + (4,)
    assert blank.dtype == bool


def test_blank_draw_is_reproducible_for_a_seed():
    model = BlankPixelModel("dr1")
    coverage = np.full(SHAPE, 2, np.int8)
    one = model.blank_mask(coverage, np.random.default_rng(7))
    two = model.blank_mask(coverage, np.random.default_rng(7))
    np.testing.assert_array_equal(one, two)


def test_dr1_blank_rate_falls_with_exposure_count_as_measured():
    """DR1 H band, measured: about 5-9% blank under one exposure, 0.9% under
    two, 0.13% under three."""
    one = _blank_fraction("dr1", 1, seeds=12)
    two = _blank_fraction("dr1", 2, seeds=24)
    three = _blank_fraction("dr1", 3, seeds=48)
    assert 0.04 < one < 0.14
    assert 0.003 < two < 0.025
    assert three < 0.006
    assert one > 4 * two > 8 * three


def test_q1_has_fewer_blanks_than_dr1():
    assert _blank_fraction("q1", 1, seeds=12) < 0.7 * _blank_fraction("dr1", 1, seeds=12)
    assert _blank_fraction("q1", 2, seeds=24) < 0.5 * _blank_fraction("dr1", 2, seeds=24)


def test_bands_share_part_of_their_blanks():
    """Persistence bands are common to Y, J and H; cosmic rays and most small
    persistence clusters are not."""
    model = BlankPixelModel("dr1")
    both = either = 0
    for seed in range(12):
        blank = model.blank_mask(np.full(SHAPE, 1, np.int8), np.random.default_rng(seed))
        j, h = blank[..., BAND_NAMES.index("J_E")], blank[..., BAND_NAMES.index("H_E")]
        both += int((j & h).sum())
        either += int((j | h).sum())
    assert 0.05 < both / either < 0.9


def test_unknown_release_is_an_error():
    with pytest.raises(ValueError, match="release"):
        BlankPixelModel("dr9")


def test_coverage_map_steps_by_one_across_straight_seams():
    stepped = 0
    for seed in range(60):
        coverage = draw_coverage_map(SHAPE, 4, np.random.default_rng(seed), theta_deg=30.0)
        assert coverage.shape == SHAPE and coverage.min() >= 1
        levels = np.unique(coverage)
        assert np.all(np.diff(levels) == 1)                 # neighbouring regions differ by one exposure
        stepped += len(levels) > 1
    # About half of real 25" crops hold one exposure count, the rest two or more.
    assert 15 <= stepped <= 45


def test_larger_fields_are_crossed_by_more_seams():
    def stepped(shape):
        return sum(len(np.unique(draw_coverage_map(shape, 4, np.random.default_rng(seed), theta_deg=10.0))) > 1
                   for seed in range(60))
    assert stepped((640, 640)) > stepped((128, 128)) + 10


def test_coverage_never_drops_below_one_exposure():
    for seed in range(40):
        assert draw_coverage_map(SHAPE, 1, np.random.default_rng(seed), theta_deg=0.0).min() >= 1


def test_exposure_count_follows_the_release_shares_from_shallow_to_deep():
    """The shallowest sky gets the fewest exposures; over all depths the
    counts reproduce the measured area shares (DR1: 4.9% under one exposure,
    10.9% under two)."""
    model = BlankPixelModel("dr1")
    counts = np.array([model.exposure_count(q) for q in (np.arange(2000) + 0.5) / 2000])
    assert np.all(np.diff(counts) >= 0)
    assert counts[0] == 1 and counts[-1] >= 6
    assert np.mean(counts == 1) == pytest.approx(0.049, abs=0.003)
    assert np.mean(counts == 2) == pytest.approx(0.109, abs=0.003)
    # Q1 has no footprint files; its fields are as often single-exposure as DR1's.
    q1 = BlankPixelModel("q1")
    assert [q1.exposure_count(q) for q in (0.01, 0.5, 0.99)] == [model.exposure_count(q) for q in (0.01, 0.5, 0.99)]


# --------------------------------------------------------------------------- #
# Blank cores of bright stars
# --------------------------------------------------------------------------- #
CORE_SHAPE = (96, 96)


def _source(peak_e: float, band: str, *, fwhm_px: float = 4.5):
    """Trigger planes (H, W, 4) of one round source peaking at ``peak_e`` in ``band``."""
    y, x = np.mgrid[0:CORE_SHAPE[0], 0:CORE_SHAPE[1]]
    profile = np.exp(-((y - 48) ** 2 + (x - 48) ** 2) / (2 * (fwhm_px / 2.355) ** 2))
    trigger = np.zeros(CORE_SHAPE + (4,), np.float32)
    trigger[..., BAND_NAMES.index(band)] = peak_e * profile
    return trigger


def _core_blank(model, trigger, star, n_exposures, seed):
    """Blank cores alone (no persistence, no cosmic rays)."""
    rng = np.random.default_rng(seed)
    exposures = model.core_exposure_bits(trigger, star, n_exposures, rng, theta_deg=float(rng.uniform(0, 360)))
    bits = blank_where_all_invalid(exposures, np.full(CORE_SHAPE, n_exposures, np.int8))
    return np.stack([(bits >> k) & 1 > 0 for k in range(4)], axis=-1)


def _core_rate(release, band, ratio, n_exposures, seeds=40, *, star=True):
    model = BlankPixelModel(release)
    trigger = _source(ratio * model.star_core_level_e(band), band)
    k = BAND_NAMES.index(band)
    hits = sum(bool(_core_blank(model, trigger, trigger if star else np.zeros_like(trigger),
                                n_exposures, seed)[46:51, 46:51, k].any()) for seed in range(seeds))
    return hits / seeds


def test_star_core_levels_follow_the_measured_blanking_brightness():
    """Half of the real star cores are blank near 15 ke- (Q1) and 23 ke- (DR1)
    of predicted NISP peak, and near 1.8 Me- in VIS."""
    q1, dr1 = BlankPixelModel("q1"), BlankPixelModel("dr1")
    for band in BLANK_BAND_NAMES:
        assert 11_000 < q1.star_core_level_e(band) < 18_000
        assert q1.star_core_level_e(band) < dr1.star_core_level_e(band) < 30_000
    assert 1.2e6 < dr1.star_core_level_e("VIS") < 2.4e6


@pytest.mark.parametrize("band", ["VIS", "Y_E", "J_E", "H_E"])
def test_bright_star_core_is_blank_and_faint_one_is_not(band):
    assert _core_rate("dr1", band, 4.0, 4) > 0.9
    assert _core_rate("dr1", band, 0.25, 4) < 0.05


def test_star_at_the_blanking_brightness_is_blank_about_half_the_time():
    for band in ("H_E", "VIS"):
        assert 0.3 < _core_rate("q1", band, 1.0, 4, seeds=120) < 0.7


def test_marginal_star_is_blanked_less_often_under_more_exposures():
    """Every exposure must lose the core, so more exposures save more cores."""
    assert _core_rate("q1", "H_E", 1.0, 2, seeds=150) > _core_rate("q1", "H_E", 1.0, 6, seeds=150) + 0.1


def test_blank_star_core_is_a_round_patch_about_an_arcsecond_across():
    """Real NISP cores: 80-110 pixels near the blanking brightness, filling
    about two thirds of their bounding box (a box would fill it)."""
    model = BlankPixelModel("q1")
    k = BAND_NAMES.index("H_E")
    trigger = _source(3.0 * model.star_core_level_e("H_E"), "H_E")
    areas, fills = [], []
    for seed in range(40):
        blank = _core_blank(model, trigger, trigger, 4, seed)[..., k]
        if not blank.any():
            continue
        ys, xs = np.nonzero(blank)
        areas.append(blank.sum())
        fills.append(blank.sum() / ((np.ptp(ys) + 1) * (np.ptp(xs) + 1)))
        assert abs(ys.mean() - 48) < 3 and abs(xs.mean() - 48) < 3
    assert 50 < np.median(areas) < 160
    assert np.median(fills) < 0.9


def test_only_the_bright_band_loses_its_core():
    model = BlankPixelModel("dr1")
    trigger = _source(5.0 * model.star_core_level_e("J_E"), "J_E")
    blank = _core_blank(model, trigger, trigger, 3, 1)
    assert blank[..., BAND_NAMES.index("J_E")].any()
    assert not blank[..., [0, 1, 3]].any()


def test_galaxy_light_survives_where_a_star_core_would_be_blank():
    """Smooth galaxy light is recorded well above the level that blanks star
    cores; it only loses pixels above the extended level."""
    assert _core_rate("dr1", "H_E", 1.6, 3, star=False) == 0.0
    model = BlankPixelModel("dr1")
    level = model.extended_level_e("H_E")
    assert level > 1.5 * model.star_core_level_e("H_E")
    trigger = _source(2.0 * level, "H_E", fwhm_px=24.0)
    blank = _core_blank(model, trigger, np.zeros_like(trigger), 3, 0)[..., BAND_NAMES.index("H_E")]
    assert blank.any()
    # Only the brightest pixels and the few the bilinear stencil takes with them.
    assert trigger[..., BAND_NAMES.index("H_E")][blank].min() > 0.4 * level


# --------------------------------------------------------------------------- #
# In the observation simulator
# --------------------------------------------------------------------------- #
def _hr_field(seed: int = 0, side: int = 384):
    from euclid_polish.image import Image
    data = np.random.default_rng(seed).uniform(50.0, 150.0, (side, side, 4)).astype(np.float32)
    return Image(data=data, pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
                 band_names=Config.LR_INPUT_BAND_NAMES, is_clean=True)


def _simulator(**kwargs):
    from euclid_polish.sky.observation.observation_simulator import (
        ObservationSimulator,
        ObservationSimulatorConfig,
    )
    return ObservationSimulator(config=ObservationSimulatorConfig(
        add_noise=False, add_artifacts=False, add_saturation=False, **kwargs))


def test_simulator_leaves_fields_untouched_by_default():
    lr, _ = _simulator().process(_hr_field(), np.random.default_rng(3))
    assert not np.any(lr.data == 0.0)


def test_simulator_blanks_only_nisp_pixels_and_only_to_zero():
    off, hr_off = _simulator().process(_hr_field(), np.random.default_rng(3))
    blanked = 0
    for seed in range(6):
        on, hr_on = _simulator(blank_pixels="dr1").process(_hr_field(), np.random.default_rng(seed))
        changed = on.data != off.data
        assert not changed[..., 0].any()                    # VIS is never blanked
        assert np.all(on.data[changed] == 0.0)              # blanks are exact zeros
        np.testing.assert_array_equal(hr_on.data, hr_off.data)   # the target keeps the sky
        blanked += int(changed.sum())
    assert blanked > 0


def test_simulator_rejects_an_unknown_blank_release():
    with pytest.raises(ValueError, match="release"):
        _simulator(blank_pixels="dr9")


# --------------------------------------------------------------------------- #
# Live training: a per-member knob
# --------------------------------------------------------------------------- #
def _live_forward(**kwargs):
    from euclid_polish.psf.psf_library import load_all_band_psf_sets
    from euclid_polish.training.forward_onthefly import OnTheFlyForward
    sets = load_all_band_psf_sets(psf_dir="/nonexistent_dir_for_test")
    kwargs.setdefault("crops_per_field", 1)
    kwargs.setdefault("hr_crop_size", int(Config.DEFAULT_HR_CROP_SIZE) // 2)
    return OnTheFlyForward(
        sets, seed=1, add_artifacts=False, add_saturation=False, inject_stars=False,
        psf_warp_prob=0.0, **kwargs)


def test_live_forward_takes_a_release_or_off():
    for off in ("off", None, ""):
        plain = _live_forward(blank_pixels=off)
        assert plain.blank_pixels is None and plain._sim.config.blank_pixels is None
    holes = _live_forward(blank_pixels="q1")
    assert holes.blank_pixels == "q1" and holes._sim.config.blank_pixels == "q1"
    with pytest.raises(ValueError, match="release"):
        _live_forward(blank_pixels="dr9")


def test_live_forward_crops_carry_blank_nisp_pixels():
    field = np.random.default_rng(0).uniform(50.0, 150.0, (512, 512, 4)).astype(np.float32)
    holes = _live_forward(blank_pixels="dr1", crops_per_field=4, hr_crop_size=384)
    zeros = 0
    for _ in range(6):
        lr, _hr = holes.crops(field)
        assert not np.any(lr[..., 0] == 0.0)
        zeros += int((lr[..., 1:] == 0.0).sum())
    assert zeros > 0


def test_member_spec_cli_and_continue_carry_blank_pixels(tmp_path):
    import json

    from scripts.train_ensemble import _continue_identity, build_specs, parse_args
    base = ["--count", "1", "--steps", "10"]
    spec = build_specs(parse_args(base + ["--member-spec", json.dumps([{"blank_pixels": "dr1"}])]),
                       str(tmp_path / "e1"))[0]
    assert spec.blank_pixels == "dr1"
    cli = build_specs(parse_args(base + ["--blank-pixels", "q1"]), str(tmp_path / "e2"))[0]
    assert cli.blank_pixels == "q1"
    with pytest.raises(SystemExit):
        build_specs(parse_args(base + ["--member-spec", json.dumps([{"blank_pixels": "dr9"}])]),
                    str(tmp_path / "e4"))
    d = tmp_path / "member_320"
    d.mkdir()
    (d / "origin.json").write_text(json.dumps({"blank_pixels": "dr1"}))
    assert _continue_identity("member_320", str(d), {}, {})["blank_pixels"] == "dr1"


class _Recorder:
    def __init__(self, checkpoint_dir, *, scale=2, num_res_blocks=32, seed=None, init_weights_from=None,
                 icnr=False, asinh_knee=None, asinh_knees=None, output_knee=None, learn_output_knee=False):
        self._num_res_blocks = num_res_blocks
        self._asinh_knee = asinh_knee
        self._asinh_knees = tuple(asinh_knees) if asinh_knees else None
        self._learn_output_knee = bool(learn_output_knee)
        self._output_knee = None if learn_output_knee else output_knee
        self.trained: dict = {}

    def train(self, lr, hr, **kwargs):
        self.trained = kwargs


def test_train_members_records_and_passes_blank_pixels(tmp_path, monkeypatch):
    import json

    from euclid_polish import ensemble as ens_mod
    from euclid_polish.ensemble import EnsembleModel, MemberTrainSpec
    monkeypatch.setattr(ens_mod, "Model", _Recorder)
    base = tmp_path / "ensemble"
    ens = EnsembleModel(str(base), _models=[])
    live = {"target_steps": 10, "run_steps": 10, "loss_norm": "l2", "forward_onthefly": True}
    ens.train_members("lr", "hr", [
        MemberTrainSpec(name="member_321", seed=1, blank_pixels="q1", **live),
        MemberTrainSpec(name="member_322", seed=2, blank_pixels="off", **live),
        # Records are fixed: a member trained on them has no blank-pixel setting.
        MemberTrainSpec(name="member_323", seed=3, target_steps=10, run_steps=10, loss_norm="l2"),
    ])
    for k, (name, value) in enumerate((("member_321", "q1"), ("member_322", "off"))):
        assert json.loads((base / name / "origin.json").read_text())["blank_pixels"] == value
        assert ens._models[k].trained["blank_pixels"] == value
    assert "blank_pixels" not in json.loads((base / "member_323" / "origin.json").read_text())
    assert "blank_pixels" not in ens._models[2].trained


def test_simulator_gives_shallow_scenes_fewer_exposures(monkeypatch):
    """The scene's sky noise level sets where it sits between the shallowest
    and deepest real sky, and with that its exposure count."""
    from euclid_polish.sky.observation.mer_noise_levels import load_mer_noise_levels
    sim = _simulator(blank_pixels="dr1")
    seen = []
    monkeypatch.setattr(sim._blank_model, "draw",
                        lambda shape, depth, rng, **kw: seen.append(depth) or np.zeros(tuple(shape) + (4,), bool))
    table = load_mer_noise_levels()
    h = table.levels_e[:, table.bands.index("H_E")]
    lr = np.ones((32, 32, 4), np.float32)
    rng = np.random.default_rng(0)
    sim.apply_blank_pixels(lr, rng, {"H_E": float(h.max()) * 1.01})
    sim.apply_blank_pixels(lr, rng, {"H_E": float(h.min()) * 0.99})
    sim.apply_blank_pixels(lr, rng, None)
    assert seen[0] < 0.01 and seen[1] > 0.99 and 0.0 <= seen[2] < 1.0


def test_simulator_blanks_star_cores_with_the_new_model_not_the_box(monkeypatch):
    """With blank pixels on, saturated cores come from the blank-pixel model
    and the legacy rectangle step is skipped."""
    from euclid_polish.image import Image
    from euclid_polish.sky.observation import observation_simulator as obs_mod
    from euclid_polish.sky.observation.observation_simulator import (
        ObservationSimulator,
        ObservationSimulatorConfig,
    )
    called = []
    monkeypatch.setattr(obs_mod, "apply_saturation_masking", lambda *a, **k: called.append(1))
    data = np.zeros((192, 192, 4), np.float32)
    stars = np.zeros_like(data)
    stars[96, 96, :] = 3e7                      # one very bright star in every band
    hr = Image(data=data, pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
               band_names=Config.LR_INPUT_BAND_NAMES, is_clean=True)
    sim = ObservationSimulator(config=ObservationSimulatorConfig(
        add_noise=False, add_artifacts=False, add_saturation=True, blank_pixels="q1"))
    lr, _ = sim.process(hr, np.random.default_rng(0), star_hr_4ch=stars)
    assert not called
    assert np.all(lr.data[47:49, 47:49, :] == 0.0)          # the core is blank in all four bands
    legacy = ObservationSimulator(config=ObservationSimulatorConfig(
        add_noise=False, add_artifacts=False, add_saturation=True))
    legacy.process(hr, np.random.default_rng(0), star_hr_4ch=stars)
    assert called                                            # without blank pixels the legacy step still runs


def test_simulator_keeps_star_cores_when_saturation_is_off():
    from euclid_polish.image import Image
    data = np.zeros((192, 192, 4), np.float32)
    stars = np.zeros_like(data)
    stars[96, 96, :] = 3e7
    hr = Image(data=data, pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
               band_names=Config.LR_INPUT_BAND_NAMES, is_clean=True)
    lr, _ = _simulator(blank_pixels="q1").process(hr, np.random.default_rng(0), star_hr_4ch=stars)
    assert lr.data[48, 48, 0] > 0.0             # add_saturation=False: VIS (no random blanks) keeps its core


# --------------------------------------------------------------------------- #
# On by default for new live training
# --------------------------------------------------------------------------- #
def test_new_live_training_blanks_pixels_by_default(tmp_path):
    import json

    from euclid_polish.ensemble import MemberTrainSpec
    from scripts.train_ensemble import _continue_identity, build_specs, parse_args
    assert Config.TRAIN_BLANK_PIXELS == "dr1"
    assert _live_forward().blank_pixels == "dr1"
    assert MemberTrainSpec(name="m", seed=1, target_steps=1, run_steps=1, loss_norm="l2").blank_pixels == "dr1"
    base = ["--count", "1", "--steps", "10"]
    assert build_specs(parse_args(base), str(tmp_path / "e1"))[0].blank_pixels == "dr1"
    off = build_specs(parse_args(base + ["--member-spec", json.dumps([{"blank_pixels": "off"}])]),
                      str(tmp_path / "e2"))[0]
    assert off.blank_pixels == "off"
    assert build_specs(parse_args(base + ["--blank-pixels", "off"]), str(tmp_path / "e3"))[0].blank_pixels == "off"


def test_members_trained_before_blank_pixels_continue_without_them(tmp_path):
    import json

    from scripts.train_ensemble import _continue_identity
    d = tmp_path / "member_257"
    d.mkdir()
    (d / "origin.json").write_text(json.dumps({"loss_floor": 0.25}))
    assert _continue_identity("member_257", str(d), {}, {})["blank_pixels"] == "off"


# --------------------------------------------------------------------------- #
# Record generation (train / validate / test sets)
# --------------------------------------------------------------------------- #
def test_record_generation_blanks_pixels_like_live_training_by_default():
    import argparse

    import scripts.run_pipeline as rp
    assert rp._observation_config_from_args(rp.parse_args([])).blank_pixels == Config.TRAIN_BLANK_PIXELS
    assert rp._observation_config_from_args(rp.parse_args(["--blank-pixels", "q1"])).blank_pixels == "q1"
    # "off" keeps the legacy rectangular blackout, as the earlier record sets were made.
    assert rp._observation_config_from_args(rp.parse_args(["--blank-pixels", "off"])).blank_pixels is None
    # A minimal programmatic Namespace gets the default, like the PSF warp.
    assert rp._observation_config_from_args(argparse.Namespace()).blank_pixels == Config.TRAIN_BLANK_PIXELS
    with pytest.raises(ValueError, match="blank_pixels"):
        rp._observation_config_from_args(argparse.Namespace(blank_pixels="dr9"))
    with pytest.raises(SystemExit):
        rp.parse_args(["--blank-pixels", "dr9"])


def test_parallel_generation_hands_the_blank_release_to_its_workers(tmp_path, monkeypatch):
    import inspect

    import scripts.run_pipeline as rp
    from tests.test_run_pipeline_parallel import _InlinePool, _sim_fwd
    real_init = rp._gen_init_worker
    sim, fwd = _sim_fwd()
    builds = []

    def build(*initargs):
        builds.append(initargs)
        rp._W_SIM, rp._W_FWD, rp._W_RECORDS_DIR = sim, fwd, str(tmp_path)

    _InlinePool.instances = []
    for name in ("_W_SIM", "_W_FWD", "_W_RECORDS_DIR"):
        monkeypatch.setattr(rp, name, getattr(rp, name))
    monkeypatch.setattr(rp, "_gen_init_worker", build)
    monkeypatch.setattr(rp, "_fork_context", lambda: object())
    monkeypatch.setattr(rp, "ProcessPoolExecutor", _InlinePool)
    for flag, expected in (([], Config.TRAIN_BLANK_PIXELS), (["--blank-pixels", "off"], "off")):
        builds.clear()
        rp.step_generate_and_convolve_parallel(rp.parse_args([
            "--records-dir", str(tmp_path / (expected or "x")), "--force", "--seed", "7",
            "--ntrain", "2", "--nvalid", "2", "--ntest", "2", "--gen-workers", "2",
            "--image-size", "96", "--galaxy-density-arcmin2", "0",
            "--lens-density-arcmin2", "0", "--star-density-arcmin2", "0", *flag,
        ]))
        bound = inspect.signature(real_init).bind(*builds[0])
        assert bound.arguments["blank_pixels"] == expected
