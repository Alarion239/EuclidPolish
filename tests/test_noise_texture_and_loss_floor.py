"""Two counters to the sim-to-real gap (2026-10-18): a randomised training
noise texture (real Euclid noise is a different texture than our fixed
4-exposure stack) and a floor inside the loss stretch (the unbiased way to
stop rewarding sub-noise detail; noise on the targets would darken faint
light)."""
from __future__ import annotations

import json

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish import ensemble as ens_mod
from euclid_polish.config import Config
from euclid_polish.ensemble import EnsembleModel, MemberTrainSpec
from euclid_polish.model import Model, loss_floor_e
from euclid_polish.psf.psf_library import load_all_band_psf_sets
from euclid_polish.sky.observation import observation_simulator as obs_mod
from euclid_polish.sky.observation.noise import (
    NoiseTextureJitter,
    apply_archive_noise,
    dithered_unit_noise,
    jittered_unit_noise,
    pixel_scatter_factor,
)
from euclid_polish.training.augmentation import asinh_stretch_multi_knee
from euclid_polish.training.forward_onthefly import OnTheFlyForward
from euclid_polish.training.losses import build_loss, knee_floored_loss, knee_stretched_loss
from scripts.train_ensemble import _continue_identity, build_specs, parse_args

KNEES = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
NISP = Config.get_band("H_E")


def _lag1(u):
    return float(np.corrcoef(u[:, 1:].ravel(), u[:, :-1].ravel())[0, 1])


def _plaid(u):
    """Std modulation of the variance over pixel phase mod 3 (NISP detector)."""
    v = np.array([[np.mean(u[py::3, px::3] ** 2) for px in range(3)] for py in range(3)])
    return float(np.sqrt(v.max() / v.min()) - 1.0)


def _large_scale_variance(u, block=24):
    h, w = (s // block * block for s in u.shape)
    sums = u[:h, :w].reshape(h // block, block, w // block, block).sum(axis=(1, 3))
    return float(np.var(sums) / block ** 2)


def test_more_exposures_weaken_the_plaid_and_keep_unit_large_scale_variance():
    rng = np.random.default_rng(1)
    four = [dithered_unit_noise((240, 240), NISP, rng, n_exposures=4) for _ in range(6)]
    sixteen = [dithered_unit_noise((240, 240), NISP, rng, n_exposures=16) for _ in range(6)]
    assert np.median([_plaid(u) for u in sixteen]) < 0.6 * np.median([_plaid(u) for u in four])
    for u in four[:2] + sixteen[:2]:
        assert _large_scale_variance(u) == pytest.approx(1.0, rel=0.2)


def test_jittered_texture_spans_lower_correlation_and_keeps_the_noise_level():
    rng = np.random.default_rng(2)
    fixed = [dithered_unit_noise((240, 240), NISP, rng) for _ in range(6)]
    jitter = NoiseTextureJitter(exposures=(4, 16), white_max=0.15)
    jittered = [jittered_unit_noise((240, 240), NISP, rng, jitter) for _ in range(24)]
    assert min(_lag1(u) for u in jittered) < min(_lag1(u) for u in fixed) - 0.02
    assert np.median([_large_scale_variance(u) for u in jittered]) == pytest.approx(1.0, rel=0.2)
    none = NoiseTextureJitter(exposures=(4, 4), white_max=0.0)
    a = jittered_unit_noise((60, 60), NISP, np.random.default_rng(3), none)
    r = np.random.default_rng(3)
    r.integers(4, 5)                                        # the exposure draw
    r.uniform(0.0, 0.0)                                     # the white-share draw
    np.testing.assert_allclose(a, dithered_unit_noise((60, 60), NISP, r, n_exposures=4))


@pytest.mark.parametrize("bad", [{"exposures": (0, 4)}, {"exposures": (8, 4)}, {"white_max": -0.1},
                                 {"white_max": 1.5}])
def test_noise_texture_jitter_rejects_bad_ranges(bad):
    with pytest.raises(ValueError):
        NoiseTextureJitter(**bad)


def test_apply_archive_noise_uses_the_randomised_texture_only_when_asked():
    signal = np.zeros((90, 90), np.float32)
    plain = apply_archive_noise(signal, NISP, np.random.default_rng(4))
    again = apply_archive_noise(signal, NISP, np.random.default_rng(4))
    np.testing.assert_array_equal(plain, again)
    jittered = apply_archive_noise(signal, NISP, np.random.default_rng(4),
                                   noise_texture=NoiseTextureJitter())
    assert not np.allclose(jittered, plain)


def test_the_training_forward_passes_the_texture_to_the_noise(monkeypatch):
    seen = []
    real = obs_mod.apply_archive_noise

    def spy(*args, **kwargs):
        seen.append(kwargs.get("noise_texture"))
        return real(*args, **kwargs)

    monkeypatch.setattr(obs_mod, "apply_archive_noise", spy)
    sets = load_all_band_psf_sets(psf_dir="/nonexistent_dir_for_test")
    field = np.abs(np.random.default_rng(0).normal(20, 5, (128, 128, 4))).astype(np.float32)
    kw = {"seed": 1, "crops_per_field": 1, "hr_crop_size": int(Config.DEFAULT_HR_CROP_SIZE) // 2,
          "add_artifacts": False, "add_saturation": False, "inject_stars": False, "psf_warp_prob": 0.0}
    OnTheFlyForward(sets, **kw).crops(field)
    assert seen and all(t is None for t in seen)
    seen.clear()
    fwd = OnTheFlyForward(sets, noise_texture_jitter=True, **kw)
    assert fwd.noise_texture_jitter
    fwd.crops(field)
    assert seen and all(isinstance(t, NoiseTextureJitter) for t in seen)
    assert seen[0].exposures == tuple(Config.NOISE_TEXTURE_JITTER_EXPOSURES)


def test_loss_floor_e_is_a_fraction_of_each_bands_pixel_scatter():
    floors = loss_floor_e(1.0)
    for f, name in zip(floors, Config.LR_INPUT_BAND_NAMES, strict=True):
        band = Config.get_band(name)
        assert f == pytest.approx(band.mer_rms_e * pixel_scatter_factor(band) / 4)
    assert loss_floor_e(0.25) == pytest.approx(tuple(0.25 * f for f in floors))
    assert 0.6 < pixel_scatter_factor(Config.get_band("VIS")) < 0.75      # ~2/3
    assert 0.18 < pixel_scatter_factor(NISP) < 0.26                       # ~2/9
    with pytest.raises(ValueError):
        loss_floor_e(-1.0)


def test_the_floored_loss_is_unbiased_zero_floor_is_the_plain_loss():
    rng = np.random.default_rng(5)
    hr = tf.constant(rng.uniform(0.0, 50.0, (1, 6, 6, 4)).astype(np.float32))
    target = asinh_stretch_multi_knee(hr, KNEES)
    floored = knee_floored_loss(build_loss("l2"), KNEES, (0.5, 0.1, 0.1, 0.1), 4)
    assert float(floored(hr, target)) == pytest.approx(0.0, abs=1e-6)      # perfect = 0
    sr = hr * 1.1
    np.testing.assert_allclose(float(knee_floored_loss(build_loss("l2"), KNEES, (0, 0, 0, 0), 4)(sr, target)),
                               float(knee_stretched_loss(build_loss("l2"), KNEES)(sr, target)), rtol=1e-5)
    with pytest.raises(ValueError):
        knee_floored_loss(build_loss("l2"), KNEES, (0.1, 0.1), 4)


def test_the_floor_stops_rewarding_sub_noise_detail_but_not_real_light():
    def loss_of(sr_val, hr_val, floor):
        hr = tf.fill((1, 4, 4, 4), hr_val)
        sr = tf.fill((1, 4, 4, 4), sr_val)
        return float(knee_floored_loss(build_loss("l2"), KNEES, (floor,) * 4, 4)(
            sr, asinh_stretch_multi_knee(hr, KNEES)))
    # 0 vs 0.01 e-: a big penalty without a floor, ~40x smaller with a 0.2 e- floor.
    assert loss_of(0.0, 0.01, 0.2) < 0.05 * loss_of(0.0, 0.01, 0.0)
    # 5 vs 10 e- (real light): barely changed by the floor.
    assert loss_of(5.0, 10.0, 0.2) == pytest.approx(loss_of(5.0, 10.0, 0.0), rel=0.15)


def _member(tmp_path):
    return Model(str(tmp_path / "m"), scale=2, num_res_blocks=1, asinh_knees=KNEES,
                 output_knee=0.1, learn_output_knee=True)


def test_member_loss_with_a_floor_and_loss_knees(tmp_path):
    m = _member(tmp_path)
    rng = np.random.default_rng(6)
    hr = tf.constant(rng.uniform(0.0, 30.0, (1, 8, 8, 4)).astype(np.float32))
    target = asinh_stretch_multi_knee(hr, KNEES)
    sr = hr * 0.9
    floor = loss_floor_e(0.25)
    six = KNEES[2:]
    got = m._member_loss("l2", "plain", six, 0.25)(sr, target)
    expected = knee_floored_loss(build_loss("l2"), six, floor, 4)(sr, asinh_stretch_multi_knee(hr, six))
    np.testing.assert_allclose(float(got), float(expected), rtol=1e-5)
    fixed = Model(str(tmp_path / "f"), scale=2, num_res_blocks=1, asinh_knees=KNEES, output_knee=10.0)
    with pytest.raises(ValueError):
        fixed._member_loss("l2", "plain", None, 0.25)


def test_member_spec_cli_and_continue_carry_both_knobs(tmp_path):
    member = {"asinh_knees": list(KNEES), "learn_output_knee": True, "loss_floor": 0.25,
              "noise_texture_jitter": True}
    spec = build_specs(parse_args(["--count", "1", "--steps", "10", "--member-spec", json.dumps([member])]),
                       str(tmp_path / "e1"))[0]
    assert spec.loss_floor == 0.25 and spec.noise_texture_jitter
    cli = build_specs(parse_args(["--count", "1", "--steps", "10", "--asinh-knees", ",".join(map(str, KNEES)),
                                  "--learn-output-knee", "--loss-floor", "0.5", "--noise-texture-jitter"]),
                      str(tmp_path / "e2"))[0]
    assert cli.loss_floor == 0.5 and cli.noise_texture_jitter
    plain = build_specs(parse_args(["--count", "1", "--steps", "10"]), str(tmp_path / "e3"))[0]
    assert plain.loss_floor is None and not plain.noise_texture_jitter
    for bad in ({**member, "loss_floor": -1}, {**member, "noise_texture_jitter": 1}):
        with pytest.raises(SystemExit):
            build_specs(parse_args(["--count", "1", "--steps", "10", "--member-spec", json.dumps([bad])]),
                        str(tmp_path / "e4"))
    d = tmp_path / "member_310"
    d.mkdir()
    (d / "origin.json").write_text(json.dumps({"asinh_knees": list(KNEES), "loss_floor": 0.25,
                                               "noise_texture_jitter": True}))
    identity = _continue_identity("member_310", str(d), {}, {})
    assert identity["loss_floor"] == 0.25 and identity["noise_texture_jitter"] is True


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


def test_train_members_records_and_passes_both_knobs(tmp_path, monkeypatch):
    monkeypatch.setattr(ens_mod, "Model", _Recorder)
    base = tmp_path / "ensemble"
    spec = MemberTrainSpec(name="member_311", seed=1, target_steps=10, run_steps=10, loss_norm="l2",
                           asinh_knees=KNEES, knee_loss="balanced", output_knee=0.1, learn_output_knee=True,
                           loss_floor=0.25, noise_texture_jitter=True)
    ens = EnsembleModel(str(base), _models=[])
    ens.train_members("lr", "hr", [spec])
    origin = json.loads((base / "member_311" / "origin.json").read_text())
    assert origin["loss_floor"] == 0.25 and origin["noise_texture_jitter"] is True
    assert "learned_output_knee" in origin
    kw = ens._models[0].trained
    assert kw["loss_floor"] == 0.25 and kw["noise_texture_jitter"] is True
