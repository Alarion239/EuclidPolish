"""The residual-isotropy penalty (2026-10-20): on average over recent batches
a member's error must not prefer a direction, which the pixel-axis grid
does. Isotropic errors cost nothing whatever their spectrum, a grid costs,
the gradient removes the grid and keeps the isotropic error, the moving
average travels with the checkpoint, and the knob reaches a member from the
CLI and member specs."""
from __future__ import annotations

import json

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish import ensemble as ens_mod
from euclid_polish.ensemble import EnsembleModel, MemberTrainSpec
from euclid_polish.model import Model
from euclid_polish.training.isotropy import (
    DEFAULT_KNEE_E,
    DEFAULT_SCALES,
    LEGACY_SCALES,
    MultiScaleIsotropy,
    ResidualIsotropy,
)
from euclid_polish.training.models.wdsr import wdsr
from euclid_polish.training.trainer import Trainer
from scripts.train_ensemble import _continue_identity, build_specs, parse_args

KNEES = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def _isotropic_field(alpha: float, n: int, size: int = 256, seed: int = 0) -> np.ndarray:
    """``n`` Gaussian random fields with power spectrum ``|k|**alpha``."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(size)
    k = np.hypot(*np.meshgrid(f, f, indexing="ij"))
    amplitude = np.where(k > 0, np.maximum(k, 1e-12) ** (alpha / 2.0), 0.0)
    white = np.fft.fft2(rng.standard_normal((n, size, size)))
    field = np.fft.ifft2(white * amplitude).real
    return (field / field.std()).astype(np.float32)[..., None]


def _penalty_of(residual: np.ndarray, iso: ResidualIsotropy) -> float:
    """The penalty of a residual given in stretched units (target 0)."""
    sr = iso.knee_e * np.sinh(residual)
    return float(iso.penalty(tf.constant(sr), tf.zeros_like(tf.constant(sr))))


@pytest.mark.parametrize("alpha", [0.0, -3.0])
def test_isotropic_errors_cost_nothing_whatever_their_spectrum(alpha):
    # A plain ring mean would see a steep isotropic spectrum as 4-fold
    # anisotropic by tens of percent; the interpolated reference does not.
    iso = ResidualIsotropy(n_bands=1, decay=0.0)
    assert _penalty_of(_isotropic_field(alpha, 64), iso) < 2e-3


def test_a_pixel_axis_grid_costs():
    iso = ResidualIsotropy(n_bands=1, decay=0.0)
    field = _isotropic_field(-1.0, 64, seed=1)
    yy, xx = np.mgrid[0:256, 0:256]
    grid = (np.sin(2 * np.pi * xx / 6.0) + np.sin(2 * np.pi * yy / 9.0))[None, :, :, None]
    assert _penalty_of(field + 0.4 * grid.astype(np.float32), iso) > 0.03
    assert iso.excess()[0] > 0.03


def test_the_penalty_starts_isotropic_and_learns_on_average():
    iso = ResidualIsotropy(n_bands=1, decay=0.9)
    np.testing.assert_array_equal(iso.mean.numpy(), 1.0)
    assert iso.excess()[0] == 0.0
    yy, xx = np.mgrid[0:128, 0:128]
    grid = np.sin(2 * np.pi * xx / 6.0)[None, :, :, None].astype(np.float32)
    rng = np.random.default_rng(2)
    first = _penalty_of(rng.standard_normal((4, 128, 128, 1)).astype(np.float32) + grid, iso)
    for _ in range(30):
        later = _penalty_of(rng.standard_normal((4, 128, 128, 1)).astype(np.float32) + grid, iso)
    assert 0.0 < first < later                          # one batch moves the average 10 %


def test_the_gradient_removes_the_grid_and_keeps_the_isotropic_error():
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:128, 0:128]
    grid = (np.sin(2 * np.pi * xx / 6.0) + np.sin(2 * np.pi * yy / 9.0)).astype(np.float32)
    noise = rng.standard_normal((8, 128, 128, 1)).astype(np.float32)
    iso = ResidualIsotropy(n_bands=1, decay=0.9)
    residual = tf.Variable(noise + 0.4 * grid[None, :, :, None])
    opt = tf.keras.optimizers.Adam(0.02)

    def amplitude(x):
        return float((x[..., 0] * grid).mean() / (grid * grid).mean())

    penalties = []
    for _ in range(150):
        with tf.GradientTape() as tape:
            p = iso.penalty(iso.knee_e * tf.sinh(residual), tf.zeros_like(residual))
        opt.apply_gradients([(tape.gradient(p, residual), residual)])
        penalties.append(float(p))
    x = residual.numpy()
    assert amplitude(x) < 0.5 * 0.4
    assert (x - amplitude(x) * grid[None, :, :, None]).var() > 0.9 * noise.var()
    assert np.mean(penalties[-20:]) < 0.5 * penalties[0]


def test_bands_are_scored_separately_and_images_must_hold_a_window():
    iso = ResidualIsotropy(n_bands=2, decay=0.0)
    field = _isotropic_field(0.0, 64, seed=4)
    yy, xx = np.mgrid[0:256, 0:256]
    striped = field + 0.5 * np.sin(2 * np.pi * xx / 5.0)[None, :, :, None].astype(np.float32)
    _penalty_of(np.concatenate([field, striped], axis=-1), iso)
    clean, grid = iso.excess()
    assert clean < 2e-3 < 0.03 < grid
    with pytest.raises(ValueError, match="window"):
        _penalty_of(np.zeros((1, 32, 32, 2), np.float32), iso)
    with pytest.raises(ValueError):
        ResidualIsotropy(knee_e=0.0)


# -- the trainer ---------------------------------------------------------------

def _electron_model(seed=7):
    tf.keras.utils.set_random_seed(seed)
    return wdsr(scale=2, nchan_in=4, nchan_out=4, num_res_blocks=1, num_filters=4)


def _electron_batch(seed=0):
    rng = np.random.default_rng(seed)
    lr = tf.constant(rng.normal(size=(2, 64, 64, 4)).astype(np.float32))
    hr = tf.constant(np.abs(rng.normal(size=(2, 128, 128, 4))).astype(np.float32))
    return lr, hr


def _identity(lr, hr):
    return lr, hr


def _trainer(path, *, isotropy=None, weight=0.0, model=None):
    return Trainer(model or _electron_model(), loss=lambda sr, hr: tf.reduce_mean(tf.square(sr - hr)),
                   checkpoint_dir=str(path), learning_rate=1e-3, output_electrons=True,
                   batch_stretch=_identity, isotropy=isotropy, isotropy_weight=weight)


def test_the_trainer_adds_the_penalty_to_the_gradient_not_to_the_logged_loss(tmp_path):
    lr, hr = _electron_batch()
    plain = _trainer(tmp_path / "plain")
    weightless = _trainer(tmp_path / "zero", isotropy=ResidualIsotropy(), weight=0.0)
    weighted = _trainer(tmp_path / "one", isotropy=ResidualIsotropy(), weight=1.0)
    losses = {}
    for name, trainer in (("plain", plain), ("zero", weightless), ("one", weighted)):
        losses[name] = [float(trainer.train_step(lr, hr)[0]) for _ in range(3)]
    assert losses["plain"] == losses["zero"]
    for a, b in zip(plain.model.get_weights(), weightless.model.get_weights(), strict=True):
        np.testing.assert_array_equal(a, b)
    assert losses["one"][0] == losses["plain"][0]       # same weights at the first step
    assert any(not np.array_equal(a, b) for a, b in
               zip(plain.model.get_weights(), weighted.model.get_weights(), strict=True))
    window = weighted._read_window()
    assert window["isotropy"] > 0.0
    assert plain._read_window()["isotropy"] == 0.0
    assert not np.array_equal(weighted._isotropy.mean.numpy(), 1.0)


def test_the_penalty_needs_electrons_in_and_out(tmp_path):
    with pytest.raises(ValueError, match="electrons"):
        Trainer(_electron_model(), checkpoint_dir=str(tmp_path / "a"), output_electrons=True,
                isotropy=ResidualIsotropy(), isotropy_weight=1.0)
    with pytest.raises(ValueError, match="electrons"):
        Trainer(_electron_model(), checkpoint_dir=str(tmp_path / "b"), batch_stretch=_identity,
                isotropy=ResidualIsotropy(), isotropy_weight=1.0)


def test_the_moving_average_travels_with_the_checkpoint(tmp_path):
    lr, hr = _electron_batch()
    first = _trainer(tmp_path / "m", isotropy=ResidualIsotropy(), weight=1.0)
    for _ in range(2):
        first.train_step(lr, hr)
    first.checkpoint_manager.save()
    saved = first._isotropy.mean.numpy()
    resumed = _trainer(tmp_path / "m", isotropy=ResidualIsotropy(), weight=1.0)
    np.testing.assert_array_equal(resumed._isotropy.mean.numpy(), saved)
    # A checkpoint without the penalty resumes with it (an isotropic start),
    # and one with it loads into a trainer without it.
    plain = _trainer(tmp_path / "p")
    plain.train_step(lr, hr)
    plain.checkpoint_manager.save()
    added = _trainer(tmp_path / "p", isotropy=ResidualIsotropy(), weight=1.0)
    assert int(added.checkpoint.step.numpy()) == 1
    np.testing.assert_array_equal(added._isotropy.mean.numpy(), 1.0)
    dropped = _trainer(tmp_path / "m")
    assert int(dropped.checkpoint.step.numpy()) == 2


# -- the member knob -------------------------------------------------------------

def test_model_train_checks_the_knob_before_building_anything(tmp_path):
    learned = Model(str(tmp_path / "l"), scale=2, num_res_blocks=1, asinh_knees=KNEES,
                    output_knee=10.0, learn_output_knee=True)
    with pytest.raises(ValueError, match="live"):
        learned.train("missing_lr", "missing_hr", steps=1, isotropy_penalty=0.1)
    with pytest.raises(ValueError, match="non-negative"):
        learned.train("missing_lr", "missing_hr", steps=1, isotropy_penalty=-1.0)
    fixed = Model(str(tmp_path / "f"), scale=2, num_res_blocks=1, asinh_knees=KNEES, output_knee=10.0)
    with pytest.raises(ValueError, match="learned-output-knee"):
        fixed.train("missing_lr", "missing_hr", steps=1, isotropy_penalty=0.1, forward_onthefly=True)


def test_member_spec_cli_and_continue_carry_the_knob(tmp_path):
    member = {"asinh_knees": list(KNEES), "learn_output_knee": True, "isotropy_penalty": 0.5}
    spec = build_specs(parse_args(["--count", "1", "--steps", "10", "--member-spec", json.dumps([member])]),
                       str(tmp_path / "e1"))[0]
    assert spec.isotropy_penalty == 0.5 and spec.isotropy_knee is None
    cli = build_specs(parse_args(["--count", "1", "--steps", "10", "--asinh-knees", ",".join(map(str, KNEES)),
                                  "--learn-output-knee", "--isotropy-penalty", "0.2",
                                  "--isotropy-knee", "0.05"]), str(tmp_path / "e2"))[0]
    assert cli.isotropy_penalty == 0.2 and cli.isotropy_knee == 0.05
    plain = build_specs(parse_args(["--count", "1", "--steps", "10"]), str(tmp_path / "e3"))[0]
    assert plain.isotropy_penalty == 0.0 and plain.isotropy_knee is None
    for bad in ({**member, "isotropy_penalty": -1}, {**member, "isotropy_penalty": "x"},
                {**member, "isotropy_knee": 0}, {"asinh_knees": list(KNEES), "isotropy_knee": 0.1}):
        with pytest.raises(SystemExit):
            build_specs(parse_args(["--count", "1", "--steps", "10", "--member-spec", json.dumps([bad])]),
                        str(tmp_path / "e4"))
    d = tmp_path / "member_312"
    d.mkdir()
    (d / "origin.json").write_text(json.dumps({"asinh_knees": list(KNEES), "isotropy_penalty": 0.5,
                                               "isotropy_knee": 0.01}))
    identity = _continue_identity("member_312", str(d), {}, {})
    assert identity["isotropy_penalty"] == 0.5 and identity["isotropy_knee"] == 0.01
    assert identity["isotropy_scales"] == LEGACY_SCALES          # recorded before the scales were
    kept = _continue_identity("member_312", str(d), {}, {"isotropy_penalty": 0.0})
    assert "isotropy_penalty" not in kept
    (d / "origin.json").write_text(json.dumps({"asinh_knees": list(KNEES), "isotropy_penalty": 0.5,
                                               "isotropy_knee": 0.01, "isotropy_scales": [[64, 2, 21]]}))
    assert _continue_identity("member_312", str(d), {}, {})["isotropy_scales"] == ((64.0, 2.0, 21.0, 1.0),)


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


def test_train_members_records_and_passes_the_knob(tmp_path, monkeypatch):
    monkeypatch.setattr(ens_mod, "Model", _Recorder)
    base = tmp_path / "ensemble"
    on = MemberTrainSpec(name="member_313", seed=1, target_steps=10, run_steps=10, loss_norm="l2",
                         asinh_knees=KNEES, knee_loss="balanced", output_knee=10.0, learn_output_knee=True,
                         isotropy_penalty=0.5)
    off = MemberTrainSpec(name="member_314", seed=2, target_steps=10, run_steps=10, loss_norm="l2",
                          asinh_knees=KNEES, knee_loss="balanced", output_knee=10.0, learn_output_knee=True)
    ens = EnsembleModel(str(base), _models=[])
    ens.train_members("lr", "hr", [on, off])
    origin = json.loads((base / "member_313" / "origin.json").read_text())
    assert origin["isotropy_penalty"] == 0.5 and origin["isotropy_knee"] == DEFAULT_KNEE_E
    assert origin["isotropy_scales"] == [list(scale) for scale in DEFAULT_SCALES]
    assert "isotropy_penalty" not in json.loads((base / "member_314" / "origin.json").read_text())
    kw_on, kw_off = (m.trained for m in ens._models)
    assert kw_on["isotropy_penalty"] == 0.5 and kw_on["isotropy_knee"] is None
    assert kw_on["isotropy_scales"] is None
    assert "isotropy_penalty" not in kw_off


# -- the range down to the Nyquist period, and several scales ---------------------

@pytest.mark.parametrize("alpha", [0.0, -3.0])
def test_isotropic_errors_cost_nothing_down_to_the_nyquist_period(alpha):
    iso = ResidualIsotropy(n_bands=1, decay=0.0, period_min=2.0)
    assert _penalty_of(_isotropic_field(alpha, 64), iso) < 2e-3


def test_alternating_columns_cost_once_the_range_reaches_two_pixels():
    field = _isotropic_field(0.0, 64, seed=5)
    columns = 0.3 * (-1.0) ** np.arange(256)[None, None, :, None]       # period 2 px along x
    striped = (field + columns).astype(np.float32)
    assert _penalty_of(striped, ResidualIsotropy(n_bands=1, decay=0.0, period_min=3.0)) < 2e-3
    assert _penalty_of(striped, ResidualIsotropy(n_bands=1, decay=0.0, period_min=2.0)) > 2e-3


def test_long_periods_need_a_window_that_holds_them():
    ResidualIsotropy(window=128, period_min=21.0, period_max=42.0)
    with pytest.raises(ValueError, match="too long"):
        ResidualIsotropy(window=64, period_min=21.0, period_max=42.0)


def test_scales_average_and_each_keeps_its_moving_average(tmp_path):
    rng = np.random.default_rng(6)
    sr = rng.normal(0.0, 0.05, (4, 256, 256, 2)).astype(np.float32)
    sr[..., ::3, :, 0] += 0.05                                            # rows every 3 px in band 0
    hr = np.zeros_like(sr)
    multi = MultiScaleIsotropy(DEFAULT_SCALES, n_bands=2, decay=0.0)
    alone = [ResidualIsotropy(window=int(w), period_min=a, period_max=b, n_bands=2, decay=0.0)
             for w, a, b, _ in DEFAULT_SCALES]
    weights = [weight for *_, weight in DEFAULT_SCALES]
    got = float(multi.penalty(tf.constant(sr), tf.constant(hr)))
    expected = np.average([float(iso.penalty(tf.constant(sr), tf.constant(hr))) for iso in alone], weights=weights)
    assert got == pytest.approx(expected, rel=1e-5)
    np.testing.assert_allclose(multi.excess(), np.average([iso.excess() for iso in alone], axis=0,
                                                          weights=weights), rtol=1e-5)
    assert multi.window == 128
    for text in ("3-21 px x1", "2-3 px x0.003", "21-42 px x1"):
        assert text in multi.describe()
    assert "64px 2-3" in multi.summary()
    with pytest.raises(ValueError, match="positive"):
        MultiScaleIsotropy(((64, 3, 21, 0.0),))
    path = tf.train.Checkpoint(isotropy=multi).save(str(tmp_path / "ckpt"))
    fresh = MultiScaleIsotropy(DEFAULT_SCALES, n_bands=2, decay=0.0)
    tf.train.Checkpoint(isotropy=fresh).restore(path).assert_consumed()
    for a, b in zip(multi.scales, fresh.scales, strict=True):
        np.testing.assert_array_equal(a.mean.numpy(), b.mean.numpy())


def test_the_scales_knob_from_cli_and_member_spec(tmp_path):
    member = {"asinh_knees": list(KNEES), "learn_output_knee": True, "isotropy_penalty": 0.2,
              "isotropy_scales": [[64, 2, 21], [128, 21, 42]]}
    spec = build_specs(parse_args(["--count", "1", "--steps", "10", "--member-spec", json.dumps([member])]),
                       str(tmp_path / "s1"))[0]
    assert spec.isotropy_scales == ((64.0, 2.0, 21.0, 1.0), (128.0, 21.0, 42.0, 1.0))
    cli = build_specs(parse_args(["--count", "1", "--steps", "10", "--asinh-knees", ",".join(map(str, KNEES)),
                                  "--learn-output-knee", "--isotropy-penalty", "0.2",
                                  "--isotropy-scales", "64:3:21,64:2:3:0.01"]), str(tmp_path / "s2"))[0]
    assert cli.isotropy_scales == ((64.0, 3.0, 21.0, 1.0), (64.0, 2.0, 3.0, 0.01))
    for bad in ({**member, "isotropy_scales": [[64, 1, 21]]}, {**member, "isotropy_scales": [[64, 21, 42]]},
                {**member, "isotropy_scales": [[64, 3, 21, 0]]}, {**member, "isotropy_scales": "x"},
                {"asinh_knees": list(KNEES), "isotropy_scales": [[64, 2, 21]]}):
        with pytest.raises(SystemExit):
            build_specs(parse_args(["--count", "1", "--steps", "10", "--member-spec", json.dumps([bad])]),
                        str(tmp_path / "s3"))


def test_model_train_needs_crops_that_hold_the_largest_window(tmp_path):
    learned = Model(str(tmp_path / "l"), scale=2, num_res_blocks=1, asinh_knees=KNEES,
                    output_knee=10.0, learn_output_knee=True)
    with pytest.raises(ValueError, match="fit"):
        learned.train("missing_lr", "missing_hr", steps=1, isotropy_penalty=0.1, forward_onthefly=True,
                      hr_crop_size=96)

