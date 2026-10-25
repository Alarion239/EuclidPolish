"""The brightness-consistency penalty (``brightness_consistency``): the sky
re-observed as Euclid would must match the truth within Euclid's noise at
every block scale; the trainer couples it to the loss without a weight."""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.model import Model
from euclid_polish.sky.observation.noise import pixel_scatter_factor
from euclid_polish.training.brightness import (
    BrightnessConsistency,
    block_noise_factors,
    core_kernel,
)
from euclid_polish.training.models.wdsr import wdsr
from euclid_polish.training.trainer import Trainer

KNEES = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
WHITE = (1.0, 0.5, 0.25, 0.125)          # block-mean scatter of white noise at 1/2/4/8 px


def _gauss(size=9, sigma=1.5):
    y, x = np.mgrid[:size, :size] - size // 2
    kernel = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def _penalty(n_bands=1):
    return BrightnessConsistency([_gauss()] * n_bands, [1.0] * n_bands, [WHITE] * n_bands)


def _img(a):
    return tf.constant(np.asarray(a, np.float32)[None, ..., None])


def test_block_noise_starts_at_the_pixel_scatter_and_falls_with_the_block():
    for name in ("VIS", "H_E"):
        band = Config.get_band(name)
        factors = block_noise_factors(band)
        assert factors[0] == pytest.approx(pixel_scatter_factor(band), rel=0.05)
        assert list(factors) == sorted(factors, reverse=True)


def test_the_core_kernel_is_a_centred_unit_sum_cutout():
    big = _gauss(31, 3.0)
    core = core_kernel(big, 9)
    assert core.shape == (9, 9) and core.sum() == pytest.approx(1.0)
    assert np.unravel_index(core.argmax(), core.shape) == (4, 4)
    with pytest.raises(ValueError, match="odd"):
        core_kernel(big, 8)
    with pytest.raises(ValueError, match="smaller"):
        core_kernel(big, 41)


def test_reobserving_blurs_and_sum_rebins_without_losing_flux():
    image = np.zeros((48, 48))
    image[24, 24] = 100.0
    out = _penalty().reobserve(_img(image)).numpy()
    assert out.shape == (1, 20, 20, 1)              # (48 - 9 + 1) // 2
    assert out.sum() == pytest.approx(100.0, rel=1e-4)


def test_a_perfect_model_and_errors_inside_the_noise_cost_nothing():
    rng = np.random.default_rng(0)
    truth = np.abs(rng.normal(size=(48, 48))) * 20.0
    penalty = _penalty()
    assert float(penalty.penalty(_img(truth), _img(truth))) == 0.0
    close = truth + rng.normal(scale=0.1, size=truth.shape)
    assert float(penalty.penalty(_img(close), _img(truth))) == 0.0


def test_a_missing_core_costs_and_photon_noise_sets_the_error_bar():
    penalty = _penalty()

    def five_percent_short(level):
        truth = np.zeros((48, 48))
        truth[22:26, 22:26] = level
        return float(penalty.penalty(_img(0.95 * truth), _img(truth)))

    truth = np.zeros((48, 48))
    truth[22:26, 22:26] = 1e4
    holed = truth.copy()
    holed[22:26, 22:26] = 0.0
    assert float(penalty.penalty(_img(holed), _img(truth))) > 10.0 * five_percent_short(1e4)
    # Euclid measures a bright core to ~1 %, a faint one to worse than 100 %.
    assert five_percent_short(1e4) > 0.0
    assert five_percent_short(1.0) == 0.0


def test_the_charge_is_quadratic_just_outside_the_band_and_linear_far_outside():
    penalty = BrightnessConsistency([np.ones((1, 1))], [1.0], [(1.0,)], scales=(1,), rebin=1)
    truth = tf.zeros((1, 1, 1, 1))

    def charge(z):
        return float(penalty.penalty(tf.fill((1, 1, 1, 1), float(z)), truth))

    assert charge(0.5) == 0.0 and charge(1.0) == 0.0
    assert charge(1.5) == pytest.approx(0.5 * 0.5 ** 2)
    assert charge(2.0) == pytest.approx(0.5)
    assert charge(1001.0) == pytest.approx(999.5)            # linear: e − 1/2
    sr = tf.Variable(tf.fill((1, 1, 1, 1), 1e6))
    with tf.GradientTape() as tape:
        value = penalty.penalty(sr, truth)
    assert float(tf.abs(tape.gradient(value, sr))[0, 0, 0, 0]) == pytest.approx(1.0)


def test_block_means_catch_an_extended_deficit_that_single_pixels_miss():
    truth = np.full((96, 96), 4.0)
    dimmer = 0.9 * truth
    penalty = _penalty()
    z = penalty.z_scores(_img(dimmer), _img(truth))
    assert np.abs(z[0].numpy()).max() < 1.0          # every pixel within its error bar
    assert np.abs(z[-1].numpy()).min() > 1.0         # the 8 × 8 means are not
    assert float(penalty.penalty(_img(dimmer), _img(truth))) > 0.0


def test_inputs_are_checked():
    with pytest.raises(ValueError, match="one entry per band"):
        BrightnessConsistency([_gauss()], [1.0, 1.0], [WHITE])
    with pytest.raises(ValueError, match="odd square"):
        BrightnessConsistency([np.ones((8, 8))], [1.0], [WHITE])
    with pytest.raises(ValueError, match="positive factors"):
        BrightnessConsistency([_gauss()], [1.0], [(1.0, 0.5)])
    with pytest.raises(ValueError, match="sky_rms_e"):
        BrightnessConsistency([_gauss()], [0.0], [WHITE])


# -- the trainer ------------------------------------------------------------------

def _electron_model(seed=7):
    tf.keras.utils.set_random_seed(seed)
    return wdsr(scale=2, nchan_in=4, nchan_out=4, num_res_blocks=1, num_filters=4)


def _identity(lr, hr):
    return lr, hr


def _trainer(path, brightness=None):
    return Trainer(_electron_model(), loss=lambda sr, hr: tf.reduce_mean(tf.square(sr - hr)),
                   checkpoint_dir=str(path), learning_rate=1e-3, output_electrons=True,
                   batch_stretch=_identity, brightness=brightness)


def test_the_trainer_adds_the_penalty_to_the_gradient_not_to_the_logged_loss(tmp_path):
    rng = np.random.default_rng(0)
    lr = tf.constant(rng.normal(size=(2, 32, 32, 4)).astype(np.float32))
    hr = tf.constant((np.abs(rng.normal(size=(2, 64, 64, 4))) * 50.0).astype(np.float32))
    plain = _trainer(tmp_path / "plain")
    held = _trainer(tmp_path / "held", _penalty(4))
    plain_losses = [float(plain.train_step(lr, hr)[0]) for _ in range(2)]
    held_losses = [float(held.train_step(lr, hr)[0]) for _ in range(2)]
    assert held_losses[0] == plain_losses[0]            # same weights at the first step
    assert any(not np.array_equal(a, b) for a, b in
               zip(plain.model.get_weights(), held.model.get_weights(), strict=True))
    assert held._read_window()["brightness"] > 0.0
    assert plain._read_window()["brightness"] == 0.0


def test_the_trainer_needs_electrons_in_and_out(tmp_path):
    with pytest.raises(ValueError, match="electrons"):
        Trainer(_electron_model(), checkpoint_dir=str(tmp_path / "a"), output_electrons=True,
                brightness=_penalty(4))
    with pytest.raises(ValueError, match="electrons"):
        Trainer(_electron_model(), checkpoint_dir=str(tmp_path / "b"), batch_stretch=_identity,
                brightness=_penalty(4))


def test_model_train_checks_the_knob_before_building_anything(tmp_path):
    learned = Model(str(tmp_path / "l"), scale=2, num_res_blocks=1, asinh_knees=KNEES,
                    output_knee=10.0, learn_output_knee=True)
    with pytest.raises(ValueError, match="live"):
        learned.train("missing_lr", "missing_hr", steps=1, brightness_consistency=True)
    fixed = Model(str(tmp_path / "f"), scale=2, num_res_blocks=1, asinh_knees=KNEES, output_knee=10.0)
    with pytest.raises(ValueError, match="learned-output-knee"):
        fixed.train("missing_lr", "missing_hr", steps=1, brightness_consistency=True,
                    forward_onthefly=True)
