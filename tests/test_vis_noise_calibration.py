"""Focused tests for the immutable delivered-MER VIS noise model."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import euclid_polish.sky.observation.observation_simulator as observation_module
from euclid_polish.config import Config
from euclid_polish.sky.observation.noise import (
    apply_archive_noise,
    apply_band_noise,
)
from euclid_polish.sky.observation.noise_calibration import VISNoiseCalibration
from euclid_polish.sky.observation.observation_simulator import (
    ObservationSimulator,
    ObservationSimulatorConfig,
)


def _calibration(
    *, residual_scale: float = 11.0, owns_field_scale: bool = True,
) -> VISNoiseCalibration:
    return VISNoiseCalibration.build(
        coloring_kernel=((0.0, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.0)),
        residual_scale=residual_scale,
        owns_field_scale=owns_field_scale,
        source_release="Euclid-Q1-MER",
        estimator_version="source-masked-mad-v1",
    )


def test_vis_noise_calibration_round_trips_and_is_immutable():
    calibration = _calibration()

    assert len(calibration.fingerprint) == 64
    assert VISNoiseCalibration.from_payload(calibration.to_payload()) == calibration
    with pytest.raises(FrozenInstanceError):
        calibration.residual_scale = 9.0  # type: ignore[misc]


def test_vis_noise_calibration_rejects_tampering_and_schema_extras():
    payload = _calibration().to_payload()
    payload["residual_scale"] = 13.0
    with pytest.raises(ValueError, match="fingerprint does not match"):
        VISNoiseCalibration.from_payload(payload)

    payload = _calibration().to_payload()
    payload["review_note"] = "not part of the runtime model"
    with pytest.raises(ValueError, match="invalid VIS noise calibration schema"):
        VISNoiseCalibration.from_payload(payload)


@pytest.mark.parametrize(
    "quantiles, message",
    [
        ((0.8, 1.0), "exactly five"),
        ((0.8, 0.9, 1.0, 0.95, 1.2), "nondecreasing"),
        ((0.8, 0.9, 1.1, 1.2, 1.3), "normalized near one"),
        ((0.0, 0.9, 1.0, 1.1, 1.2), "finite and positive"),
    ],
)
def test_vis_noise_calibration_rejects_invalid_field_quantiles(
    quantiles, message,
):
    with pytest.raises(ValueError, match=message):
        VISNoiseCalibration.build(
            coloring_kernel=((1.0,),),
            residual_scale=10.0,
            field_scale_quantiles=quantiles,
            source_release="Euclid-Q1-MER",
            estimator_version="test-v1",
        )


@pytest.mark.parametrize(
    "kernel, message",
    [
        (((1.0, 0.0),), "dimensions must both be odd"),
        (((1.0, 1.0, 1.0),), "must be L2-normalized"),
        (((0.0, np.nan, 0.0),), "only finite values"),
    ],
)
def test_vis_noise_calibration_rejects_invalid_kernel(kernel, message):
    with pytest.raises(ValueError, match=message):
        VISNoiseCalibration.build(
            coloring_kernel=kernel,
            residual_scale=10.0,
            source_release="Euclid-Q1-MER",
            estimator_version="test-v1",
        )


def test_vis_noise_calibration_sets_absolute_robust_rms_and_preserves_mean():
    rng = np.random.default_rng(41)
    residual = rng.normal(3.25, 27.0, size=(384, 384)).astype(np.float32)
    calibration = VISNoiseCalibration.build(
        coloring_kernel=((1.0,),),
        residual_scale=9.5,
        source_release="Euclid-Q1-MER",
        estimator_version="test-v1",
    )

    colored = calibration.apply(residual)
    median = float(np.median(colored))
    robust_rms = 1.4826 * float(np.median(np.abs(colored - median)))

    assert float(colored.mean()) == pytest.approx(float(residual.mean()), abs=2e-6)
    assert robust_rms == pytest.approx(9.5, rel=0.015)


def test_vis_noise_calibration_draws_interpolated_field_scale():
    residual = np.random.default_rng(8).normal(size=(256, 256)).astype(np.float32)
    calibration = VISNoiseCalibration.build(
        coloring_kernel=((1.0,),),
        residual_scale=10.0,
        field_scale_quantiles=(0.7, 0.85, 1.0, 1.25, 1.6),
        source_release="Euclid-Q1-MER",
        estimator_version="test-v1",
    )
    expected_rng = np.random.default_rng(29)
    expected_factor = float(np.interp(
        float(expected_rng.random()),
        calibration.FIELD_SCALE_PROBABILITIES,
        calibration.field_scale_quantiles,
    ))

    median_field = calibration.apply(residual)
    sampled_field = calibration.apply(residual, rng=np.random.default_rng(29))
    mean = float(residual.mean())

    np.testing.assert_allclose(
        sampled_field - mean,
        expected_factor * (median_field - mean),
        rtol=2e-5,
        atol=2e-5,
    )


def test_vis_noise_calibration_uses_linear_convolution_without_wrap():
    residual = np.zeros((31, 31), dtype=np.float32)
    residual[0, 0] = 1.0
    root_half = float(np.sqrt(0.5))
    calibration = VISNoiseCalibration.build(
        coloring_kernel=(
            (0.0, 0.0, 0.0),
            (root_half, -root_half, 0.0),
            (0.0, 0.0, 0.0),
        ),
        residual_scale=1.0,
        source_release="Euclid-Q1-MER",
        estimator_version="test-v1",
    )

    colored = calibration.apply(residual)

    # The edge impulse affects its local output. Circular convolution would
    # also create a feature in the last column; linear padded convolution does
    # not, so the last column remains equal to an interior blank column.
    assert abs(float(colored[0, 0] - colored[0, 15])) > 1.0
    np.testing.assert_array_equal(colored[:, -1], colored[:, 15])
    assert float(colored.mean()) == pytest.approx(float(residual.mean()), abs=1e-7)


def test_calibrated_archive_noise_changes_only_vis_stochastic_residual():
    signal = np.linspace(0.0, 100.0, 96 * 96, dtype=np.float32).reshape(96, 96)
    calibration = _calibration(residual_scale=12.0)
    expected_rng = np.random.default_rng(18)
    raw = apply_band_noise(
        signal,
        Config.BAND_VIS,
        expected_rng,
        add_artifacts=False,
    )
    expected = signal + calibration.apply(raw - signal, rng=expected_rng)

    actual = apply_archive_noise(
        signal,
        Config.BAND_VIS,
        np.random.default_rng(18),
        add_artifacts=False,
        vis_noise_calibration=calibration,
    )

    np.testing.assert_array_equal(actual, expected.astype(np.float32))


def test_vis_calibration_that_owns_scale_bypasses_generic_noise_map():
    signal = np.full((96, 96), 20.0, dtype=np.float32)
    calibration = _calibration(owns_field_scale=True)
    plain = apply_archive_noise(
        signal,
        Config.BAND_VIS,
        np.random.default_rng(55),
        vis_noise_calibration=calibration,
    )
    mapped = apply_archive_noise(
        signal,
        Config.BAND_VIS,
        np.random.default_rng(55),
        noise_scale_map=np.full(signal.shape, 4.0, dtype=np.float32),
        vis_noise_calibration=calibration,
    )

    np.testing.assert_array_equal(mapped, plain)


def test_vis_calibration_can_explicitly_defer_to_generic_noise_map():
    signal = np.full((96, 96), 20.0, dtype=np.float32)
    calibration = _calibration(owns_field_scale=False)
    plain = apply_archive_noise(
        signal,
        Config.BAND_VIS,
        np.random.default_rng(56),
        vis_noise_calibration=calibration,
    )
    mapped = apply_archive_noise(
        signal,
        Config.BAND_VIS,
        np.random.default_rng(56),
        noise_scale_map=np.full(signal.shape, 1.5, dtype=np.float32),
        vis_noise_calibration=calibration,
    )

    np.testing.assert_allclose(
        mapped - signal,
        1.5 * (plain - signal),
        rtol=3e-5,
        atol=3e-5,
    )


def test_vis_calibration_argument_leaves_nisp_bitwise_unchanged():
    signal = np.full((90, 93), 20.0, dtype=np.float32)
    plain = apply_archive_noise(
        signal,
        Config.BAND_Y_E,
        np.random.default_rng(77),
        add_artifacts=False,
    )
    with_vis_model = apply_archive_noise(
        signal,
        Config.BAND_Y_E,
        np.random.default_rng(77),
        add_artifacts=False,
        vis_noise_calibration=_calibration(),
    )

    np.testing.assert_array_equal(with_vis_model, plain)


def test_simulator_routes_calibration_and_shared_scale_to_correct_bands(
    monkeypatch,
):
    calibration = _calibration(owns_field_scale=True)
    simulator = ObservationSimulator(config=ObservationSimulatorConfig(
        add_noise=True,
        add_artifacts=False,
        add_saturation=False,
        randomize_psf=False,
        vis_noise_calibration=calibration,
    ))
    seen = {}

    def fake_archive_noise(signal, band, rng, **kwargs):
        del rng
        seen[band.name] = kwargs
        return signal

    monkeypatch.setattr(
        observation_module,
        "apply_archive_noise",
        fake_archive_noise,
    )
    hr = np.zeros((32, 32), dtype=np.float32)
    shared_scale = np.full((16, 16), 1.2, dtype=np.float32)

    simulator._process_one_band(
        hr,
        Config.BAND_VIS,
        np.random.default_rng(1),
        noise_scale_map=shared_scale,
    )
    simulator._process_one_band(
        hr,
        Config.BAND_Y_E,
        np.random.default_rng(2),
        noise_scale_map=shared_scale,
    )

    assert seen["VIS"]["vis_noise_calibration"] is calibration
    assert seen["VIS"]["noise_scale_map"] is None
    assert seen["Y_E"]["vis_noise_calibration"] is None
    assert seen["Y_E"]["noise_scale_map"] is shared_scale
