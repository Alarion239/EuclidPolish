"""Schema tests for the reviewed VIS noise calibration artifact."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from euclid_polish.sky.observation.noise_calibration import VISNoiseCalibration


def _calibration(
    *, residual_scale: float = 11.0, owns_field_scale: bool = True,
) -> VISNoiseCalibration:
    return VISNoiseCalibration.build(
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
            residual_scale=10.0,
            field_scale_quantiles=quantiles,
            source_release="Euclid-Q1-MER",
            estimator_version="test-v1",
        )


def test_vis_noise_calibration_rejects_retired_coloring_schema():
    payload = _calibration().to_payload()
    payload["version"] = 1
    payload["coloring_kernel"] = [[1.0]]
    payload.pop("mode")

    with pytest.raises(ValueError, match="invalid VIS noise calibration schema"):
        VISNoiseCalibration.from_payload(payload)


def test_vis_noise_calibration_rejects_non_amplitude_mode():
    payload = _calibration().to_payload()
    payload["mode"] = "correlated"

    with pytest.raises(ValueError, match="mode must be 'amplitude_only'"):
        VISNoiseCalibration.from_payload(payload)
