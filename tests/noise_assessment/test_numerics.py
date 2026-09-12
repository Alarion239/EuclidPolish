"""Small controlled experiments, independent of web/training fixtures."""

import numpy as np
import pytest
from scipy import sparse
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

from euclid_polish.noise_assessment.measurement import mer_conversion
from euclid_polish.noise_assessment.numerics import (
    PropagatedNoise,
    amplitude_decision,
    bilinear_operator,
    clustered_interval,
    convolution_operator,
    difference_measurement,
    independent_components,
    median_prediction,
)


def identity_noise(shape, variance):
    return PropagatedNoise(sparse.eye(np.prod(shape), format="csr"), np.full(shape, variance), shape)


def test_bilinear_exact_weights_variance_and_covariance():
    yy, xx = np.indices((3, 3))
    matrix, valid = bilinear_operator((5, 5), xx + 0.25, yy + 0.5)
    noise = PropagatedNoise(matrix, np.ones((5, 5)) * 4, (3, 3))
    assert valid.all()
    np.testing.assert_allclose(noise.diagonal(), 4 * (0.375**2 + 0.125**2) * 2)
    assert noise.covariance(0, 1) > 0
    aperture = np.ones((3, 3))
    assert noise.aperture_variance(aperture) > noise.diagonal().sum()
    rng = np.random.default_rng(7)
    draws = np.array([noise.draw(rng).sum() for _ in range(4000)])
    assert np.var(draws, ddof=1) == pytest.approx(noise.aperture_variance(aperture), rel=0.05)


def test_composed_resampling_convolution_retains_cross_terms():
    shape = (13, 13)
    yy, xx = np.indices(shape)
    resample, covered = bilinear_operator(shape, xx + 0.2, yy + 0.3)
    convolve, valid = convolution_operator(shape, np.ones((3, 3)), covered)
    composed = PropagatedNoise(convolve @ resample, np.ones(shape), shape)
    # Sequential diagonal-only propagation loses induced covariance.
    wrong = convolve.power(2) @ (resample.power(2) @ np.ones(np.prod(shape)))
    assert np.all(composed.diagonal()[valid] > wrong.reshape(shape)[valid] * 1.5)
    rng = np.random.default_rng(818)
    draws = np.array([composed.draw(rng)[6, 6] for _ in range(3000)])
    assert np.var(draws, ddof=1) == pytest.approx(composed.diagonal()[6, 6], rel=0.06)


def test_coverage_border_and_invalid_flags():
    valid = np.ones((5, 5), bool)
    valid[2, 2] = False
    yy, xx = np.indices((5, 5))
    matrix, kept = bilinear_operator((5, 5), xx + 0.25, yy + 0.25, valid)
    assert not kept[-1].any()
    assert not kept[:, -1].any()
    assert not kept[1:3, 1:3].any()
    assert np.all(np.asarray(matrix.sum(axis=1)).ravel()[~kept.ravel()] == 0)
    _, integer = bilinear_operator((5, 5), xx, yy)
    assert integer.all()


def test_gaussian_difference_and_poisson_source_noise_unequal_times():
    rng = np.random.default_rng(401)
    shape = (128, 128)
    yy, xx = np.indices(shape)
    rate = 30 + 200 * np.exp(-((xx - 64) ** 2 + (yy - 64) ** 2) / 300)
    times = (80.0, 500.0)
    a, b = [rng.poisson(rate * t) / t for t in times]
    na = PropagatedNoise(sparse.eye(rate.size, format="csr") / times[0], rate * times[0], shape)
    nb = PropagatedNoise(sparse.eye(rate.size, format="csr") / times[1], rate * times[1], shape)
    stats, arrays = difference_measurement(a, b, na, nb, np.ones(shape, bool), ["a"], ["b"])
    assert stats["std_z"] == pytest.approx(1, abs=0.05)
    assert abs(stats["mean_z"]) < 0.05
    assert arrays["predicted_variance"][64, 64] > 5 * arrays["predicted_variance"][0, 0]
    noise = identity_noise(shape, 7)
    a, b = rate + noise.draw(rng), rate + noise.draw(rng)
    gaussian, _ = difference_measurement(a, b, noise, noise, np.ones(shape, bool), ["c"], ["d"])
    assert gaussian["std_z"] == pytest.approx(1, abs=0.05)
    assert gaussian["tail_abs_z_gt_3"] < 0.006


def test_cross_convolution_cancels_structure_but_detects_mismatch():
    shape = (48, 48)
    source = np.zeros(shape)
    source[24, 24] = 100_000
    p = np.zeros((7, 7))
    p[3, 3] = 1
    p = gaussian_filter(p, 0.7)
    p /= p.sum()
    q = np.zeros((9, 9))
    q[4, 4] = 1
    q = gaussian_filter(q, 1.2)
    q /= q.sum()
    a, b = convolve2d(source, p, mode="same"), convolve2d(source, q, mode="same")
    ca, va = convolution_operator(shape, q)
    cb, vb = convolution_operator(shape, p)
    matched_a, matched_b = (ca @ a.ravel()).reshape(shape), (cb @ b.ravel()).reshape(shape)
    np.testing.assert_allclose(matched_a[va & vb], matched_b[va & vb], atol=1e-10)
    noise = identity_noise(shape, 1)
    bad, _ = difference_measurement(a, b, noise, noise, va & vb, ["a"], ["b"])
    assert bad["tail_abs_z_gt_5"] > 0.02
    yy, xx = np.indices(shape)
    shift, shifted_valid = bilinear_operator(shape, xx + 1, yy)
    shifted = (shift @ matched_b.ravel()).reshape(shape)
    bad_registration, _ = difference_measurement(
        matched_a, shifted, noise, noise, va & vb & shifted_valid, ["a"], ["b"]
    )
    assert bad_registration["std_z"] > 5


def test_correlated_noise_apertures_reject_white_noise_assumption():
    shape = (32, 32)
    matrix, _ = convolution_operator(shape, np.ones((3, 3)))
    noise = PropagatedNoise(matrix, np.ones(shape) * 9, shape)
    aperture = np.zeros(shape)
    aperture[10:20, 10:20] = 1
    assert noise.aperture_variance(aperture) > 7 * np.sum(noise.diagonal() * aperture)


def test_median_prediction_unequal_noise_and_actual_times():
    shape = (8, 8)
    times = [80.0, 500.0, 500.0]
    signals = [np.ones(shape) * 10 for _ in times]
    noises = [identity_noise(shape, 10 / t) for t in times]
    metadata, predicted = median_prediction(noises, signals, [np.ones(shape, bool)] * 3, draws=2000)
    rng = np.random.default_rng(193)
    samples = np.median(np.stack([rng.normal(10, np.sqrt(10 / t), (2000, 8, 8)) for t in times]), axis=0)
    assert np.mean(predicted) == pytest.approx(np.mean(np.var(samples, axis=0)), rel=0.05)
    assert metadata["combination"] == "MEDIAN"
    assert np.mean(predicted) != pytest.approx(sum(10 / t for t in times) / 9, rel=0.05)


def test_independence_is_by_connected_native_ancestry():
    assert independent_components([["a"], ["a", "b"], ["b"], ["c"], [], []]) == [[0, 1, 2], [3], [4, 5]]
    summary = clustered_interval([1, 1, 1], [["a"], ["a", "b"], ["b"]])
    assert summary["independent_components"] == 1
    assert summary["ci95"] is None
    noise = identity_noise((8, 8), 1)
    with pytest.raises(ValueError, match="disjoint"):
        difference_measurement(
            np.zeros((8, 8)), np.zeros((8, 8)), noise, noise, np.ones((8, 8), bool), ["a", "b"], ["b", "c"]
        )


def test_unit_conversion_and_bad_units():
    header = {"DATASETR": "Q1_R1", "FILTER": "VIS", "MAGZERO": 24.6, "BUNIT": "ADU/s"}
    conversion = mer_conversion(header, "VIS")
    assert conversion["science_factor"] == conversion["rms_factor"]
    assert conversion["variance_factor"] == conversion["science_factor"] ** 2
    with pytest.raises(ValueError, match="unit"):
        mer_conversion({**header, "BUNIT": "MJy/sr"}, "VIS")
    with pytest.raises(ValueError, match="MAGZERO"):
        mer_conversion({**header, "MAGZERO": float("nan")}, "VIS")
    with pytest.raises(ValueError, match="release"):
        mer_conversion({**header, "DATASETR": "Q2_R1"}, "VIS")


def test_predeclared_amplitude_target_uses_intervals():
    assert amplitude_decision([0.97, 1.03]) == "within 5% target"
    assert amplitude_decision([1.06, 1.1]) == "disagreement"
    assert amplitude_decision([0.8, 1.2]) == "insufficient precision"
    assert amplitude_decision(None) == "insufficient precision"


def test_missing_difference_coverage_keeps_inspectable_arrays():
    shape = (8, 8)
    noise = identity_noise(shape, 1)
    valid = np.zeros(shape, bool)
    valid[:2] = True
    stats, arrays = difference_measurement(
        np.ones(shape), np.zeros(shape), noise, noise, valid, ["first"], ["second"]
    )
    assert stats["status"] == "insufficient coverage"
    assert stats["std_z"] is None
    assert arrays["difference"].shape == shape
    assert np.isfinite(arrays["z"]).sum() == 16


def test_median_monte_carlo_retains_spatial_covariance():
    shape = (8, 8)
    yy, xx = np.indices(shape)
    operator, valid = bilinear_operator((10, 10), xx + 0.5, yy + 0.5)
    noise = PropagatedNoise(operator, np.ones((10, 10)), shape)
    metadata, variance = median_prediction([noise, noise], [np.zeros(shape)] * 2, [valid] * 2, draws=2000)
    expected = noise.covariance(0, 1) / 2
    assert metadata["spatial_covariance"][0]["mean_covariance"] == pytest.approx(expected, rel=0.06)
    assert np.mean(variance) == pytest.approx(0.125, rel=0.05)
    aperture = np.zeros(shape)
    aperture[:5, :5] = 1
    aperture_result = next(r for r in metadata["aperture_sum_variance"] if r["square_side_pixels"] == 5)
    assert aperture_result["mean_variance"] == pytest.approx(noise.aperture_variance(aperture) / 2, rel=0.08)
