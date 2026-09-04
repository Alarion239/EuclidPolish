"""Focused tests for empirical VIS-noise fitting and artifact boundaries."""

from __future__ import annotations

import json

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.photometry import adu_per_s_to_electrons_factor
from euclid_polish.sky.observation.noise_calibration import VISNoiseCalibration
from euclid_polish.web.helpers import vis_noise_calibration as calibration


def _write_vis_bundle(
    path,
    data,
    *,
    primary: dict | None = None,
    unit: str | None = "electron",
    magzero: float | None = None,
):
    image = fits.ImageHDU(np.asarray(data, dtype=np.float32), name="VIS")
    if unit is not None:
        image.header["BUNIT"] = unit
    if magzero is not None:
        image.header["MAGZERO"] = magzero
    header = fits.Header()
    for key, value in (primary or {}).items():
        header[key] = value
    fits.HDUList([fits.PrimaryHDU(header=header), image]).writeto(path)


def _runtime(*, scale: float) -> dict:
    return VISNoiseCalibration.build(
        residual_scale=scale,
        field_scale_quantiles=(0.8, 0.9, 1.0, 1.1, 1.2),
        owns_field_scale=True,
        source_release="Q1_R1",
        estimator_version="test-v1",
    ).to_payload()


def test_measurement_residual_preserves_low_frequency_power_and_masks_source():
    side = 128
    yy, xx = np.indices((side, side))
    wave = np.sin(2.0 * np.pi * xx / 40.0)
    image = (
        100.0
        + 10.0 * wave
        + 0.03 * xx
        - 0.02 * yy
        + np.random.default_rng(3).normal(size=(side, side))
    )
    image[60:65, 60:65] += 200.0

    residual, mask = calibration._source_masked_residual(image)
    usable = ~mask
    recovered_amplitude = float(
        np.sum(residual[usable] * wave[usable]) / np.sum(np.square(wave[usable]))
    )

    assert mask[62, 62]
    assert recovered_amplitude == pytest.approx(10.0, rel=0.08)


def test_pair_corrected_psd_is_stable_under_structured_source_mask():
    rng = np.random.default_rng(5)
    white = rng.normal(size=(128, 128))
    field = (
        white + np.roll(white, 1, axis=0) + np.roll(white, 1, axis=1)
    ) / np.sqrt(3.0)
    unmasked = np.zeros(field.shape, dtype=bool)
    structured_mask = rng.random(field.shape) < 0.28
    structured_mask[35:60, 50:78] = True
    edges = calibration._power_edges(128)

    baseline, _ = calibration._normalized_psd(
        field, unmasked, radial_edges=edges,
    )
    masked, _ = calibration._normalized_psd(
        field, structured_mask, radial_edges=edges,
    )
    mode_counts = calibration._radial_mode_counts(edges, field.shape[0])

    assert float(np.sum(baseline * mode_counts)) == pytest.approx(1.0)
    assert float(np.sum(masked * mode_counts)) == pytest.approx(1.0)
    assert float(np.sum(np.minimum(baseline, masked) * mode_counts)) > 0.96
    assert float(np.median(np.abs(np.log10(masked / baseline)))) < 0.04


def test_radial_psd_uses_fourier_mode_integrated_variance_normalization():
    tile_size = 64
    edges = calibration._power_edges(tile_size)
    counts = calibration._radial_mode_counts(edges, tile_size)
    kernel = np.asarray([
        [0.0, 0.2, 0.0],
        [0.1, 0.9, 0.1],
        [0.0, 0.3, 0.0],
    ])
    kernel /= np.linalg.norm(kernel)

    model_power = calibration._kernel_power(kernel, edges, tile_size)
    measured_power, _ = calibration._normalized_psd(
        np.random.default_rng(11).normal(size=(tile_size, tile_size)),
        np.zeros((tile_size, tile_size), dtype=bool),
        radial_edges=edges,
    )

    assert float(np.sum(model_power * counts)) == pytest.approx(1.0)
    assert float(np.sum(measured_power * counts)) == pytest.approx(1.0)
    assert not np.isclose(float(np.sum(model_power)), 1.0)


def test_vis_units_are_explicit_and_missing_magzero_never_defaults(tmp_path):
    data = np.full((8, 8), 2.0, dtype=np.float32)
    total_path = tmp_path / "total.fits"
    _write_vis_bundle(total_path, data, unit="electron")
    np.testing.assert_array_equal(
        calibration._load_vis_image({"output_path": str(total_path)}, tmp_path),
        data,
    )

    rate_path = tmp_path / "rate.fits"
    _write_vis_bundle(rate_path, data, unit=None, magzero=24.6)
    expected = data * adu_per_s_to_electrons_factor(24.6, Config.BAND_VIS)
    np.testing.assert_allclose(
        calibration._load_vis_image({"output_path": str(rate_path)}, tmp_path),
        expected,
    )

    missing_path = tmp_path / "missing.fits"
    _write_vis_bundle(missing_path, data, unit=None)
    with pytest.raises(ValueError, match="finite MAGZERO"):
        calibration._load_vis_image({"output_path": str(missing_path)}, tmp_path)

    unknown_path = tmp_path / "unknown.fits"
    _write_vis_bundle(unknown_path, data, unit="MJy/sr", magzero=24.6)
    with pytest.raises(ValueError, match="unsupported VIS BUNIT"):
        calibration._load_vis_image({"output_path": str(unknown_path)}, tmp_path)


def test_parent_holdout_is_field_stratified_and_keeps_two_train_parents():
    parents = [
        {"parent_id": f"{field}-{index}", "field": field}
        for field in ("EDF-N", "EDF-S", "EDF-F")
        for index in range(2)
    ]

    train, holdout = calibration._parent_split(parents, seed=19)

    assert len(train) == 3
    assert len(holdout) == 3
    assert {parent["field"] for parent in holdout} == {"EDF-N", "EDF-S", "EDF-F"}
    assert {parent["parent_id"] for parent in train}.isdisjoint(
        {parent["parent_id"] for parent in holdout}
    )


def test_synced_manifest_absolute_path_rebases_to_local_cutouts(tmp_path):
    cutouts = tmp_path / Config.EuclidSky.CUTOUTS_SUBDIR
    cutouts.mkdir()
    local_path = cutouts / "sky_0007.fits"
    expected = np.arange(64, dtype=np.float32).reshape(8, 8)
    parent = {
        "parent_id": "parent-7",
        "release_name": "Q1_R1",
        "product_type": "DpdMerBksMosaic",
        "mosaic_product_oid": "oid-7",
    }
    _write_vis_bundle(local_path, expected, primary={
        "POS_ID": 7,
        "VIS_PIX": 8,
        "PARENT": "parent-7",
        "RELEASE": "Q1_R1",
        "HIERARCH MOSAIC_PRODUCT_OID": "oid-7",
        "PRODTYPE": "DpdMerBksMosaic",
        "RA": 10.0,
        "DEC": 20.0,
    })
    sample = {
        "sample_id": 7,
        "parent_id": "parent-7",
        "parent": parent,
        "ra": 10.0,
        "dec": 20.0,
        "field": "EDF-N",
        "status": "cached",
        "actual_shape": [8, 8],
        "output_path": "/fasrc/data/cutouts/sky_0007.fits",
    }
    manifest = {
        "kind": "euclid_vis_noise_sampling",
        "version": 1,
        "source_release": "Q1_R1",
        "archive": {
            "product_type": "DpdMerBksMosaic",
            "requested_release_name": "Q1_R1",
        },
        "selection": {"cutout_size_vis_pixels": 8},
        "samples": [sample],
    }

    loaded = calibration._load_vis_image(
        sample,
        tmp_path,
        manifest,
    )

    np.testing.assert_array_equal(loaded, expected)

    with fits.open(local_path, mode="update") as hdul:
        hdul[0].header["PARENT"] = "stale-parent"
    with pytest.raises(ValueError, match="does not match manifest"):
        calibration._load_vis_image(sample, tmp_path, manifest)


def test_state_keeps_valid_active_when_new_candidate_differs(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(Config, "EUCLID_SKY_DIR", str(tmp_path / "sky"))
    active = _runtime(scale=19.0)
    calibration._write_json(calibration.active_vis_noise_path(), active)
    candidate = {
        "fingerprint": _runtime(scale=21.0)["fingerprint"],
        "valid": True,
        "sample_summary": {"independent_parent_count": 5},
    }
    calibration._write_json(calibration.vis_noise_candidate_path(), candidate)

    state = calibration.vis_noise_state()

    assert state["is_active"] is True
    assert state["candidate_is_active"] is False
    assert state["active"]["fingerprint"] == active["fingerprint"]


def test_sampling_readiness_requires_two_parents_in_every_q1_field(
    tmp_path, monkeypatch,
):
    sky_root = tmp_path / "euclid_sky"
    monkeypatch.setattr(Config, "EUCLID_SKY_DIR", str(sky_root))
    expected = sky_root / "vis_noise_samples" / "vis_noise_sampling_manifest.json"
    assert calibration.default_sampling_manifest_path() == expected
    expected.parent.mkdir(parents=True)

    samples = []
    for index in range(6):
        field = "EDF-N" if index < 3 else "EDF-S"
        parent = {
            "parent_id": f"parent-{index}",
            "release_name": "Q1_R1",
            "product_type": "DpdMerBksMosaic",
            "mosaic_product_oid": f"oid-{index}",
        }
        samples.append({
            "sample_id": index,
            "parent_id": parent["parent_id"],
            "parent": parent,
            "field": field,
            "ra": 10.0 + index,
            "dec": 20.0,
            "status": "cached",
            "actual_shape": [128, 128],
        })
    calibration._write_json(expected, {
        "kind": "euclid_vis_noise_sampling",
        "version": 1,
        "source_release": "Q1_R1",
        "archive": {
            "product_type": "DpdMerBksMosaic",
            "requested_release_name": "Q1_R1",
        },
        "selection": {"cutout_size_vis_pixels": 128},
        "samples": samples,
    })

    state = calibration.vis_noise_state()

    assert state["can_fit"] is False
    assert "each of EDF-N, EDF-S, and EDF-F" in state["unavailable_reason"]


def test_activation_rejects_invalid_or_too_few_parents(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    candidate = {
        "version": calibration.VIS_NOISE_CANDIDATE_VERSION,
        "valid": False,
        "runtime": _runtime(scale=19.0),
        "sample_summary": {"independent_parent_count": 5},
    }
    calibration._write_json(calibration.vis_noise_candidate_path(), candidate)
    with pytest.raises(ValueError, match="No valid fitted"):
        calibration.activate_vis_noise_candidate()

    candidate["valid"] = True
    candidate["version"] = calibration.VIS_NOISE_CANDIDATE_VERSION - 1
    calibration._write_json(calibration.vis_noise_candidate_path(), candidate)
    with pytest.raises(ValueError, match="current schema"):
        calibration.activate_vis_noise_candidate()

    candidate["version"] = calibration.VIS_NOISE_CANDIDATE_VERSION
    candidate["sample_summary"]["independent_parent_count"] = 2
    calibration._write_json(calibration.vis_noise_candidate_path(), candidate)
    with pytest.raises(ValueError, match="at least 3"):
        calibration.activate_vis_noise_candidate()

    candidate["sample_summary"] = {
        "independent_parent_count": 6,
        "fields": {"EDF-N": 6},
    }
    calibration._write_json(calibration.vis_noise_candidate_path(), candidate)
    with pytest.raises(ValueError, match="each Q1 deep field"):
        calibration.activate_vis_noise_candidate()


def test_near_size_cutouts_keep_intended_tile_count_without_duplicate_edges():
    image = np.arange(511 * 513, dtype=np.float64).reshape(511, 513)

    tiles = calibration._iter_tiles(image, 256)

    assert len(tiles) == 4
    assert all(tile.shape == (256, 256) for tile in tiles)
    np.testing.assert_array_equal(tiles[-1], image[-256:, 256:512])

    large = np.zeros((2561, 2562), dtype=np.float32)
    assert len(calibration._iter_tiles(large, 256)) == 100

    # SODA uses the local mosaic WCS to rasterize an angular request.  The
    # pilot returned 2580/2584 pixels for two valid nominal 2560-pixel fields.
    assert calibration._vis_noise_shape_matches((2580, 2584), 2560)
    assert not calibration._vis_noise_shape_matches((2580, 2587), 2560)
    oversized = np.zeros((2584, 2580), dtype=np.uint8)
    oversized[12, 10] = 7
    oversized_tiles = calibration._iter_tiles(oversized, 256)
    assert len(oversized_tiles) == 100
    assert oversized_tiles[0][0, 0] == 7


def test_field_scale_quantiles_use_independent_parent_medians_not_patch_extrema():
    parent_rms = np.asarray([8.0, 9.0, 10.0, 11.0, 12.0])
    parents = []
    for index, rms in enumerate(parent_rms):
        patches = np.asarray([rms - 0.2, rms - 0.1, rms, rms + 0.1, rms + 0.2])
        if index == len(parent_rms) - 1:
            patches[-1] = 1000.0
        parents.append({"rms_e": rms, "patch_rms_e": patches})

    residual_scale, factors = calibration._parent_weighted_rms(parents)

    patch_values = np.concatenate([parent["patch_rms_e"] for parent in parents])
    patch_weights = np.concatenate([
        np.full(len(parent["patch_rms_e"]), 1.0 / len(parent["patch_rms_e"]))
        for parent in parents
    ])
    expected_scale = calibration._weighted_quantiles(
        patch_values, patch_weights, np.asarray([0.5]),
    )[0]
    parent_quantiles = np.quantile(
        parent_rms, calibration._FIELD_SCALE_PROBABILITIES,
    )

    assert residual_scale == pytest.approx(expected_scale)
    np.testing.assert_allclose(factors, parent_quantiles / np.median(parent_rms))
    assert factors[-1] == pytest.approx(1.2)


def test_field_scale_endpoints_extrapolate_central_quantiles_not_outliers():
    parent_rms = np.asarray([
        2.0, 18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 80.0,
    ])
    parents = [
        {"rms_e": rms, "patch_rms_e": np.asarray([rms])}
        for rms in parent_rms
    ]

    _, factors = calibration._parent_weighted_rms(parents)
    p16, p50, p84 = np.quantile(parent_rms, (0.16, 0.5, 0.84))
    ratio = 0.16 / 0.34
    expected = np.asarray([
        p16 - ratio * (p50 - p16),
        p16,
        p50,
        p84,
        p84 + ratio * (p84 - p50),
    ]) / p50

    np.testing.assert_allclose(factors, expected)
    assert factors[0] > parent_rms.min() / p50
    assert factors[-1] < parent_rms.max() / p50


def test_fit_is_parent_grouped_and_emits_strict_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path / "data"))
    cutouts = tmp_path / "sample" / Config.EuclidSky.CUTOUTS_SUBDIR
    cutouts.mkdir(parents=True)
    samples = []
    fields = ("EDF-N", "EDF-S", "EDF-F", "EDF-N", "EDF-S", "EDF-F")
    for index, field_name in enumerate(fields):
        rng = np.random.default_rng(80 + index)
        white = rng.normal(0.0, 6.0 + 0.03 * index, size=(128, 128))
        image = (
            white + 0.35 * np.roll(white, 1, axis=0)
            + 0.2 * np.roll(white, 1, axis=1)
        ) / np.sqrt(1.0 + 0.35**2 + 0.2**2)
        path = cutouts / f"sky_{index:04d}.fits"
        parent = {
            "parent_id": f"parent-{index}",
            "release_name": "Q1_R1",
            "product_type": "DpdMerBksMosaic",
            "mosaic_product_oid": f"oid-{index}",
        }
        _write_vis_bundle(path, image, primary={
            "POS_ID": index,
            "VIS_PIX": 128,
            "PARENT": f"parent-{index}",
            "RELEASE": "Q1_R1",
            "HIERARCH MOSAIC_PRODUCT_OID": f"oid-{index}",
            "PRODTYPE": "DpdMerBksMosaic",
            "RA": 10.0 + index,
            "DEC": 20.0,
        })
        samples.append({
            "sample_id": index,
            "parent_id": f"parent-{index}",
            "parent": parent,
            "field": field_name,
            "ra": 10.0 + index,
            "dec": 20.0,
            "status": "written",
            "actual_shape": [128, 128],
            # Exercise FASRC-to-local rebasing while fitting.
            "output_path": f"/fasrc/data/cutouts/{path.name}",
        })
    manifest_path = tmp_path / "sample" / "vis_noise_sampling_manifest.json"
    manifest_path.write_text(json.dumps({
        "kind": "euclid_vis_noise_sampling",
        "version": 1,
        "source_release": "Q1_R1",
        "samples": samples,
        "selection": {"method": "test", "cutout_size_vis_pixels": 128},
        "support": {"role": "test"},
        "archive": {
            "product_type": "DpdMerBksMosaic",
            "requested_release_name": "Q1_R1",
        },
    }))

    candidate = calibration.fit_vis_noise_candidate(
        manifest_file=manifest_path,
        tile_size=64,
        max_lag=4,
    )

    assert candidate["sample_summary"]["independent_parent_count"] == 6
    assert candidate["sample_summary"]["train_parent_count"] == 3
    assert candidate["sample_summary"]["holdout_parent_count"] == 3
    assert candidate["version"] == calibration.VIS_NOISE_CANDIDATE_VERSION
    assert candidate["valid"] is True
    assert candidate["quality_gates"]["all_q1_fields_ge_2_parents"] is True
    assert candidate["quality_gates"]["passed"] is True
    assert candidate["source_release"] == "Q1_R1"
    assert candidate["residual_scale"] > 0.0
    assert candidate["owns_field_scale"] is True
    assert candidate["runtime"]["mode"] == "amplitude_only"
    assert candidate["quality_gates"]["spatial_structure_is_diagnostic_only"] is True
    assert candidate["validation"]["power"]["evaluation_range_arcsec"] == [0.25, 8.0]
    power = candidate["validation"]["power"]
    assert power["normalization"] == "unit_integrated_fourier_variance"
    assert power["log10_ratio_semantics"].startswith("log10(model_rms^2")
    np.testing.assert_allclose(
        np.dot(power["real"], power["fourier_mode_count"]), 1.0,
    )
    np.testing.assert_allclose(
        np.dot(power["model"], power["fourier_mode_count"]), 1.0,
    )
    fitted_power = candidate["fit"]["power"]
    assert fitted_power["normalization"] == "unit_integrated_fourier_variance"
    np.testing.assert_allclose(
        np.dot(
            fitted_power["normalized_psd"],
            fitted_power["fourier_mode_count"],
        ),
        1.0,
    )
    assert "large_scale_gt_8_arcsec" in candidate["validation"]["power"]
    assert set(candidate["runtime"]) == calibration._RUNTIME_KEYS
    assert calibration.runtime_vis_noise_payload(candidate) == candidate["runtime"]
    assert (
        candidate["runtime"]["estimator_version"]
        == calibration.VIS_NOISE_ESTIMATOR_VERSION
    )
    assert calibration.vis_noise_candidate_path().is_file()
