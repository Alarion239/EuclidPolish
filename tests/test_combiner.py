"""Incremental all-inference combiner tests."""

import json

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.eval.combiner import (
    ACTIVE_COMBINER_KINDS,
    BAND_NAMES,
    COMBINER_MODELS,
    CROSS_STAGE_MIN_SEPARATION,
    DEFAULT_WITHIN_STAGE_MIN_SEPARATION,
    RAW_INCREMENTAL_MINMEANMAX_RBF_KIND,
    FitBufferAccumulator,
    RawIncrementalMinMeanMaxRBFCombiner,
    _all_inference_features,
    _best_member_achievable_l1_gain,
    _best_psnr_initial_logits,
    _member_interval_error_floor,
    _recoverable_error,
    _weighted_kmeans,
    _weighted_separated_center_indices,
    combiner_model_spec,
    fit_combiner,
    fit_combiner_minibatched,
    load_combiner,
    normalize_model_kind,
    save_combiner,
)


def _training_problem(seed=7, n=2400):
    rng = np.random.default_rng(seed)
    target = rng.exponential(2.0, (n, 4)).astype(np.float32)
    raw = np.repeat(target[:, None, :], 3, axis=1)
    raw += rng.normal(0.0, 0.08, raw.shape).astype(np.float32)
    artifact = (target[:, 0] < 0.7) & (rng.random(n) < 0.45)
    raw[artifact, 0, :] += 4.0
    return raw, target


def test_only_incremental_raw_combiner_is_registered():
    assert ACTIVE_COMBINER_KINDS == (RAW_INCREMENTAL_MINMEANMAX_RBF_KIND,)
    assert tuple(COMBINER_MODELS) == ACTIVE_COMBINER_KINDS
    assert combiner_model_spec().default_kernels == 128
    assert normalize_model_kind(None) == RAW_INCREMENTAL_MINMEANMAX_RBF_KIND
    with pytest.raises(ValueError, match="unsupported combiner"):
        normalize_model_kind("stats_rbf_gate")


def test_aligned_fit_buffer_retains_all_members_and_bands():
    accumulator = FitBufferAccumulator(
        BAND_NAMES, max_rows=500, n_bright_bins=4,
        per_bin_per_field=50, seed=2)
    rng = np.random.default_rng(3)
    predictions = rng.uniform(0, 4, (3, 12, 11, 4)).astype(np.float32)
    target = rng.uniform(0, 4, (12, 11, 4)).astype(np.float32)
    accumulator.add(predictions, target)
    rows, truth = accumulator.buffer()
    assert rows.ndim == 3 and rows.shape[1:] == (3, 4)
    assert truth.shape == (len(rows), 4)
    assert accumulator.member_validation_psnr().shape == (3,)
    assert np.all(np.isfinite(accumulator.member_validation_psnr()))
    assert set(accumulator.member_validation_metrics()) \
        == {"asinh_l1", "vis_asinh_psnr"}


def test_weighted_kmeans_respects_same_stage_separation():
    rows = np.asarray([[0.0], [0.02], [0.04], [1.0], [2.0], [3.0]])
    centers = _weighted_kmeans(
        rows, np.ones(len(rows)), 2, seed=2,
        min_separation=0.2)
    assert len(centers) == 2
    assert abs(float(centers[0, 0] - centers[1, 0])) >= 0.2


def test_weighted_kmeans_uses_smaller_cross_stage_separation():
    rows = np.asarray([[0.12], [0.13], [1.0], [2.0], [3.0]])
    anchors = np.asarray([[0.0]])
    centers = _weighted_kmeans(
        rows, np.ones(len(rows)), 2, seed=3,
        existing_centers=anchors,
        min_separation=0.35,
        existing_min_separation=0.1,
    )
    assert len(centers) == 2
    assert float(np.min(np.abs(centers - anchors[0]))) >= 0.1
    assert abs(float(centers[0, 0] - centers[1, 0])) >= 0.35


def test_weighted_separated_kmeanspp_distinguishes_stage_separation():
    points = np.asarray([[0.12], [0.2], [0.5], [0.9]])
    picked = _weighted_separated_center_indices(
        points,
        np.ones(len(points)),
        2,
        min_separation=0.35,
        existing_centers=np.asarray([[0.0]]),
        existing_min_separation=0.1,
        seed=5,
    )
    centers = points[picked]

    assert len(centers) == 2
    assert float(np.min(centers)) >= 0.1
    assert abs(float(centers[0, 0] - centers[1, 0])) >= 0.35


def test_recoverable_error_subtracts_the_member_interval_floor():
    members = np.asarray([
        [[1.0], [3.0]],
        [[1.0], [3.0]],
        [[1.0], [3.0]],
        [[1.0], [3.0]],
    ])
    targets = np.asarray([[2.0], [0.0], [4.0], [4.0]])
    current = np.asarray([[3.0], [1.0], [1.0], [3.0]])

    floor = _member_interval_error_floor(members, targets)
    recoverable = _recoverable_error(current, targets, floor)

    np.testing.assert_allclose(floor[:, 0], [0.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(recoverable[:, 0], [1.0, 0.0, 2.0, 0.0])


def test_best_member_gain_is_shared_band_and_definitely_achievable():
    members = np.asarray([[
        [0.0, 0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0, 2.0],
    ]])
    target = np.asarray([[0.0, 0.0, 0.0, 0.0]])
    current = np.asarray([[1.0, 1.0, 1.0, 1.0]])

    gain, floor = _best_member_achievable_l1_gain(members, target, current)

    np.testing.assert_allclose(floor, [0.0])
    np.testing.assert_allclose(gain, [1.0])


def test_best_psnr_initial_logits_are_near_one_hot_but_trainable():
    logits, probabilities, best = _best_psnr_initial_logits(
        np.asarray([10.0, 30.0, 20.0]), best_weight=0.99)

    assert best == 1
    np.testing.assert_allclose(probabilities, [0.005, 0.99, 0.005])
    shifted = np.exp(logits - np.max(logits))
    np.testing.assert_allclose(shifted / np.sum(shifted), probabilities)
    assert np.all(probabilities > 0.0)


def test_all_inference_coordinates_are_signed_per_band_asinh():
    scales = np.asarray([
        Config.get_band(name).asinh_stretch_scale_e for name in BAND_NAMES
    ])
    expected = np.asarray([
        [[-2.0, -1.0, 0.5, 1.5], [0.0, 1.0, 2.0, 3.0]],
    ])
    raw = np.sinh(expected) * scales[None, None, :]

    np.testing.assert_allclose(
        _all_inference_features(raw), expected.reshape(1, -1), atol=1e-12)


def test_single_member_convex_output_preserves_negative_sky_subtracted_values():
    combiner = RawIncrementalMinMeanMaxRBFCombiner(
        member_labels=["m0"],
        n_kernels=0,
        coefficients=np.empty((0, 1), np.float32),
        centers=np.empty((0, 4), np.float32),
        scales=np.ones(4, np.float32),
        sigmas=np.empty((0,), np.float32),
        increment_ids=np.empty((0,), np.int32),
        reference_features=np.zeros(4, np.float32),
        output_floors=np.ones(4, np.float32),
        baseline_member_index=None,
    )
    pixels = np.asarray([[[-20.0, -10.0, 0.0, 10.0]]])

    np.testing.assert_allclose(combiner.predict_pixels(pixels), pixels[:, 0, :])


def test_incremental_raw_combiner_is_shared_asinh_convex_and_improves_uniform(
        tmp_path):
    raw, target = _training_problem()
    scales = np.asarray([
        Config.get_band(name).asinh_stretch_scale_e for name in BAND_NAMES
    ])
    combiner = fit_combiner(
        (raw, target), ["m0", "m1", "m2"],
        n_kernels=32, seed=3,
        member_validation_metrics={
            "asinh_l1": np.asarray([0.1, 0.2, 0.3]),
            "vis_asinh_psnr": np.asarray([10.0, 30.0, 20.0]),
            "coherence_overall": np.asarray([0.2, 0.9, 0.1]),
            "coherence_sr": np.asarray([0.3, 0.8, 0.0]),
        })
    members_asinh = np.arcsinh(raw / scales[None, None, :])
    uniform = np.sinh(np.mean(members_asinh, axis=1)) * scales
    uniform_l1 = float(np.mean(np.abs(
        np.arcsinh(uniform / scales) - np.arcsinh(target / scales))))

    assert isinstance(combiner, RawIncrementalMinMeanMaxRBFCombiner)
    assert combiner.kind == RAW_INCREMENTAL_MINMEANMAX_RBF_KIND
    assert combiner.centers.shape[1] == raw.shape[1] * raw.shape[2]
    assert combiner.fit_meta["feature_schema"] \
        == "all_member_inferences_asinh_v9_staged_global_logit_convex_output"
    assert combiner.baseline_member_index is None
    assert combiner.fit_meta["initial_prediction"] \
        == "uniform_member_average_in_asinh_space"
    assert combiner.fit_meta["baseline_selection_metric"] \
        == "not_applicable_convex_member_gate"
    assert combiner.fit_meta["validation_prefix_metric"] \
        == "joint_asinh_L1_and_VIS_asinh_MSE"
    assert combiner.fit_meta["asinh_l1_gradient_tolerance"] == 1e-7
    assert combiner.fit_meta["residual_weight"] \
        == "current_equal_band_recoverable_asinh_l1"
    assert combiner.fit_meta["coefficient_parameterization"] \
        == "rbf_member_logits"
    assert combiner.fit_meta["output"] \
        == "shared_weight_convex_member_average_in_asinh_space"
    assert max(row["optimizer_iterations"]
               for row in combiner.fit_meta["center_history"]) > 0
    assert "train_mean_recoverable_l1" \
        in combiner.fit_meta["center_history"][0]
    prediction = combiner.predict_pixels(raw)
    weights = combiner.weights_from_electrons(raw)
    np.testing.assert_allclose(np.sum(weights, axis=1), 1.0, atol=1e-12)
    assert np.all(weights >= 0.0)
    np.testing.assert_allclose(
        prediction,
        np.sinh(np.einsum("nm,nmc->nc", weights, members_asinh)) * scales,
        atol=1e-10)
    assert np.all(prediction >= np.min(raw, axis=1) - 1e-10)
    assert np.all(prediction <= np.max(raw, axis=1) + 1e-10)
    assert float(np.mean(np.abs(
        np.arcsinh(prediction / scales) - np.arcsinh(target / scales)))) \
        <= uniform_l1 + 1e-12
    assert combiner.fit_meta["loss"] == "smooth_asinh_l1_plus_ridge"
    assert combiner.fit_meta["within_increment_min_separation_normalized"] \
        == DEFAULT_WITHIN_STAGE_MIN_SEPARATION
    assert combiner.fit_meta["cross_increment_min_separation_normalized"] \
        == CROSS_STAGE_MIN_SEPARATION == 0.1
    assert combiner.fit_meta["minimum_center_separation_normalized"] \
        == CROSS_STAGE_MIN_SEPARATION

    normalized = ((combiner.centers - combiner.reference_features[None, :])
                  / combiner.scales[None, :])
    if len(normalized) > 1:
        distance = np.sqrt(np.sum(
            (normalized[:, None, :] - normalized[None, :, :]) ** 2, axis=2))
        np.fill_diagonal(distance, np.inf)
        same_stage = (
            combiner.increment_ids[:, None] == combiner.increment_ids[None, :])
        same_stage &= ~np.eye(len(normalized), dtype=bool)
        cross_stage = ~same_stage & ~np.eye(len(normalized), dtype=bool)
        if np.any(same_stage):
            assert float(np.min(distance[same_stage])) \
                >= DEFAULT_WITHIN_STAGE_MIN_SEPARATION
        if np.any(cross_stage):
            assert float(np.min(distance[cross_stage])) \
                >= CROSS_STAGE_MIN_SEPARATION

    save_combiner(combiner, str(tmp_path))
    loaded = load_combiner(
        str(tmp_path), member_labels=combiner.member_labels)
    assert isinstance(loaded, RawIncrementalMinMeanMaxRBFCombiner)
    assert loaded.baseline_member_index == combiner.baseline_member_index
    np.testing.assert_allclose(
        loaded.predict_pixels(raw[:50]), combiner.predict_pixels(raw[:50]))
    np.testing.assert_allclose(
        loaded.global_logits,
        np.zeros(len(combiner.member_labels))
        if combiner.global_logits is None else combiner.global_logits)


def test_minibatched_combiner_streams_every_pixel_with_disjoint_fields():
    rng = np.random.default_rng(17)
    fields = {}
    for field_index in range(5):
        target = rng.uniform(0.0, 3.0, (8, 7, 4)).astype(np.float32)
        raw = np.repeat(target[None, ...], 3, axis=0)
        raw += rng.normal(0.0, 0.15, raw.shape).astype(np.float32)
        raw[0] += 0.5
        fields[field_index] = (raw, target)

    passes = []

    def field_factory(indices):
        passes.append(tuple(indices))
        for field_index in indices:
            raw, target = fields[field_index]
            yield field_index, raw, target

    combiner = fit_combiner_minibatched(
        field_factory,
        list(fields),
        ["m0", "m1", "m2"],
        n_kernels=8,
        seed=4,
        batch_rows=16,
        epochs=1,
        normalizer_rows=80,
        candidate_rows=80,
        increment_size=4,
        member_validation_psnr=np.asarray([30.0, 20.0, 10.0]),
    )

    meta = combiner.fit_meta
    train_fields = set(meta["training_field_indices"])
    validation_fields = set(meta["validation_field_indices"])
    assert train_fields.isdisjoint(validation_fields)
    assert train_fields | validation_fields == set(fields)
    assert meta["training_pixels_per_epoch"] == len(train_fields) * 8 * 7
    assert meta["validation_pixels_per_epoch"] == len(validation_fields) * 8 * 7
    assert meta["center_history"][0]["train_pixels"] \
        == meta["training_pixels_per_epoch"]
    assert meta["center_history"][0]["optimizer_iterations"] > 1
    assert meta["coefficient_parameterization"] \
        == "global_plus_rbf_member_logits"
    assert meta["output"] \
        == "shared_weight_convex_member_average_in_asinh_space"
    assert combiner.coefficients.shape[1] == 3
    assert combiner.global_logits.shape == (3,)
    assert meta["initial_best_member_index"] == 0
    assert meta["initial_member_probabilities"][0] == pytest.approx(0.99)
    assert meta["increment_size"] == 4
    assert [row["n_centers"] for row in meta["center_history"]] == [4, 8]
    sample = fields[0][0].reshape(3, -1, 4).transpose(1, 0, 2)[:20]
    weights = combiner.weights_from_electrons(sample)
    np.testing.assert_allclose(np.sum(weights, axis=1), 1.0, atol=1e-12)
    assert np.all(weights >= 0.0)
    assert max(combiner.sigmas, default=0.0) <= 8.0
    assert meta["global_center_min_separation_normalized"] \
        == CROSS_STAGE_MIN_SEPARATION
    normalized = ((combiner.centers - combiner.reference_features[None, :])
                  / combiner.scales[None, :])
    if len(normalized) > 1:
        distance = np.sqrt(np.sum(
            (normalized[:, None, :] - normalized[None, :, :]) ** 2, axis=2))
        np.fill_diagonal(distance, np.inf)
        assert float(np.min(distance)) >= DEFAULT_WITHIN_STAGE_MIN_SEPARATION
    assert len(passes) >= 5


def test_loader_rejects_retired_combiner_artifact(tmp_path):
    artifact = tmp_path / combiner_model_spec().artifact_dir
    artifact.mkdir()
    (artifact / "combiner.json").write_text(json.dumps({
        "kind": "stats_rbf_gate", "member_labels": ["m0"],
    }))
    np.savez(artifact / "combiner.npz", value=np.ones(1))
    assert load_combiner(str(tmp_path)) is None
