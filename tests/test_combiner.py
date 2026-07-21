"""Incremental raw min/mean/max combiner tests."""

import json

import numpy as np
import pytest

from euclid_polish.eval.combiner import (
    ACTIVE_COMBINER_KINDS,
    BAND_NAMES,
    COMBINER_MODELS,
    CROSS_STAGE_MIN_SEPARATION,
    DEFAULT_WITHIN_STAGE_MIN_SEPARATION,
    RAW_INCREMENTAL_MINMEANMAX_RBF_KIND,
    FitBufferAccumulator,
    RawIncrementalMinMeanMaxRBFCombiner,
    _weighted_kmeans,
    combiner_model_spec,
    fit_combiner,
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


def test_weighted_kmeans_respects_same_stage_separation():
    rows = np.asarray([[0.0], [0.02], [0.04], [1.0], [2.0], [3.0]])
    centers = _weighted_kmeans(
        rows, np.ones(len(rows)), 2, seed=2,
        min_separation=0.2)
    assert len(centers) == 2
    assert abs(float(centers[0, 0] - centers[1, 0])) >= 0.2


def test_incremental_raw_combiner_improves_mean_and_allows_cross_stage_overlap(
        tmp_path):
    raw, target = _training_problem()
    baseline_l1 = float(np.mean(np.abs(
        np.maximum(np.mean(raw, axis=1), 0.0) - target)))
    combiner = fit_combiner(
        (raw, target), ["m0", "m1", "m2"],
        n_kernels=32, seed=3)

    assert isinstance(combiner, RawIncrementalMinMeanMaxRBFCombiner)
    assert combiner.kind == RAW_INCREMENTAL_MINMEANMAX_RBF_KIND
    assert float(np.mean(np.abs(combiner.predict_pixels(raw) - target))) \
        < baseline_l1
    assert combiner.fit_meta["within_increment_min_separation_normalized"] \
        == DEFAULT_WITHIN_STAGE_MIN_SEPARATION
    assert combiner.fit_meta["cross_increment_min_separation_normalized"] \
        == CROSS_STAGE_MIN_SEPARATION == 0.0

    normalized = combiner.centers / combiner.scales[None, :]
    for stage in np.unique(combiner.increment_ids):
        centers = normalized[combiner.increment_ids == stage]
        distance = np.sqrt(np.sum(
            (centers[:, None, :] - centers[None, :, :]) ** 2, axis=2))
        np.fill_diagonal(distance, np.inf)
        assert float(np.min(distance)) >= DEFAULT_WITHIN_STAGE_MIN_SEPARATION
    first = normalized[combiner.increment_ids == 1]
    second = normalized[combiner.increment_ids == 2]
    cross_distance = np.sqrt(np.sum(
        (first[:, None, :] - second[None, :, :]) ** 2, axis=2))
    assert float(np.min(cross_distance)) < DEFAULT_WITHIN_STAGE_MIN_SEPARATION

    save_combiner(combiner, str(tmp_path))
    loaded = load_combiner(
        str(tmp_path), member_labels=combiner.member_labels)
    assert isinstance(loaded, RawIncrementalMinMeanMaxRBFCombiner)
    np.testing.assert_allclose(
        loaded.predict_pixels(raw[:50]), combiner.predict_pixels(raw[:50]))


def test_loader_rejects_retired_combiner_artifact(tmp_path):
    artifact = tmp_path / combiner_model_spec().artifact_dir
    artifact.mkdir()
    (artifact / "combiner.json").write_text(json.dumps({
        "kind": "stats_rbf_gate", "member_labels": ["m0"],
    }))
    np.savez(artifact / "combiner.npz", value=np.ones(1))
    assert load_combiner(str(tmp_path)) is None
