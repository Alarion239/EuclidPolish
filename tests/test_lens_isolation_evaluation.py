from __future__ import annotations

import os

import numpy as np

from euclid_polish.experiments.lens_isolation.evaluation import (
    evaluate_crop_arrays,
    sample_random_crop_coordinates,
    write_report,
)


def test_random_crop_coordinates_are_block_aligned_and_deterministic():
    first = sample_random_crop_coordinates(
        np.random.default_rng(7), field_size=(510, 510), crop_size=96, scale=2
    )
    second = sample_random_crop_coordinates(
        np.random.default_rng(7), field_size=(510, 510), crop_size=96, scale=2
    )
    assert first == second
    assert first[0] % 2 == first[1] % 2 == 0
    assert 0 <= first[0] <= 414
    assert 0 <= first[1] <= 414


def test_metrics_group_fixed_random_crops_by_observed_target_not_catalog_labels():
    targets = np.zeros((2, 2, 2, 1), np.float32)
    targets[0] = 2
    predictions = targets.copy()
    predictions[1] = 0.25
    metrics, rows = evaluate_crop_arrays(
        {"ensemble": predictions, "zero": np.zeros_like(predictions)}, targets
    )
    assert metrics["ensemble"]["aggregate_mae"] > 0
    assert metrics["ensemble"]["target_present"]["flux_retention_mean"] == 1
    assert metrics["ensemble"]["zero_target"]["residual_flux_mean"] == 1
    assert metrics["zero"]["target_present"]["flux_retention_mean"] == 0
    assert [row["label"] for row in rows if row["approach"] == "ensemble"] == [1, 0]


def test_write_report_emits_machine_readable_files_and_roc_plot(tmp_path):
    metrics = {"ensemble": {"auc": 1.0}, "zero": {"auc": 0.5}}
    rows = [
        {"index": 0, "approach": "ensemble", "label": 1, "score": 9.0},
        {"index": 1, "approach": "ensemble", "label": 0, "score": 1.0},
        {"index": 0, "approach": "zero", "label": 1, "score": 0.0},
        {"index": 1, "approach": "zero", "label": 0, "score": 0.0},
    ]
    paths = write_report(str(tmp_path), metrics, rows)
    assert set(paths) == {"metrics", "predictions", "roc"}
    assert all(os.path.isfile(path) for path in paths.values())
