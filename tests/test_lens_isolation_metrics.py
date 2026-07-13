from __future__ import annotations

import numpy as np

from euclid_polish.experiments.lens_isolation.metrics import evaluate_predictions


def test_metrics_cover_detection_reconstruction_and_negative_residuals():
    targets = np.zeros((4, 2, 2, 1), np.float32)
    targets[:2] = 2
    predictions = targets.copy()
    predictions[2:] = 0.25
    metrics = evaluate_predictions(
        predictions,
        targets,
        labels=np.array([1, 1, 0, 0]),
        theta_e=np.array([0.5, 1.5, np.nan, np.nan]),
    )
    assert metrics["auc"] == 1
    assert metrics["positive_flux_retention_mean"] == 1
    assert metrics["positive_mae"] == 0
    assert metrics["negative_residual_flux_mean"] == 1
    assert "tpr_at_fpr" in metrics


def test_zero_baseline_has_zero_positive_recall():
    targets = np.ones((2, 2, 2, 1), np.float32)
    metrics = evaluate_predictions(np.zeros_like(targets), targets, labels=np.ones(2), theta_e=np.ones(2))
    assert metrics["positive_flux_retention_mean"] == 0
