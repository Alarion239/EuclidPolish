"""Pure-array reconstruction and lens-detection metrics."""

from __future__ import annotations

import numpy as np

from euclid_polish.lensfinder.metrics import (
    auc,
    roc_curve,
    threshold_at_fpr,
    tpr_at_threshold,
    tpr_vs_theta_e,
)


def evaluate_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    labels: np.ndarray,
    theta_e: np.ndarray,
    disagreement: np.ndarray | None = None,
    fprs: tuple[float, ...] = (0.001, 0.01, 0.05),
) -> dict:
    predictions = np.asarray(predictions, np.float32)
    targets = np.asarray(targets, np.float32)
    labels = np.asarray(labels, np.int8)
    theta_e = np.asarray(theta_e, float)
    if predictions.shape != targets.shape or predictions.shape[0] != labels.size:
        raise ValueError("predictions, targets, and labels are not aligned")
    scores = np.maximum(predictions, 0).sum(axis=tuple(range(1, predictions.ndim)))
    fpr_curve, tpr_curve, _ = roc_curve(scores, labels)
    positives, negatives = labels == 1, labels == 0
    tpr_at = {}
    for target_fpr in fprs:
        threshold = threshold_at_fpr(scores, labels, target_fpr)
        tpr_at[str(target_fpr)] = {
            "threshold": threshold,
            "tpr": tpr_at_threshold(scores, labels, threshold),
        }

    target_flux = np.maximum(targets, 0).sum(axis=tuple(range(1, targets.ndim)))
    prediction_flux = np.maximum(predictions, 0).sum(axis=tuple(range(1, predictions.ndim)))
    retention = np.divide(
        prediction_flux,
        target_flux,
        out=np.zeros_like(prediction_flux),
        where=target_flux > 0,
    )
    positive_mae = (
        float(np.mean(np.abs(predictions[positives] - targets[positives])))
        if positives.any()
        else float("nan")
    )
    positive_mse = (
        float(np.mean((predictions[positives] - targets[positives]) ** 2))
        if positives.any()
        else float("nan")
    )
    peak = float(np.max(targets[positives])) if positives.any() else 0.0
    psnr = (
        float("inf")
        if positive_mse == 0
        else (10 * np.log10(peak**2 / positive_mse) if peak > 0 else float("nan"))
    )
    finite_theta = theta_e[positives & np.isfinite(theta_e)]
    if finite_theta.size:
        lo, hi = float(finite_theta.min()), float(finite_theta.max())
        if lo == hi:
            hi = lo + 1e-6
        theta_metric = tpr_vs_theta_e(
            scores,
            labels,
            theta_e,
            target_fpr=0.01,
            bins=np.linspace(lo, hi + np.finfo(float).eps, 5),
        )
    else:
        theta_metric = {"threshold": float("nan"), "target_fpr": 0.01, "bins": []}
    output = {
        "auc": auc(fpr_curve, tpr_curve),
        "tpr_at_fpr": tpr_at,
        "theta_e_recall": theta_metric,
        "positive_flux_retention_mean": (
            float(retention[positives].mean()) if positives.any() else float("nan")
        ),
        "positive_mae": positive_mae,
        "positive_psnr": psnr,
        "negative_residual_flux_mean": (
            float(prediction_flux[negatives].mean()) if negatives.any() else float("nan")
        ),
        "scores": scores.tolist(),
    }
    if disagreement is not None:
        output["disagreement_mean"] = float(np.asarray(disagreement).mean())
    return output
