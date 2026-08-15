"""Evaluate lens-only reconstructions on fixed random normal-field cutouts."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Mapping
from typing import Any

import numpy as np

from euclid_polish.config import Config
from euclid_polish.experiments.lens_isolation.ensemble import LensIsolationEnsemble
from euclid_polish.image.tfio import read_images, tfrecord_path


def roc_curve(
    scores: list[float], labels: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return FPR, TPR, and thresholds for binary scores."""
    score_array = np.asarray(scores, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    if score_array.size == 0:
        return (
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([np.inf, -np.inf]),
        )
    order = np.argsort(-score_array, kind="mergesort")
    score_array = score_array[order]
    label_array = label_array[order]
    positives = max(int((label_array == 1).sum()), 1)
    negatives = max(int((label_array == 0).sum()), 1)
    true_positives = np.cumsum(label_array == 1)
    false_positives = np.cumsum(label_array == 0)
    distinct = np.r_[np.diff(score_array) != 0, True]
    tpr = np.r_[0.0, true_positives[distinct] / positives]
    fpr = np.r_[0.0, false_positives[distinct] / negatives]
    thresholds = np.r_[np.inf, score_array[distinct]]
    return fpr, tpr, thresholds


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Return trapezoidal area under a ROC curve."""
    x = np.asarray(fpr, dtype=float)
    y = np.asarray(tpr, dtype=float)
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2.0))


def sample_random_crop_coordinates(
    rng: np.random.Generator,
    *,
    field_size: tuple[int, int] | int,
    crop_size: int,
    scale: int,
) -> tuple[int, int]:
    """Sample the same uniformly random, block-aligned offsets as training."""
    height, width = (field_size, field_size) if isinstance(field_size, int) else field_size
    if crop_size < 1 or scale < 1 or crop_size % scale:
        raise ValueError("crop_size must be positive and divisible by scale")
    max_y = (int(height) - int(crop_size)) // scale * scale
    max_x = (int(width) - int(crop_size)) // scale * scale
    if max_y < 0 or max_x < 0:
        raise ValueError(f"field {(height, width)} is smaller than crop {crop_size}")
    y = int(rng.integers(0, max_y + 1)) // scale * scale
    x = int(rng.integers(0, max_x + 1)) // scale * scale
    return y, x


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    if not labels.any() or labels.all():
        return None
    fpr, tpr, _ = roc_curve(scores.tolist(), labels.astype(int).tolist())
    return float(auc(fpr, tpr))


def evaluate_crop_arrays(
    predictions_by_approach: Mapping[str, np.ndarray],
    targets: np.ndarray,
    *,
    coordinates: list[tuple[int, int]] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Report aggregate and target-content groups after sampling is fixed."""
    targets = np.asarray(targets, np.float32)
    if targets.ndim != 4:
        raise ValueError(f"targets must be (N,H,W,C), got {targets.shape}")
    metrics: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    target_flux = np.maximum(targets, 0.0).sum(axis=(1, 2, 3))
    labels = target_flux > 0.0
    for approach, predictions in predictions_by_approach.items():
        predictions = np.asarray(predictions, np.float32)
        if predictions.shape != targets.shape:
            raise ValueError(
                f"{approach} predictions {predictions.shape} do not match targets {targets.shape}"
            )
        positive_flux = np.maximum(predictions, 0.0).sum(axis=(1, 2, 3))
        positive = labels
        zero = ~positive
        retention = np.divide(
            positive_flux,
            target_flux,
            out=np.zeros_like(positive_flux),
            where=target_flux > 0.0,
        )
        metrics[approach] = {
            "aggregate_mae": float(np.mean(np.abs(predictions - targets))),
            "auc": _auc(positive_flux, labels),
            "target_present": {
                "count": int(positive.sum()),
                "reconstruction_mae": (
                    float(np.mean(np.abs(predictions[positive] - targets[positive])))
                    if positive.any()
                    else None
                ),
                "flux_retention_mean": float(retention[positive].mean()) if positive.any() else None,
            },
            "zero_target": {
                "count": int(zero.sum()),
                "residual_flux_mean": float(positive_flux[zero].mean()) if zero.any() else None,
            },
        }
        for index, score in enumerate(positive_flux):
            row: dict[str, Any] = {
                "index": index,
                "approach": approach,
                "label": int(labels[index]),
                "target_flux": float(target_flux[index]),
                "score": float(score),
            }
            if coordinates is not None:
                row["y_hr"], row["x_hr"] = coordinates[index]
            rows.append(row)
    return metrics, rows


def evaluate_records(
    ensemble_dir: str,
    records_dir: str,
    *,
    seed: int = 0,
    crop_size: int = Config.DEFAULT_HR_CROP_SIZE,
    limit: int | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate held-out dirty/lens pairs without consulting source metadata."""
    ensemble = LensIsolationEnsemble(ensemble_dir)
    dirty = read_images(tfrecord_path(records_dir, "dirty_test"), num_images=2**31 - 1)
    targets = read_images(tfrecord_path(records_dir, "lens_test"), num_images=2**31 - 1)
    if limit is not None:
        dirty, targets = dirty[: int(limit)], targets[: int(limit)]
    if not dirty or len(dirty) != len(targets):
        raise ValueError("dirty_test and lens_test must be non-empty and position-aligned")

    rng = np.random.default_rng(seed)
    target_crops: list[np.ndarray] = []
    prediction_crops: dict[str, list[np.ndarray]] = {"ensemble": [], "zero": []}
    prediction_crops.update({name: [] for name in ensemble.member_names})
    coordinates: list[tuple[int, int]] = []
    for image, target in zip(dirty, targets, strict=True):
        target_data = np.asarray(target.data, np.float32)
        y, x = sample_random_crop_coordinates(
            rng,
            field_size=target_data.shape[:2],
            crop_size=crop_size,
            scale=Config.DEFAULT_REBIN_FACTOR,
        )
        stack = ensemble.member_arrays(np.asarray(image.data, np.float32))
        if stack.shape[1:] != target_data.shape:
            raise ValueError("member prediction and lens target shapes are not aligned")
        target_crops.append(target_data[y : y + crop_size, x : x + crop_size, :])
        prediction_crops["ensemble"].append(stack.mean(axis=0)[y : y + crop_size, x : x + crop_size, :])
        prediction_crops["zero"].append(np.zeros_like(target_crops[-1]))
        for member_index, name in enumerate(ensemble.member_names):
            prediction_crops[name].append(stack[member_index, y : y + crop_size, x : x + crop_size, :])
        coordinates.append((y, x))
    return evaluate_crop_arrays(
        {name: np.stack(crops) for name, crops in prediction_crops.items()},
        np.stack(target_crops),
        coordinates=coordinates,
    )


def write_report(
    output_dir: str,
    metrics: Mapping[str, Mapping[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> dict[str, str]:
    """Atomically write machine-readable metrics, rows, and an optional ROC plot."""
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.json")
    predictions_path = os.path.join(output_dir, "predictions.csv")
    _write_json(metrics_path, metrics)
    fields = sorted({key for row in prediction_rows for key in row}) or [
        "index",
        "approach",
        "label",
        "score",
    ]
    _write_csv(predictions_path, fields, prediction_rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    roc_path = os.path.join(output_dir, "roc.png")
    by_approach: dict[str, list[dict[str, Any]]] = {}
    for row in prediction_rows:
        by_approach.setdefault(str(row["approach"]), []).append(row)
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    for approach, rows in sorted(by_approach.items()):
        labels = [int(row["label"]) for row in rows]
        if not any(labels) or all(labels):
            continue
        fpr, tpr, _ = roc_curve([float(row["score"]) for row in rows], labels)
        value = metrics.get(approach, {}).get("auc")
        suffix = "n/a" if value is None else f"{float(value):.3f}"
        axis.plot(fpr, tpr, label=f"{approach} · AUC {suffix}")
    axis.plot([0, 1], [0, 1], "--", color="0.6", linewidth=1)
    axis.set(xlabel="false-positive rate", ylabel="true-positive rate", xlim=(0, 1), ylim=(0, 1))
    axis.grid(alpha=0.2)
    if axis.lines[1:]:
        axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    figure.savefig(roc_path, dpi=160)
    plt.close(figure)
    return {"metrics": metrics_path, "predictions": predictions_path, "roc": roc_path}


def _write_json(path: str, payload: Mapping[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=os.path.basename(path) + ".tmp-", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_csv(path: str, fields: list[str], rows: list[dict[str, Any]]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=os.path.basename(path) + ".tmp-", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
