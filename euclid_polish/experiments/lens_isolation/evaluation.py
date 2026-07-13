"""Stream fixed test pairs through experimental and baseline reconstructors."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from typing import Any

import numpy as np

from euclid_polish.experiments.lens_isolation.ensemble import LensIsolationEnsemble
from euclid_polish.experiments.lens_isolation.metrics import evaluate_predictions
from euclid_polish.image.tfio import read_images, tfrecord_path
from euclid_polish.lensfinder.metrics import roc_curve
from euclid_polish.model import Model


def _all_images(path: str):
    return read_images(path, num_images=2**31 - 1)


def evaluate_records(
    ensemble_dir: str,
    records_dir: str,
    *,
    include_sources: bool = True,
) -> tuple[dict[str, dict], list[dict[str, Any]]]:
    ensemble = LensIsolationEnsemble(ensemble_dir)
    dirty = _all_images(tfrecord_path(records_dir, "dirty_test"))
    targets = _all_images(tfrecord_path(records_dir, "lens_test"))
    manifest_path = os.path.join(records_dir, "manifest_test.csv")
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    if not (len(dirty) == len(targets) == len(manifest)):
        raise ValueError("dirty, target, and manifest test records are not aligned")
    labels = np.array([int(row["label"]) for row in manifest], np.int8)
    theta = np.array(
        [float(row["theta_E_arcsec"]) if row.get("theta_E_arcsec") else np.nan for row in manifest]
    )
    target_arrays = np.stack([image.data for image in targets])

    approach_predictions: dict[str, np.ndarray] = {"zero": np.zeros_like(target_arrays)}
    member_lists: dict[str, list[np.ndarray]] = {name: [] for name in ensemble.member_names}
    means, disagreements = [], []
    for image in dirty:
        stack = ensemble.member_arrays(image.data)
        means.append(stack.mean(axis=0))
        disagreements.append(stack.std(axis=0))
        for index, name in enumerate(ensemble.member_names):
            member_lists[name].append(stack[index])
    approach_predictions["ensemble"] = np.stack(means)
    for name, predictions in member_lists.items():
        approach_predictions[name] = np.stack(predictions)

    if include_sources:
        seen: set[str] = set()
        for member_dir in ensemble._member_dirs:
            origin_path = os.path.join(member_dir, "origin.json")
            if not os.path.isfile(origin_path):
                continue
            with open(origin_path, encoding="utf-8") as handle:
                source = str(json.load(handle).get("source", ""))
            if not source or source in seen or not os.path.isdir(source):
                continue
            seen.add(source)
            source_model = Model(source)
            approach_predictions[f"source:{os.path.basename(source)}"] = np.stack(
                [source_model.upsample_array(image.data) for image in dirty]
            )

    metrics = {}
    prediction_rows: list[dict[str, Any]] = []
    disagreement = np.stack(disagreements)
    for approach, predictions in approach_predictions.items():
        result = evaluate_predictions(
            predictions,
            target_arrays,
            labels=labels,
            theta_e=theta,
            disagreement=disagreement if approach == "ensemble" else None,
        )
        scores = result.pop("scores")
        metrics[approach] = result
        for index, score in enumerate(scores):
            prediction_rows.append(
                {
                    "index": index,
                    "approach": approach,
                    "label": int(labels[index]),
                    "theta_E_arcsec": "" if np.isnan(theta[index]) else theta[index],
                    "score": score,
                }
            )
    return metrics, prediction_rows


def write_report(
    output_dir: str,
    metrics: dict[str, dict],
    prediction_rows: list[dict[str, Any]],
) -> dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.json")
    predictions_path = os.path.join(output_dir, "predictions.csv")
    fd, temp_metrics = tempfile.mkstemp(prefix="metrics.json.tmp-", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True, allow_nan=True)
        os.replace(temp_metrics, metrics_path)
    finally:
        if os.path.exists(temp_metrics):
            os.unlink(temp_metrics)
    fd, temp_predictions = tempfile.mkstemp(prefix="predictions.csv.tmp-", dir=output_dir)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            fields = ["index", "approach", "label", "theta_E_arcsec", "score"]
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            writer.writerows(prediction_rows)
        os.replace(temp_predictions, predictions_path)
    finally:
        if os.path.exists(temp_predictions):
            os.unlink(temp_predictions)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    roc_path = os.path.join(output_dir, "roc.png")
    by_approach: dict[str, list[dict[str, Any]]] = {}
    for row in prediction_rows:
        by_approach.setdefault(str(row["approach"]), []).append(row)
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    for approach, rows in sorted(by_approach.items()):
        fpr, tpr, _thresholds = roc_curve(
            [float(row["score"]) for row in rows],
            [int(row["label"]) for row in rows],
        )
        axis.plot(fpr, tpr, label=f"{approach} · AUC {metrics[approach]['auc']:.3f}")
    axis.plot([0, 1], [0, 1], "--", color="0.6", linewidth=1)
    axis.set(xlabel="false-positive rate", ylabel="true-positive rate", xlim=(0, 1), ylim=(0, 1))
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    figure.savefig(roc_path, dpi=160)
    plt.close(figure)
    return {"metrics": metrics_path, "predictions": predictions_path, "roc": roc_path}
