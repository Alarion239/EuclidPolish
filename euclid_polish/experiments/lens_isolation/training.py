"""Virgin source forks and a dedicated selective-reconstruction trainer."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from typing import Any

import numpy as np
import tensorflow as tf

from euclid_polish.experiments.lens_isolation.config import assert_safe_output
from euclid_polish.experiments.lens_isolation.loss import LensIsolationLoss
from euclid_polish.model import Model
from euclid_polish.training.lr_schedule import WarmupCosineDecay


def checkpoint_fingerprint(path: str) -> str:
    """SHA-256 over every source-checkpoint byte, in stable path order."""
    digest = hashlib.sha256()
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    found = False
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            if name == "origin.json":
                continue
            found = True
            full = os.path.join(root, name)
            rel = os.path.relpath(full, path)
            digest.update(rel.encode("utf-8"))
            with open(full, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    if not found:
        raise FileNotFoundError(f"no checkpoint files under {path}")
    return digest.hexdigest()


def _write_json_atomic(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=os.path.basename(path) + ".tmp-", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def fork_member(
    source: str,
    target: str,
    *,
    seed: int,
    dataset_fingerprint: str,
    model_factory=Model,
    protected_roots: tuple[str, ...] | None = None,
):
    """Build a step-zero model from read-only source weights."""
    source = os.path.abspath(source)
    if not os.path.isdir(source):
        raise FileNotFoundError(source)
    target = assert_safe_output(
        target,
        source=source,
        protected_roots=protected_roots,
    )
    if os.path.exists(target) and os.listdir(target):
        raise ValueError(f"fork target must be virgin: {target}")
    source_before = checkpoint_fingerprint(source)
    os.makedirs(target, exist_ok=True)
    try:
        model = model_factory(target, seed=int(seed), init_weights_from=source)
        origin = {
            "experiment": "lens_isolation",
            "source": source,
            "source_fingerprint": source_before,
            "dataset_fingerprint": dataset_fingerprint,
            "seed": int(seed),
            "initial_step": 0,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        _write_json_atomic(os.path.join(target, "origin.json"), origin)
    except Exception:
        if os.path.isdir(target) and not os.listdir(target):
            os.rmdir(target)
        raise
    source_after = checkpoint_fingerprint(source)
    if source_after != source_before:
        raise RuntimeError("source checkpoint changed while it was being forked")
    return model


class LensIsolationTrainer:
    """Small dedicated loop with AUC-best and balanced-loss-best tracks."""

    def __init__(
        self,
        model: Model,
        checkpoint_dir: str,
        *,
        steps: int,
        lr_peak: float = 1e-5,
        lr_final: float = 1e-6,
        lr_warmup_steps: int = 500,
        loss: LensIsolationLoss | None = None,
    ) -> None:
        self.model_wrapper = model
        self.model = model._tf_model
        self.loss = loss or LensIsolationLoss()
        self.schedule = WarmupCosineDecay(
            peak_lr=lr_peak,
            final_lr=lr_final,
            warmup_steps=lr_warmup_steps,
            total_steps=steps,
        )
        self.optimizer = tf.keras.optimizers.Adam(lr_peak)
        self.checkpoint = tf.train.Checkpoint(
            step=tf.Variable(0, dtype=tf.int64),
            best_auc=tf.Variable(-1.0),
            best_loss=tf.Variable(float("inf")),
            optimizer=self.optimizer,
            model=self.model,
        )
        self.auc_manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_dir, max_to_keep=3)
        self.loss_manager = tf.train.CheckpointManager(
            self.checkpoint, os.path.join(checkpoint_dir, "loss_best"), max_to_keep=3
        )

    @tf.function
    def _train_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = tf.reduce_mean(self.loss(targets, predictions))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables, strict=True))
        return loss

    def evaluate(self, dataset) -> dict[str, float]:
        losses: list[float] = []
        labels: list[float] = []
        scores: list[float] = []
        for inputs, targets in dataset:
            predictions = self.model(inputs, training=False)
            batch_losses = self.loss(targets, predictions)
            losses.extend(np.asarray(batch_losses).reshape(-1).tolist())
            target_flux = tf.reduce_sum(tf.nn.relu(targets), axis=(1, 2, 3))
            pred_flux = tf.reduce_sum(tf.nn.relu(predictions), axis=(1, 2, 3))
            labels.extend(np.asarray(target_flux > 0, np.float32).tolist())
            scores.extend(np.asarray(pred_flux, np.float32).tolist())
        if not losses:
            raise ValueError("validation dataset is empty")
        auc = tf.keras.metrics.AUC()
        auc.update_state(labels, scores)
        return {"loss": float(np.mean(losses)), "auc": float(auc.result())}

    def train(self, dataset, validation, *, steps: int, evaluate_every: int = 500, callback=None):
        iterator = iter(dataset)
        for step in range(1, int(steps) + 1):
            self.optimizer.learning_rate.assign(self.schedule(step - 1))
            inputs, targets = next(iterator)
            train_loss = float(self._train_step(inputs, targets))
            self.checkpoint.step.assign(step)
            if step % evaluate_every == 0 or step == steps:
                metrics = self.evaluate(validation)
                metrics.update(step=step, train_loss=train_loss)
                if metrics["auc"] > float(self.checkpoint.best_auc):
                    self.checkpoint.best_auc.assign(metrics["auc"])
                    self.auc_manager.save(checkpoint_number=step)
                if metrics["loss"] < float(self.checkpoint.best_loss):
                    self.checkpoint.best_loss.assign(metrics["loss"])
                    self.loss_manager.save(checkpoint_number=step)
                if callback is not None:
                    callback(metrics)
        return self
