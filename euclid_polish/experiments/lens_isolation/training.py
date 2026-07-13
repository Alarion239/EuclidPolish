"""Virgin source forks and normal fixed-record training dispatch."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from typing import Any

from euclid_polish.experiments.lens_isolation.config import TrainConfig, assert_safe_output
from euclid_polish.image.tfio import tfrecord_path
from euclid_polish.model import Model


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
            digest.update(os.path.relpath(full, path).encode("utf-8"))
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
    """Build a step-zero experiment model from read-only source weights."""
    source = os.path.abspath(source)
    if not os.path.isdir(source):
        raise FileNotFoundError(source)
    target = assert_safe_output(target, source=source, protected_roots=protected_roots)
    if os.path.exists(target) and os.listdir(target):
        raise ValueError(f"fork target must be virgin: {target}")
    source_before = checkpoint_fingerprint(source)
    os.makedirs(target, exist_ok=True)
    try:
        model = model_factory(target, seed=int(seed), init_weights_from=source)
        _write_json_atomic(
            os.path.join(target, "origin.json"),
            {
                "experiment": "lens_isolation",
                "source": source,
                "source_fingerprint": source_before,
                "dataset_fingerprint": dataset_fingerprint,
                "seed": int(seed),
                "initial_step": 0,
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        )
    except Exception:
        if os.path.isdir(target) and not os.listdir(target):
            os.rmdir(target)
        raise
    if checkpoint_fingerprint(source) != source_before:
        raise RuntimeError("source checkpoint changed while it was being forked")
    return model


def train_member(
    model: Model,
    records_dir: str,
    config: TrainConfig,
    *,
    reporter=None,
    member_index: int = 0,
    member_count: int = 1,
) -> None:
    """Use the unchanged normal record-mode training interface.

    The target record is the experiment's only training difference.  Random
    block-aligned crops, augmentation, asinh stretch, optimisation, validation,
    rollback, checkpoint selection, and standard training logs all remain in
    :meth:`Model.train`.
    """
    dirty_train = tfrecord_path(records_dir, "dirty_train")
    lens_train = tfrecord_path(records_dir, "lens_train")

    def step_callback(current_step: int, _total_steps: int) -> None:
        if reporter is not None:
            reporter.set_step(
                member_index * config.steps + int(current_step),
                member_count * config.steps,
                f"member {member_index + 1} step {current_step}",
            )

    def eval_callback(metrics: dict[str, Any]) -> None:
        if reporter is not None:
            reporter.metric({**metrics, "member": member_index + 1})

    model.train(
        lr_path=dirty_train,
        hr_path=lens_train,
        forward_onthefly=False,
        steps=config.steps,
        batch_size=config.batch_size,
        lr_peak=config.lr_peak,
        lr_final=config.lr_final,
        lr_warmup_steps=config.lr_warmup_steps,
        loss_norm=config.loss_norm,
        noise_aug=config.noise_aug,
        bootstrap=config.bootstrap,
        evaluate_every=config.evaluate_every,
        step_callback=step_callback,
        eval_callback=eval_callback,
    )
