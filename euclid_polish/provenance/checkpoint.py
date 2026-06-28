"""Checkpoint identity.

A TensorFlow checkpoint's unit of identity is its *directory*, not a single
file. ``<ckpt_dir>/provenance.json`` records the model artifact's
:class:`~euclid_polish.provenance.records.Stamp` — its id, the training run
(``Process`` of kind ``"trainingrun"``) that produced it, and the input record
ids it trained on. Absent the sidecar a
checkpoint is "legacy" (unknown provenance).
"""

from __future__ import annotations

import os

from euclid_polish.provenance._util import _atomic_write_json
from euclid_polish.provenance.ids import ProvId
from euclid_polish.provenance.records import Stamp

PROVENANCE_FILENAME = "provenance.json"


def checkpoint_provenance_path(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, PROVENANCE_FILENAME)


def write_checkpoint_provenance(ckpt_dir: str, stamp: Stamp) -> str:
    """Write (atomically) the checkpoint's provenance stamp; return its path."""
    os.makedirs(ckpt_dir, exist_ok=True)
    path = checkpoint_provenance_path(ckpt_dir)
    _atomic_write_json(path, stamp.to_dict())
    return path


def read_checkpoint_provenance(ckpt_dir: str) -> Stamp | None:
    """The checkpoint's stamp, or ``None`` if it has no provenance sidecar."""
    path = checkpoint_provenance_path(ckpt_dir)
    if not os.path.exists(path):
        return None
    with open(path) as fp:
        return Stamp.from_json(fp.read())


def model_id_of_checkpoint(ckpt_dir: str) -> ProvId | None:
    """The model artifact's :class:`ProvId`, or ``None`` for a legacy checkpoint."""
    stamp = read_checkpoint_provenance(ckpt_dir)
    return stamp.id if stamp is not None else None
