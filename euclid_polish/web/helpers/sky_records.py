"""SR status utilities for the sky viewer (the /sky "Generate SR" button).

Tracks whether checkpoints and dirty records are locally available, and
enumerates existing SR cubes. The SR generation itself is done by
:meth:`~euclid_polish.model.Model.upsample_batch`; callers in
``web/routes/views.py`` call that and then persist each SR cube with
:func:`sr_path`.
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional

from euclid_polish.config import Config
from euclid_polish.image.tfio import tfrecord_path
from euclid_polish.provenance.records import Format, SRCutoutArtifact

#: Subsets we generate SR for, in priority order.
SUBSETS = ("validate", "train")


def sky_sr_dir() -> str:
    """Local directory holding generated sky SR cubes (one ``.npy`` each)."""
    return os.path.join(Config.VIS_DIR, "sky_sr")


def sr_path(subset: str, idx: int) -> str:
    return os.path.join(sky_sr_dir(), f"sr_{subset}_{int(idx):04d}.npy")


def sr_count(subset: str) -> int:
    """How many SR cubes have been generated for ``subset``."""
    return len(glob.glob(os.path.join(sky_sr_dir(), f"sr_{subset}_*.npy")))


def checkpoint_present(checkpoint: Optional[str] = None) -> bool:
    """True when a usable checkpoint is on disk (cheap; no TensorFlow import).

    Mirrors ``tf.train.latest_checkpoint`` without paying its import cost:
    a TF checkpoint dir carries a ``checkpoint`` pointer file plus per-step
    ``*.index`` shards.
    """
    ck = checkpoint or Config.DEFAULT_CHECKPOINT_DIR
    if not os.path.isdir(ck):
        return False
    return (os.path.isfile(os.path.join(ck, "checkpoint"))
            or bool(glob.glob(os.path.join(ck, "*.index"))))


def records_present(records_dir: str, subset: str = "validate") -> bool:
    """True when the dirty (LR) TFRecord for ``subset`` is in the local cache."""
    return os.path.exists(tfrecord_path(records_dir, f"dirty_{subset}"))


def present_subsets(records_dir: str) -> List[str]:
    return [s for s in SUBSETS if records_present(records_dir, s)]


def record_sr_cube(store, npy_path: str, subset: str, idx: int, *,
                   model_id=None, input_id=None, produced_by=None,
                   git=None, sidecar_dir: Optional[str] = None) -> SRCutoutArtifact:
    """Persist an :class:`SRCutoutArtifact` for one SR cube, next to the data.

    Parents are ``(model_id, input_id)`` (whichever are known) so the cube can
    later be told apart from a stale one. The sidecar is named by the artifact's
    id; the ``.npy`` keeps its viewer-visible name.
    """
    parents = tuple(p for p in (model_id, input_id) if p is not None)
    art = SRCutoutArtifact(
        id=store.mint(), git=git, produced_by=produced_by,
        format=Format.NPY, path=npy_path, parents=parents,
        descriptors={"subset": subset, "index": int(idx)},
    )
    store.put(art, sidecar_dir=sidecar_dir or sky_sr_dir())
    return art
