"""Read the ensemble page's cached per-field cubes so the synthetic evaluator can
reuse an already-computed member stack for a field instead of re-running the CNN.

The cubes are written by ``euclid_polish.web.helpers.ensemble_viz.job_ensemble_evaluate``
into ``euclid_polish.web.helpers.viewer_data._ensemble_cubes_dir()`` — one
``member{i}_{rec:05d}.npy`` per member per evaluated field, plus a ``viz_index.json``
manifest ``{subset, indices, member_labels, ...}``.

Membership: the cube files are keyed by *stack position* in the manifest's
``member_labels``, so a reader asks for the members it wants BY LABEL and gets
exactly those cubes (a pruned production gate reads only its members' files).
A membership mismatch is never destructive: members the cache lacks are
reported (:func:`missing_cached_members`) and the caller falls back to
inference; the cache itself is left alone (the ensemble page's evaluation
rewrites it for a new membership).
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

import numpy as np

from euclid_polish.config import Config
from euclid_polish.ensemble import default_ensemble_dir
from euclid_polish.ensemble_registry import regime_labels
from euclid_polish.image.collection import ImageSet
from euclid_polish.image.tfio import tfrecord_path


def _default_cubes_dir(starless: bool = False) -> str:
    # Must match euclid_polish.web.helpers.viewer_data._ensemble_cubes_dir().
    # starfull and starless caches are fully detached; STARFULL is the
    # production regime (starless is opt-in), so it is the default.
    regime = "starless" if starless else "starfull"
    return os.path.join(Config.VIS_DIR, "ensemble", regime, "cubes")


def cached_member_labels(cubes_dir: str | None = None, *,
                         starless: bool = False) -> list[str] | None:
    """The ``member_labels`` recorded in the cache manifest, or ``None``."""
    d = cubes_dir or _default_cubes_dir(starless)
    try:
        with open(os.path.join(d, "viz_index.json")) as f:
            return [str(x) for x in json.load(f).get("member_labels", [])]
    except (OSError, ValueError, TypeError):
        return None


def _read_manifest(d: str) -> dict | None:
    try:
        with open(os.path.join(d, "viz_index.json")) as f:
            man = json.load(f)
    except (OSError, ValueError, TypeError):
        return None
    return man if isinstance(man, dict) else None


def _member_cube_paths(man: dict, d: str, field_index: int, *, subset: str,
                       labels: Sequence[str]) -> dict[str, str | None]:
    """``label → cube path`` (``None`` when the cache lacks it) for one field."""
    have = {str(x): i for i, x in enumerate(man.get("member_labels", []) or [])}
    field_ok = (str(man.get("subset", "")) == str(subset)
                and int(field_index) in {int(i) for i in man.get("indices", []) or []})
    out: dict[str, str | None] = {}
    for label in labels:
        pos = have.get(str(label))
        path = (os.path.join(d, f"member{pos}_{int(field_index):05d}.npy")
                if field_ok and pos is not None else None)
        out[str(label)] = path if path is not None and os.path.isfile(path) else None
    return out


def _wanted_labels(active: Sequence[str] | None, starless: bool) -> list[str]:
    return [str(x) for x in (active if active is not None
                             else regime_labels(default_ensemble_dir(), starless))]


def load_cached_member_stack(field_index: int, *, subset: str,
                             cubes_dir: str | None = None,
                             active: Sequence[str] | None = None,
                             starless: bool = False
                             ) -> np.ndarray | None:
    """The cached ``(M, H, W, C)`` stack of the members ``active`` names (by
    label, in that order; default: the ACTIVE labels of the regime — STARFULL
    unless ``starless``), or ``None``.

    Returns ``None`` unless ``<cubes_dir>/viz_index.json`` loads, its ``subset``
    equals ``subset``, ``field_index`` is among its ``indices`` and every
    requested member has its ``member{i}_{field_index:05d}.npy`` (``i`` = its
    position in the manifest's ``member_labels``). A cache written for another
    membership is NOT deleted: :func:`missing_cached_members` names what it
    lacks. Never raises — any error degrades to ``None`` so callers fall back
    to inference.
    """
    d = cubes_dir or _default_cubes_dir(starless)
    try:
        man = _read_manifest(d)
        if man is None:
            return None
        want = _wanted_labels(active, starless)
        if not want:
            return None
        paths = _member_cube_paths(man, d, field_index, subset=subset, labels=want)
        if any(p is None for p in paths.values()):
            return None
        return np.stack([np.load(paths[label]).astype(np.float32) for label in want],
                        axis=0)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def missing_cached_members(field_index: int, *, subset: str,
                           cubes_dir: str | None = None,
                           active: Sequence[str] | None = None,
                           starless: bool = False) -> list[str]:
    """The requested members (``active``, as for
    :func:`load_cached_member_stack`) whose cube for ``field_index`` the cache
    lacks — all of them when there is no cache for that field."""
    d = cubes_dir or _default_cubes_dir(starless)
    want = _wanted_labels(active, starless)
    try:
        man = _read_manifest(d)
        if man is None:
            return want
        paths = _member_cube_paths(man, d, field_index, subset=subset, labels=want)
    except (OSError, ValueError, KeyError, TypeError):
        return want
    return [label for label in want if paths.get(label) is None]


def cached_field_lr_path(cubes_dir: str, field_index: int) -> str:
    return os.path.join(cubes_dir, f"lr_{int(field_index):05d}.npy")


def save_cached_field_lr(cubes_dir: str, field_index: int, lr: np.ndarray) -> None:
    """Store a field's LR input next to its member cubes (for combiners that
    read the LR, e.g. the spatial gate's blackout mask)."""
    np.save(cached_field_lr_path(cubes_dir, field_index),
            np.asarray(lr, np.float32))


def load_cached_field_lr(cubes_dir: str, field_index: int, *,
                         records_dir: str | None, subset: str) -> np.ndarray | None:
    """The ``(h, w, C)`` LR input of a cached field.

    Reads ``lr_{index}.npy`` from the cube bucket; buckets written before LR
    inputs were cached fall back to the ``dirty_{subset}`` records (the file
    is then written so the next read is direct). ``None`` when neither exists.
    """
    path = cached_field_lr_path(cubes_dir, field_index)
    if os.path.isfile(path):
        return np.load(path)
    if not records_dir:
        return None
    records = tfrecord_path(records_dir, f"dirty_{subset}")
    if not os.path.isfile(records):
        return None
    for image in ImageSet.read(records, num_images=int(field_index) + 1):
        if image.index == int(field_index):
            lr = np.asarray(image.data, np.float32)
            save_cached_field_lr(cubes_dir, field_index, lr)
            return lr
    return None
