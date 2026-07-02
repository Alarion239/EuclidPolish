"""Read the ensemble page's cached per-field cubes so the synthetic evaluator can
reuse an already-computed member stack for a field instead of re-running the CNN.

The cubes are written by ``euclid_polish.web.helpers.ensemble_viz.job_ensemble_evaluate``
into ``euclid_polish.web.helpers.viewer_data._ensemble_cubes_dir()`` — one
``member{i}_{rec:05d}.npy`` per member per evaluated field, plus a ``viz_index.json``
manifest ``{subset, indices, member_labels, ...}``.

Membership staleness: the cube files are keyed by *stack position*, so the
whole cache is only valid for the exact membership that wrote it. On any read,
the manifest's ``member_labels`` are compared with the registry's active
labels; a mismatch (e.g. a member was archived since) deletes the cache dir on
the spot — the lazy invalidation of retired-model caches.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np

from euclid_polish.config import Config
from euclid_polish.ensemble import default_ensemble_dir
from euclid_polish.ensemble_registry import active_labels


def _default_cubes_dir() -> str:
    # Must match euclid_polish.web.helpers.viewer_data._ensemble_cubes_dir().
    return os.path.join(Config.VIS_DIR, "ensemble", "cubes")


def cached_member_labels(cubes_dir: str | None = None) -> list[str] | None:
    """The ``member_labels`` recorded in the cache manifest, or ``None``."""
    d = cubes_dir or _default_cubes_dir()
    try:
        with open(os.path.join(d, "viz_index.json")) as f:
            return [str(x) for x in json.load(f).get("member_labels", [])]
    except (OSError, ValueError, TypeError):
        return None


def load_cached_member_stack(field_index: int, *, subset: str,
                             cubes_dir: str | None = None,
                             active: list[str] | None = None
                             ) -> np.ndarray | None:
    """Return the cached ``(M, H, W, C)`` member stack for ``field_index``, or ``None``.

    Returns ``None`` unless ``<cubes_dir>/viz_index.json`` loads, its ``subset`` equals
    ``subset``, its ``member_labels`` match the registry's ACTIVE labels
    (``active`` overrides the registry lookup, for tests), ``field_index`` is among
    its ``indices``, and every ``member{i}_{field_index:05d}.npy`` exists.
    A membership mismatch DELETES the cache dir (stale, position-keyed) before
    returning ``None``. Never raises — any error degrades to ``None`` so callers
    fall back to inference.
    """
    d = cubes_dir or _default_cubes_dir()
    try:
        with open(os.path.join(d, "viz_index.json")) as f:
            man = json.load(f)
        want = (active if active is not None
                else active_labels(default_ensemble_dir()))
        have = [str(x) for x in man.get("member_labels", []) or []]
        if have != list(want):
            # Archived/added member since this cache was written → purge lazily.
            shutil.rmtree(d, ignore_errors=True)
            return None
        if str(man.get("subset", "")) != str(subset):
            return None
        indices = {int(i) for i in man.get("indices", [])}
        if int(field_index) not in indices:
            return None
        n_members = len(have)
        if n_members <= 0:
            return None
        stack = []
        for i in range(n_members):
            p = os.path.join(d, f"member{i}_{int(field_index):05d}.npy")
            if not os.path.isfile(p):
                return None
            stack.append(np.load(p).astype(np.float32))
        return np.stack(stack, axis=0)
    except (OSError, ValueError, KeyError, TypeError):
        return None
