"""The ensemble's member-cube buckets: cached member SRs the evaluation, the
combiner fits and the synthetic evaluator reuse instead of re-running the CNN.

A bucket is one directory under ``<vis>/ensemble/<regime>/`` — ``cubes``
(test evaluation, written by
``euclid_polish.web.helpers.ensemble_viz.job_ensemble_evaluate``),
``cubes_validate`` (combiner fits), ``cubes_blackout`` and
``cubes_validate_blackout`` (blackout-stamped copies, see
:func:`euclid_polish.eval.spatial_gate_fit.build_blackout_fields`). It holds
one ``member_<key>_<field:05d>.npy`` per member per field (``<key>`` from the
member label, :func:`member_cube_key`: ``196·psnr`` → ``196``), the per-field
aggregates where the bucket has them (``sr_``, ``std_``, ``pcaN_``, combiner
outputs, ``lr_``) and a manifest (``viz_index.json``; ``blackout_index.json``)
recording ``member_labels`` and ``member_fps``: each member's checkpoint
fingerprint (:func:`euclid_polish.ensemble.member_fingerprint`) when its cubes
were made.

Identity: a member's cubes are current while its recorded fingerprint equals
its checkpoint's fingerprint now (``None``, a member without a readable
checkpoint, matches only ``None``). Writers keep every member cube on disk the
product of the checkpoint its manifest records — :func:`sync_bucket_members`
deletes a changed member's cubes before recording its new fingerprint — so a
bucket fills incrementally: only missing cubes are inferred, a departed
member's are dropped, and nothing but a change of the records a bucket was
made from invalidates the others.

Buckets written before the label keying store ``member<position>_<field>.npy``
in the order of the manifest's ``member_labels`` and record no fingerprint.
Readers still resolve them (:func:`bucket_member_path`); writers rename them on
first touch (:func:`migrate_positional_bucket`), adopting a fingerprint only
where the caller proves it — an unproven cube is stale, never served as current.

Membership: a reader asks for the members it wants BY LABEL and gets exactly
those cubes (a pruned production gate reads only its members' files). A
membership mismatch is never destructive here: members the cache lacks are
reported (:func:`missing_cached_members`) and the caller falls back to
inference.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from euclid_polish.config import Config
from euclid_polish.ensemble import default_ensemble_dir, member_fingerprints
from euclid_polish.ensemble_registry import regime_labels
from euclid_polish.image.collection import ImageSet
from euclid_polish.image.tfio import tfrecord_path

#: Manifest of the test and validate buckets.
VIZ_INDEX = "viz_index.json"
#: Manifest of the blackout buckets.
BLACKOUT_INDEX = "blackout_index.json"
#: Manifest key: ``{label: checkpoint fingerprint}`` of the members' cubes.
MEMBER_FPS_KEY = "member_fps"

_POSITIONAL_FILE = re.compile(r"member(\d+)_(\d+)\.npy")
_FIELD_FILE = re.compile(r".+_(\d{5})\.npy")


def _default_cubes_dir(starless: bool = False) -> str:
    # Must match euclid_polish.web.helpers.viewer_data._ensemble_cubes_dir().
    # starfull and starless caches are fully detached; STARFULL is the
    # production regime (starless is opt-in), so it is the default.
    regime = "starless" if starless else "starfull"
    return os.path.join(Config.VIS_DIR, "ensemble", regime, "cubes")


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #

def member_cube_key(label: str) -> str:
    """The file-name key of a member label: ``"196·psnr"`` → ``"196"``; any
    other label keeps its text with every character outside ``[A-Za-z0-9-]``
    replaced by ``-`` (``"196·loss"`` → ``"196-loss"``)."""
    text = str(label).removesuffix("·psnr")
    return re.sub(r"[^A-Za-z0-9-]", "-", text) or "-"


def member_cube_path(cubes_dir: str, label: str, field_index: int) -> str:
    """``<cubes_dir>/member_<key>_<field:05d>.npy`` — one member's cube of one field."""
    return os.path.join(cubes_dir,
                        f"member_{member_cube_key(label)}_{int(field_index):05d}.npy")


def read_bucket_manifest(cubes_dir: str, name: str = VIZ_INDEX) -> dict | None:
    """The bucket's manifest, or ``None`` when it is missing or unreadable."""
    try:
        with open(os.path.join(cubes_dir, name)) as handle:
            manifest = json.load(handle)
    except (OSError, ValueError, TypeError):
        return None
    return manifest if isinstance(manifest, dict) else None


def write_bucket_manifest(cubes_dir: str, manifest: Mapping, name: str = VIZ_INDEX) -> None:
    """Replace the bucket's manifest atomically."""
    path = os.path.join(cubes_dir, name)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as handle:
        json.dump(dict(manifest), handle)
    os.replace(tmp, path)


def manifest_member_labels(manifest: Mapping | None) -> list[str]:
    """The members a bucket holds, in stack order (a positional blackout
    manifest kept them in its ``identity``)."""
    if not manifest:
        return []
    labels = manifest.get("member_labels")
    if labels is None:
        labels = (manifest.get("identity") or {}).get("member_labels")
    return [str(v) for v in labels or []]


def is_label_keyed(manifest: Mapping | None) -> bool:
    return bool(manifest) and isinstance(manifest.get(MEMBER_FPS_KEY), dict)


def recorded_fingerprints(manifest: Mapping | None) -> dict[str, str | None]:
    """``{label: fingerprint its cubes were made with}`` (``None``: unknown —
    every member of a positional bucket)."""
    fps = (manifest or {}).get(MEMBER_FPS_KEY)
    fps = fps if isinstance(fps, dict) else {}
    return {label: fps.get(label) for label in manifest_member_labels(manifest)}


def stale_bucket_members(manifest: Mapping | None,
                         fingerprints: Mapping[str, str | None]) -> list[str]:
    """The bucket's members whose cubes were not made by the checkpoint
    ``fingerprints`` names (every member of a positional bucket, whose
    fingerprints are unknown)."""
    recorded = recorded_fingerprints(manifest)
    return [label for label in manifest_member_labels(manifest)
            if recorded[label] != fingerprints.get(label)]


def bucket_member_path(manifest: Mapping | None, cubes_dir: str, label: str,
                       field_index: int) -> str | None:
    """Where the bucket keeps ``label``'s cube of one field — label-keyed or,
    in a positional bucket, by its position in ``member_labels`` — or
    ``None`` when the bucket does not hold that member. Says nothing about
    whether the file exists."""
    labels = manifest_member_labels(manifest)
    if str(label) not in labels:
        return None
    if is_label_keyed(manifest):
        return member_cube_path(cubes_dir, label, field_index)
    return os.path.join(cubes_dir,
                        f"member{labels.index(str(label))}_{int(field_index):05d}.npy")


def bucket_member_paths(manifest: Mapping | None, cubes_dir: str, labels: Sequence[str],
                        field_index: int) -> list[str | None]:
    return [bucket_member_path(manifest, cubes_dir, label, field_index) for label in labels]


def missing_member_cubes(cubes_dir: str, labels: Sequence[str],
                         field_index: int) -> list[str]:
    """The members of ``labels`` without a cube of ``field_index`` in a
    label-keyed bucket."""
    return [str(label) for label in labels
            if not os.path.isfile(member_cube_path(cubes_dir, label, field_index))]


def _member_files(cubes_dir: str, labels: Iterable[str]) -> list[str]:
    keys = {member_cube_key(label) for label in labels}
    if not keys or not os.path.isdir(cubes_dir):
        return []
    pattern = re.compile(r"member_(" + "|".join(re.escape(k) for k in keys) + r")_\d+\.npy")
    return [os.path.join(cubes_dir, name) for name in os.listdir(cubes_dir)
            if pattern.fullmatch(name)]


def prune_bucket_fields(cubes_dir: str, keep: Iterable[int]) -> None:
    """Delete every per-field file (``<name>_<field:05d>.npy``) of a field not
    in ``keep``."""
    wanted = {int(i) for i in keep}
    if not os.path.isdir(cubes_dir):
        return
    for name in os.listdir(cubes_dir):
        match = _FIELD_FILE.fullmatch(name)
        if match and int(match.group(1)) not in wanted:
            os.remove(os.path.join(cubes_dir, name))


# --------------------------------------------------------------------------- #
# Migration and membership sync (writers)
# --------------------------------------------------------------------------- #

def migrate_positional_bucket(cubes_dir: str, name: str = VIZ_INDEX, *,
                              adopt: Mapping[str, str | None] | None = None) -> dict | None:
    """Rename a positional bucket's ``member<i>_<field>.npy`` to the label
    keying, by the manifest's ``member_labels`` order, and record the members'
    fingerprints: ``adopt[label]`` where the caller proves which checkpoint
    made those cubes, else ``None`` (stale: re-inferred by the next fill).

    Member files of a position beyond the labels or of a field the manifest
    does not list are deleted (no manifest says what made them). A positional
    blackout manifest's ``identity.member_labels`` moves to the top level.
    Returns the manifest (unchanged when already label-keyed; ``None`` when
    there is none)."""
    manifest = read_bucket_manifest(cubes_dir, name)
    if manifest is None or is_label_keyed(manifest):
        return manifest
    labels = manifest_member_labels(manifest)
    listed = {int(i) for i in manifest.get("indices", []) or []}
    for file_name in os.listdir(cubes_dir):
        match = _POSITIONAL_FILE.fullmatch(file_name)
        if match is None:
            continue
        position, field_index = int(match.group(1)), int(match.group(2))
        source = os.path.join(cubes_dir, file_name)
        if position < len(labels) and field_index in listed:
            os.replace(source, member_cube_path(cubes_dir, labels[position], field_index))
        else:
            os.remove(source)
    out = {**manifest, "member_labels": labels,
           MEMBER_FPS_KEY: {label: (adopt or {}).get(label) for label in labels}}
    identity = out.get("identity")
    if isinstance(identity, dict) and "member_labels" in identity:
        out["identity"] = {k: v for k, v in identity.items() if k != "member_labels"}
    write_bucket_manifest(cubes_dir, out, name)
    return out


@dataclass
class BucketSync:
    """What :func:`sync_bucket_members` changed."""

    manifest: dict
    dropped: list[str] = field(default_factory=list)     # left the membership
    refreshed: list[str] = field(default_factory=list)   # checkpoint changed
    added: list[str] = field(default_factory=list)       # new to the bucket

    @property
    def changed(self) -> bool:
        return bool(self.dropped or self.refreshed or self.added)


def sync_bucket_members(cubes_dir: str, manifest: Mapping | None, labels: Sequence[str],
                        fingerprints: Mapping[str, str | None], *,
                        name: str = VIZ_INDEX) -> BucketSync:
    """Make a label-keyed bucket's membership ``labels`` at ``fingerprints``
    without inference: the cubes of members that left, of members whose
    checkpoint changed and any of members the bucket did not list are
    deleted, then the manifest records ``labels`` and their fingerprints. Every
    member cube left on disk is then current; the missing ones are the fill's
    work. A membership change also clears the ``has_combiner*`` flags (the
    combiner outputs were made from the old stack). Writes the manifest.

    A positional bucket must be migrated first
    (:func:`migrate_positional_bucket`): positional member files left in the
    directory are deleted here (nothing names their member any more)."""
    manifest = dict(manifest or {})
    labels = [str(v) for v in labels]
    held = manifest_member_labels(manifest) if is_label_keyed(manifest) else []
    recorded = recorded_fingerprints(manifest) if held else {}
    sync = BucketSync(manifest)
    sync.dropped = [label for label in held if label not in labels]
    sync.refreshed = [label for label in labels
                      if label in recorded and recorded[label] != fingerprints.get(label)]
    sync.added = [label for label in labels if label not in recorded]
    for path in _member_files(cubes_dir, sync.dropped + sync.refreshed + sync.added):
        os.remove(path)
    if os.path.isdir(cubes_dir):
        for file_name in os.listdir(cubes_dir):
            if _POSITIONAL_FILE.fullmatch(file_name):
                os.remove(os.path.join(cubes_dir, file_name))
    manifest["member_labels"] = labels
    manifest[MEMBER_FPS_KEY] = {label: fingerprints.get(label) for label in labels}
    if sync.changed:
        for key in [k for k in manifest if k.startswith("has_combiner")]:
            manifest[key] = False
    os.makedirs(cubes_dir, exist_ok=True)
    write_bucket_manifest(cubes_dir, manifest, name)
    sync.manifest = manifest
    return sync


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #

def cached_member_labels(cubes_dir: str | None = None, *,
                         starless: bool = False) -> list[str] | None:
    """The ``member_labels`` recorded in the cache manifest, or ``None``."""
    manifest = read_bucket_manifest(cubes_dir or _default_cubes_dir(starless))
    return None if manifest is None else manifest_member_labels(manifest)


def _member_cube_paths(man: dict, d: str, field_index: int, *, subset: str,
                       labels: Sequence[str],
                       current: Mapping[str, str | None] | None) -> dict[str, str | None]:
    """``label → cube path`` for one field; ``None`` when the cache lacks it
    or (``current`` given) recorded another checkpoint fingerprint for it."""
    field_ok = (str(man.get("subset", "")) == str(subset)
                and int(field_index) in {int(i) for i in man.get("indices", []) or []})
    recorded = recorded_fingerprints(man)
    out: dict[str, str | None] = {}
    for label in labels:
        path = bucket_member_path(man, d, label, field_index) if field_ok else None
        if path is not None and (not os.path.isfile(path) or (
                current is not None and recorded.get(str(label)) != current.get(str(label)))):
            path = None
        out[str(label)] = path
    return out


def _wanted_labels(active: Sequence[str] | None, starless: bool) -> list[str]:
    return [str(x) for x in (active if active is not None
                             else regime_labels(default_ensemble_dir(), starless))]


def _current(want: Sequence[str], fingerprints: Mapping[str, str | None] | None,
             require_current: bool) -> Mapping[str, str | None] | None:
    if not require_current:
        return None
    if fingerprints is not None:
        return fingerprints
    return member_fingerprints(default_ensemble_dir(), want)


def load_cached_member_stack(field_index: int, *, subset: str,
                             cubes_dir: str | None = None,
                             active: Sequence[str] | None = None,
                             starless: bool = False,
                             fingerprints: Mapping[str, str | None] | None = None,
                             require_current: bool = True,
                             ) -> np.ndarray | None:
    """The cached ``(M, H, W, C)`` stack of the members ``active`` names (by
    label, in that order; default: the ACTIVE labels of the regime — STARFULL
    unless ``starless``), or ``None``.

    Returns ``None`` unless the bucket's manifest loads, its ``subset`` equals
    ``subset``, ``field_index`` is among its ``indices`` and every requested
    member has a cube of it made by its CURRENT checkpoint: the fingerprint the
    manifest records equals ``fingerprints[label]`` (default: the member's
    checkpoint under the ensemble dir now) — a positional bucket records none,
    so it is never served for a member with a checkpoint.
    ``require_current=False`` reads whatever the bucket holds (diagnostics of
    an already-fitted combiner). A cache written for another membership is
    NOT deleted: :func:`missing_cached_members` names what it lacks. Never
    raises — any error degrades to ``None`` so callers fall back to inference.
    """
    d = cubes_dir or _default_cubes_dir(starless)
    try:
        man = read_bucket_manifest(d)
        if man is None:
            return None
        want = _wanted_labels(active, starless)
        if not want:
            return None
        paths = _member_cube_paths(man, d, field_index, subset=subset, labels=want,
                                   current=_current(want, fingerprints, require_current))
        if any(p is None for p in paths.values()):
            return None
        return np.stack([np.load(paths[label]).astype(np.float32) for label in want],
                        axis=0)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def missing_cached_members(field_index: int, *, subset: str,
                           cubes_dir: str | None = None,
                           active: Sequence[str] | None = None,
                           starless: bool = False,
                           fingerprints: Mapping[str, str | None] | None = None,
                           require_current: bool = True) -> list[str]:
    """The requested members (``active``, as for
    :func:`load_cached_member_stack`) without a current cube of
    ``field_index`` — all of them when there is no cache for that field."""
    d = cubes_dir or _default_cubes_dir(starless)
    want = _wanted_labels(active, starless)
    try:
        man = read_bucket_manifest(d)
        if man is None:
            return want
        paths = _member_cube_paths(man, d, field_index, subset=subset, labels=want,
                                   current=_current(want, fingerprints, require_current))
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
