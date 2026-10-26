"""Runnable STARFULL model specs and their cached per-tile outputs (C9).

A **model spec** names one way of turning a real LR tile into an SR image:

====================  ========================================================
``production``        the production model: the registered universal model
                      the registry names production
                      (:func:`~euclid_polish.ensemble_registry.production_name`,
                      run as an ensemble of one; unavailable with the reason
                      when it cannot run), else the production combiner — the
                      spatial gate (``ACTIVE_COMBINER_KINDS[0]``); available
                      while every member it READS is an active STARFULL
                      member with a checkpoint (members that joined after the
                      fit are a ``note``), unavailable otherwise (no
                      fallback)
``mean``              plain mean of every active STARFULL member
``member:<name>``     one active member, e.g. ``member:member_170``
``gate:<variant>``    a named spatial-gate variant (``spatial_gate_<variant>/``
                      beside the production artifact), applied with its OWN
                      member labels; unavailable (with the reason) unless every
                      member it reads is an active STARFULL member
``rbf``               the legacy RBF combiner, with its own member labels
``universal:<name>``  one registered UNIVERSAL model, e.g.
                      ``universal:member_211`` — a single-image multi-knee
                      model pulled to compare on its own (registry class
                      ``universal``), never an ensemble member: not in the
                      mean, the gates or ``members``; unavailable (with the
                      reason) until its checkpoint and ``origin.json`` are on
                      disk
====================  ========================================================

and ``milestone:<name>@<N>``: one milestone checkpoint of a long run, e.g.
``milestone:member_257@2000000`` — the state a member kept at step N
(:mod:`~euclid_polish.web.helpers.milestones`), listed once pulled; it runs as
an ensemble of one from its own ensemble dir and is addressed by the label
``257@2000000`` (never a member label).

Catalogue order: production, mean, rbf, the members, the gate variants, the
universal models, the milestones (by member, then step). A universal model or
an active member that is one (195/196 — its kind stays ``member``) carries
``details`` read from its ``origin.json`` and ``training_log.csv``
(:func:`member_details`; cached by file stamp, never a TensorFlow read).

Members are always run through :class:`~euclid_polish.ensemble.EnsembleModel`
(ensemble-only rule: a single model is an ensemble of one) and addressed by
their ``NN·psnr`` labels (:class:`EnsembleMemberRunner`); a universal model
shares the members' label and directory namespace (``<ensemble>/member_NNN``)
and runs as an ensemble of one (``include_universal=True``).

Every spec has a **fingerprint**: a hash of the checkpoint identities of the
members it READS (:func:`euclid_polish.ensemble.member_fingerprint`; a pruned
gate's active members only, so retraining a member it ignores changes
nothing) and, for combiners, of the fitted artifact (``combiner.json`` +
``combiner.npz``). A cached output is *current* while its recorded
fingerprint equals the spec's fingerprint now.

The **output store** keeps one SR per (real tile, spec) under
``<EUCLID_INFERENCE_DIR>/experiments/outputs/<source>/<id>/<slug>.fits`` —
``(4, 2H, 2W)`` electrons with the LR WCS magnified ×2 — plus a
``<slug>.json`` sidecar (spec, fingerprint, members, LR hash, metrics).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from euclid_polish import ensemble_registry
from euclid_polish.config import Config
from euclid_polish.ensemble import EnsembleModel, member_fingerprint
from euclid_polish.eval.combiner import (
    ACTIVE_COMBINER_KINDS,
    COMBINER_MODELS,
    RAW_INCREMENTAL_MINMEANMAX_RBF_KIND,
    combiner_artifact_fingerprint,
    load_combiner,
)
from euclid_polish.eval.spatial_gate import SpatialGateCombiner, load_spatial_gate
from euclid_polish.web.helpers import atomic_files, milestones

SPEC_PRODUCTION = "production"
SPEC_MEAN = "mean"
SPEC_RBF = "rbf"
MEMBER_PREFIX = "member:"
GATE_PREFIX = "gate:"
MILESTONE_PREFIX = "milestone:"
UNIVERSAL_PREFIX = "universal:"
GATE_DIR_PREFIX = "spatial_gate_"
PRODUCTION_KIND = ACTIVE_COMBINER_KINDS[0]
PRODUCTION_ARTIFACT_DIR = COMBINER_MODELS[PRODUCTION_KIND].artifact_dir
RBF_KIND = RAW_INCREMENTAL_MINMEANMAX_RBF_KIND
#: SR grid magnification of every STARFULL member.
SR_FACTOR = int(Config.DEFAULT_REBIN_FACTOR)

_MEMBER_DIGITS = re.compile(r"^\d{1,6}$")
_VARIANT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
_MILESTONE_SPEC = re.compile(r"^(?:member_)?(\d{1,6})@(\d{1,12})$")
_MILESTONE_LABEL = re.compile(r"^(\d{1,6})@(\d{1,12})$")


# ---------------------------------------------------------------------------
# the STARFULL ensemble
# ---------------------------------------------------------------------------

def regime_dir() -> Path:
    """Combiner artifacts of the STARFULL regime (``<vis>/ensemble/starfull``)."""
    return Path(Config.VIS_DIR) / "ensemble" / "starfull"


def ensemble_dir() -> str:
    return ensemble_registry.default_ensemble_dir()


def active_member_labels() -> list[str]:
    """``NN·psnr`` labels of the registry-active STARFULL members, in order."""
    return list(ensemble_registry.regime_labels(ensemble_dir(), False))


def member_name(label: str) -> str:
    """``"170·psnr"`` → ``"member_170"``."""
    return "member_" + str(label).split("·")[0].removeprefix("member_")


def member_label(name: str) -> str:
    """``"member_170"`` → ``"170·psnr"``."""
    return str(name).removeprefix("member_").split("·")[0] + "·psnr"


def universal_labels() -> list[str]:
    """``NN·psnr`` labels of every REGISTERED universal model, in name order —
    on disk or not (one still being pulled lists as unavailable). Never
    ensemble members (:mod:`euclid_polish.ensemble_registry`)."""
    return [member_label(name)
            for name in ensemble_registry.universal_names(ensemble_dir())]


def member_fingerprints(labels: Iterable[str]) -> dict[str, str | None]:
    """Checkpoint identity of each member or universal model — both live in
    ``<ensemble>/member_NNN`` (``None`` without a checkpoint)."""
    base = ensemble_dir()
    return {label: (member_fingerprint(milestone_dir(label)) if is_milestone_label(label)
                    else member_fingerprint(os.path.join(base, member_name(label))))
            for label in labels}


def milestone_label(member: str, step: int) -> str:
    """``("member_257", 2000000)`` → ``"257@2000000"``."""
    return f"{str(member).removeprefix('member_')}@{int(step)}"


def is_milestone_label(label: str) -> bool:
    return bool(_MILESTONE_LABEL.fullmatch(str(label)))


def milestone_of(label: str) -> tuple[str, int]:
    """``"257@2000000"`` → ``("member_257", 2000000)``."""
    match = _MILESTONE_LABEL.fullmatch(str(label))
    if not match:
        raise ValueError(f"not a milestone label: {label!r}")
    return f"member_{match.group(1)}", int(match.group(2))


def milestone_dir(label: str) -> str:
    """The pulled checkpoint dir of a milestone label."""
    member, step = milestone_of(label)
    return str(milestones.local_root() / member / "milestones" / milestones.step_dirname(step))


def _step_text(step: int) -> str:
    """``200000`` → ``0.2M``, ``15000000`` → ``15M``, ``50000`` → ``50k``."""
    if step >= 100_000:
        return f"{step / 1e6:.2f}".rstrip("0").rstrip(".") + "M"
    return f"{step // 1000}k" if step >= 1000 else str(step)


# ---------------------------------------------------------------------------
# a model's recipe and training log (cheap reads, cached by file stamp)
# ---------------------------------------------------------------------------

_BASELINE = frozenset({"1", "1.0", "true", "True"})


def _stamp(path: str) -> tuple[int, int] | None:
    try:
        info = os.stat(path)
    except OSError:
        return None
    return info.st_mtime_ns, info.st_size


@lru_cache(maxsize=256)
def _origin_cached(path: str, stamp: tuple[int, int] | None) -> dict[str, Any] | None:
    if stamp is None:
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _origin(member_dir: str) -> dict[str, Any] | None:
    """A model's ``origin.json`` (the shared cached dict — never mutate it)."""
    path = os.path.join(member_dir, "origin.json")
    return _origin_cached(path, _stamp(path))


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@lru_cache(maxsize=256)
def _training_log_cached(path: str, stamp: tuple[int, int] | None
                         ) -> tuple[int | None, float | None, int]:
    if stamp is None:
        return None, None, 0
    steps: int | None = None
    seconds = 0.0
    rows = 0
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("is_baseline") or "").strip() in _BASELINE:
                    continue                 # a resume marker: no training in it
                step = _finite(row.get("step"))
                if step is None:
                    continue
                rows += 1
                steps = int(step) if steps is None else max(steps, int(step))
                duration = _finite(row.get("duration_s"))
                if duration is not None and duration >= 0:
                    seconds += duration
    except (OSError, csv.Error):
        return None, None, 0
    return steps, (round(seconds / 3600.0, 4) if rows else None), rows


def training_log_summary(member_dir: str) -> dict[str, Any]:
    """``{steps, hours, rows}`` of a model's ``training_log.csv``, over its
    non-baseline rows: the furthest logged step, Σ ``duration_s`` in hours
    (rolled-back windows included, never deduplicated; 4 decimals) and how
    many rows (``None`` / ``None`` / ``0`` without a log or rows). The same
    definition as ``universal_models.train_time``."""
    path = os.path.join(member_dir, "training_log.csv")
    steps, hours, rows = _training_log_cached(path, _stamp(path))
    return {"steps": steps, "hours": hours, "rows": rows}


def _int(value: Any) -> int | None:
    number = _finite(value)
    return None if number is None else int(number)


def _universal_details(member_dir: str, origin: Mapping[str, Any] | None,
                       cls: str) -> dict[str, Any]:
    recipe = origin or {}
    log = training_log_summary(member_dir)
    output_knee = _finite(recipe.get("output_knee"))
    knee = ("learned" if recipe.get("learned_output_knee")
            else "fixed" if output_knee is not None else None)
    return {
        "universal": True, "class": cls,
        "num_res_blocks": _int(recipe.get("num_res_blocks")),
        "steps": log["steps"],
        "target_steps": _int(recipe.get("target_steps")),
        "batch_size": _int(recipe.get("batch_size")),
        "crops_per_field": _int(recipe.get("crops_per_field")),
        "knee": knee,
        "output_knee": None if knee == "learned" else output_knee,
        "jit_compile": bool(recipe.get("jit_compile", False)),
        "mixed_precision": bool(recipe.get("mixed_precision", False)),
        "train_hours": log["hours"],
    }


def member_details(label: str, cls: str) -> dict[str, Any]:
    """The ``details`` of a universal model (``cls`` =
    ``ensemble_registry.UNIVERSAL_CLASS``) or of an active member
    (``MEMBER_CLASS``): ``{universal, class, num_res_blocks, steps,
    target_steps, batch_size, crops_per_field, knee: learned|fixed,
    output_knee, jit_compile, mixed_precision, train_hours}`` from its
    ``origin.json`` and ``training_log.csv`` (``steps``/``train_hours`` as
    :func:`training_log_summary`). A member that is not a universal model
    gets ``{}``. Cheap reads, cached by file stamp — never TensorFlow."""
    member_dir = os.path.join(ensemble_dir(), member_name(label))
    origin = _origin(member_dir)
    if cls != ensemble_registry.UNIVERSAL_CLASS and not ensemble_registry.is_universal_origin(origin):
        return {}
    return _universal_details(member_dir, origin, cls)


def _universal_reason(label: str, fingerprint: str | None) -> str | None:
    """Why a registered universal model cannot run now (``None`` when it can)."""
    if not fingerprint:
        return "universal model has no checkpoint (not pulled yet)"
    origin = _origin(os.path.join(ensemble_dir(), member_name(label)))
    if origin is None:
        # Without its recipe a multi-knee checkpoint would load single-knee.
        return "universal model has no origin.json yet (its knees are unknown)"
    if origin.get("starless"):
        return "universal model is starless (real tiles run STARFULL models only)"
    return None


def runnable_universal_labels(labels: Iterable[str]) -> list[str]:
    """``labels`` that are registered universal models able to run now (a
    checkpoint and a STARFULL recipe on disk), in the order given."""
    wanted = list(dict.fromkeys(labels))
    registered = set(universal_labels()) if wanted else set()
    candidates = [label for label in wanted if label in registered]
    fps = member_fingerprints(candidates)
    return [label for label in candidates if _universal_reason(label, fps.get(label)) is None]


# ---------------------------------------------------------------------------
# spec names
# ---------------------------------------------------------------------------

def _numbered_spec(raw: str, text: str, prefix: str) -> str:
    """``<prefix>member_<N>`` from any spelling of the member after ``prefix``."""
    name = text[len(prefix):].strip().removeprefix("member_").split("·")[0]
    if not _MEMBER_DIGITS.fullmatch(name):
        kind = prefix.removesuffix(":")
        raise ValueError(f"bad {kind} spec {raw!r} (use {prefix}member_<N>)")
    return f"{prefix}member_{name}"


def canonical_spec(raw: str) -> str:
    """Normalise a spec string (aliases: ``member:170``, ``member:170·psnr``,
    ``universal:211``, ``universal:211·psnr``, ``gate:spatial_gate_26m``);
    :class:`ValueError` when malformed."""
    text = str(raw or "").strip()
    lower = text.lower()
    if lower in (SPEC_PRODUCTION, SPEC_MEAN, SPEC_RBF):
        return lower
    if lower.startswith(MEMBER_PREFIX):
        return _numbered_spec(raw, text, MEMBER_PREFIX)
    if lower.startswith(UNIVERSAL_PREFIX):
        return _numbered_spec(raw, text, UNIVERSAL_PREFIX)
    if lower.startswith(MILESTONE_PREFIX):
        match = _MILESTONE_SPEC.fullmatch(text[len(MILESTONE_PREFIX):].strip())
        if not match:
            raise ValueError(f"bad milestone spec {raw!r} (use milestone:member_<N>@<step>)")
        return f"{MILESTONE_PREFIX}member_{match.group(1)}@{int(match.group(2))}"
    if lower.startswith(GATE_PREFIX):
        name = text[len(GATE_PREFIX):].strip().removeprefix(GATE_DIR_PREFIX)
        if name == PRODUCTION_ARTIFACT_DIR.removeprefix(GATE_DIR_PREFIX):
            return SPEC_PRODUCTION
        if not _VARIANT.fullmatch(name):
            raise ValueError(f"bad gate spec {raw!r} (use gate:<variant>)")
        return f"{GATE_PREFIX}{name}"
    raise ValueError(
        f"unknown model spec {raw!r}: use production, mean, rbf, "
        "member:<name>, gate:<variant>, universal:<name> or milestone:<name>@<step>")


def parse_specs(raw: str | Iterable[str]) -> list[str]:
    """Canonical, de-duplicated specs from a comma list (order kept)."""
    items = raw.split(",") if isinstance(raw, str) else list(raw)
    out: list[str] = []
    for item in items:
        if not str(item).strip():
            continue
        spec = canonical_spec(item)
        if spec not in out:
            out.append(spec)
    return out


def spec_slug(spec: str) -> str:
    """Filesystem-safe name of a canonical spec (``gate:26m`` → ``gate-26m``,
    ``universal:member_211`` → ``universal-member_211``,
    ``milestone:member_257@200000`` → ``milestone-member_257-200000``)."""
    return canonical_spec(spec).replace(":", "-").replace("@", "-")


_PREFIX_KINDS = ((MEMBER_PREFIX, "member"), (GATE_PREFIX, "gate"),
                 (UNIVERSAL_PREFIX, "universal"), (MILESTONE_PREFIX, "milestone"))


def spec_kind(spec: str) -> str:
    """The kind of a canonical spec (``production`` | ``mean`` | ``rbf`` |
    ``member`` | ``gate`` | ``universal`` | ``milestone``); an unknown spec is
    its own kind."""
    text = str(spec)
    return next((kind for prefix, kind in _PREFIX_KINDS if text.startswith(prefix)), text)


def spec_member_label(spec: str) -> str | None:
    """The ``NN·psnr`` label a ``member:`` / ``universal:`` spec runs, the
    ``NN@step`` label of a ``milestone:`` spec (``None`` for every other
    spec, a malformed one included)."""
    try:
        text = canonical_spec(spec)
    except ValueError:
        return None
    if text.startswith(MILESTONE_PREFIX):
        member, step = text[len(MILESTONE_PREFIX):].split("@")
        return milestone_label(member, int(step))
    for prefix in (MEMBER_PREFIX, UNIVERSAL_PREFIX):
        if text.startswith(prefix):
            return member_label(text[len(prefix):])
    return None


def spec_label(spec: str) -> str:
    """The catalogue label of a ``member:`` / ``universal:`` spec
    (``Member 170·psnr``, ``Universal 211``) — also for one no longer in the
    catalogue; any other spec is its own label."""
    label = spec_member_label(spec)
    if label is None:
        return canonical_spec(spec)
    if is_milestone_label(label):
        member, step = milestone_of(label)
        return f"Milestone {member.removeprefix('member_')} @ {_step_text(step)}"
    if spec_kind(canonical_spec(spec)) == "universal":
        return f"Universal {label.split('·')[0]}"
    return f"Member {label}"


def array_sha(array: np.ndarray) -> str:
    """Content hash of an LR input (float32, C order) — keys cached outputs."""
    data = np.ascontiguousarray(np.asarray(array, np.float32))
    digest = hashlib.sha256(str(data.shape).encode("utf-8"))
    digest.update(data.tobytes())
    return digest.hexdigest()[:16]


def _hash(*parts: Any) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


def spec_fingerprint(kind: str, *, combiner_kind: str | None = None,
                     combiner_fingerprint: str | None = None,
                     member_labels: Sequence[str] = (),
                     member_fingerprints: Sequence[str | None] = ()) -> str | None:
    """The fingerprint of a spec of ``kind`` built from that identity.

    The one formula behind :func:`list_specs` — also used to rebuild the
    fingerprint of a legacy SR (NEXUS / pair inference records) from the
    identity it recorded. ``None`` when a member has no checkpoint identity
    or a combiner spec has no artifact hash.
    """
    labels = [str(label) for label in member_labels]
    fps = list(member_fingerprints)
    if len(fps) != len(labels) or any(not fp for fp in fps):
        return None
    pairs = [f"{label}={fp}" for label, fp in zip(labels, fps, strict=True)]
    if kind in ("member", "universal", "milestone"):
        return _hash(kind, labels[0], fps[0]) if len(labels) == 1 else None
    if kind == "mean":
        return _hash("mean", *pairs) if labels else None
    if not combiner_fingerprint or not combiner_kind:
        return None
    return _hash(kind, combiner_kind, combiner_fingerprint, *pairs)


# ---------------------------------------------------------------------------
# the catalogue
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """One runnable (or explained-unavailable) model spec."""

    spec: str
    kind: str                       # production | mean | member | gate | rbf | universal | milestone
    label: str
    member_labels: tuple[str, ...]  # the members it was built for (staleness key)
    reads: tuple[str, ...]          # the members it actually runs
    available: bool
    reason: str | None = None
    fingerprint: str | None = None
    member_fingerprints: tuple[str | None, ...] = ()
    combiner_kind: str | None = None
    combiner_dir: str | None = None
    combiner_fingerprint: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    note: str | None = None         # a soft remark (e.g. members joined after the fit)

    @property
    def slug(self) -> str:
        return spec_slug(self.spec)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec, "kind": self.kind, "label": self.label,
            "slug": self.slug,
            "members": list(self.member_labels),
            "member_names": [member_name(label) for label in self.member_labels],
            "reads": list(self.reads),
            # How many members the spec RUNS (a pruned gate reads a subset of
            # the members it was fitted with); ``n_fitted`` keeps that total.
            "n_members": len(self.reads),
            "n_fitted": len(self.member_labels),
            "available": self.available, "reason": self.reason, "note": self.note,
            "fingerprint": self.fingerprint,
            "member_fingerprints": list(self.member_fingerprints),
            "combiner_kind": self.combiner_kind,
            "combiner_fingerprint": self.combiner_fingerprint,
            "details": dict(self.details),
        }


def _read_manifest(directory: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads((directory / "combiner.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _fit_summary(manifest: Mapping[str, Any], directory: Path) -> dict[str, Any]:
    meta = manifest.get("fit_meta") or {}
    meta = meta if isinstance(meta, Mapping) else {}
    try:
        fitted = datetime.fromtimestamp((directory / "combiner.npz").stat().st_mtime,
                                        UTC).isoformat()
    except OSError:
        fitted = None
    summary = {
        "mix_space": manifest.get("mix_space", "asinh" if manifest.get("kind") == PRODUCTION_KIND
                                  else None),
        "use_lr": manifest.get("use_lr"),
        "width": manifest.get("width"),
        "active_members": manifest.get("active_members"),
        "fitted_at": fitted,
        "artifact_dir": directory.name,
    }
    for key in ("loss", "loss_knees_e", "steps", "subset", "num_images",
                "installed_from", "fit_seconds", "complete", "used_threshold"):
        if key in meta:
            summary[key] = meta[key]
    return summary


def _membership_reason(needed: Sequence[str], active: Sequence[str],
                       fps: Mapping[str, str | None]) -> str | None:
    missing = [label for label in needed if label not in set(active)]
    if missing:
        return ("members not in the active STARFULL ensemble: "
                + ", ".join(missing))
    unchecked = [label for label in needed if not fps.get(label)]
    if unchecked:
        return "members without a checkpoint: " + ", ".join(unchecked)
    return None


def _combiner_spec(*, spec: str, kind: str, label: str, directory: Path,
                   fitted_for_current: bool, active: Sequence[str],
                   fps: Mapping[str, str | None]) -> ModelSpec:
    """A combiner-backed spec (production / gate variant / rbf)."""
    manifest = _read_manifest(directory)
    if manifest is None:
        return ModelSpec(spec, kind, label, (), (), False,
                         reason=f"{directory.name} is not fitted",
                         combiner_kind=None, combiner_dir=str(directory))
    labels = tuple(str(value) for value in manifest.get("member_labels") or [])
    active_members = manifest.get("active_members")
    combiner_kind = str(manifest.get("kind") or kind)
    # Only a spatial gate can skip (prune) members; :func:`predict` feeds
    # every other combiner all of its members.
    reads = (tuple(labels[int(i)] for i in active_members)
             if isinstance(active_members, list) and combiner_kind == PRODUCTION_KIND
             else labels)
    details = dict(_fit_summary(manifest, directory))
    is_gate = combiner_kind == PRODUCTION_KIND
    # A spatial gate's math depends only on the members it reads: it is
    # valid while those are active, however many members joined since its
    # fit; an RBF keeps needing its exact fitted membership for production.
    joined = [lb for lb in active if lb not in set(labels)] if is_gate else []
    note = (f"{len(joined)} member(s) joined after this fit; refit to consider them"
            if joined else None)
    if joined:
        details["joined_after_fit"] = joined
    if fitted_for_current and not is_gate and list(labels) != list(active):
        diff = sorted(set(active) ^ set(labels))
        reason = (f"the production combiner was fitted for {len(labels)} members; the "
                  f"active STARFULL ensemble has {len(active)} (differs in "
                  f"{', '.join(diff[:6])}{'…' if len(diff) > 6 else ''})")
        return ModelSpec(spec, kind, label, labels, reads, False, reason=reason,
                         combiner_kind=combiner_kind, combiner_dir=str(directory),
                         details=details)
    member_fps = {**fps, **member_fingerprints([lb for lb in labels if lb not in fps])}
    reason = _membership_reason(reads, active, member_fps)
    combiner = _load_combiner(str(directory), labels, combiner_kind)
    if reason is None and combiner is None:
        reason = f"{directory.name} is not a loadable {combiner_kind} artifact"
    artifact_fp = combiner_artifact_fingerprint(str(directory.parent), directory.name)
    fingerprint = None
    if reason is None and artifact_fp:
        fingerprint = spec_fingerprint(
            kind, combiner_kind=combiner_kind, combiner_fingerprint=artifact_fp,
            member_labels=reads, member_fingerprints=[member_fps.get(lb) for lb in reads])
    elif reason is None:
        reason = f"{directory.name} artifact is incomplete"
    return ModelSpec(
        spec, kind, label, labels, reads, reason is None, reason=reason,
        fingerprint=fingerprint,
        member_fingerprints=tuple(member_fps.get(lb) for lb in labels),
        combiner_kind=combiner_kind, combiner_dir=str(directory),
        combiner_fingerprint=artifact_fp, details=details, note=note)


@lru_cache(maxsize=32)
def _load_cached(directory: str, labels: tuple[str, ...], kind: str, stamp: float):
    del stamp  # part of the cache key only (artifact mtime)
    if kind == PRODUCTION_KIND:
        return load_spatial_gate(directory, member_labels=list(labels))
    path = Path(directory)
    return load_combiner(str(path.parent), member_labels=list(labels),
                         artifact_dir=path.name)


def _load_combiner(directory: str, labels: Sequence[str], kind: str):
    try:
        stamp = max(os.path.getmtime(os.path.join(directory, name))
                    for name in ("combiner.json", "combiner.npz"))
    except OSError:
        return None
    return _load_cached(directory, tuple(labels), kind, stamp)


def production_model_name() -> str | None:
    """The universal model that is production (``"member_238"``), or ``None``
    when production is the production gate."""
    return ensemble_registry.production_name(ensemble_dir())


def is_single_model(spec: ModelSpec) -> bool:
    """Whether ``spec`` runs one model and no combiner (a member, a universal
    model, a milestone, or production when it is a universal model)."""
    return spec.kind in ("member", "universal", "milestone") or (
        spec.kind == "production" and spec.combiner_dir is None and len(spec.reads) == 1)


def _universal_production_spec(name: str) -> ModelSpec:
    """``production`` as the universal model ``name`` (an ensemble of one)."""
    label = member_label(name)
    universal_spec = f"{UNIVERSAL_PREFIX}{member_name(label)}"
    fp = member_fingerprints([label]).get(label)
    reason = (None if label in set(universal_labels())
              else f"{name} is no longer a registered universal model")
    reason = reason or _universal_reason(label, fp)
    details = {**member_details(label, ensemble_registry.UNIVERSAL_CLASS),
               "universal": True, "model_spec": universal_spec, "member": member_name(label)}
    return ModelSpec(
        SPEC_PRODUCTION, "production", f"Production · {spec_label(universal_spec)}",
        (label,), (label,), reason is None,
        reason=None if reason is None else f"production model {name}: {reason}",
        fingerprint=(spec_fingerprint("universal", member_labels=(label,),
                                      member_fingerprints=(fp,))
                     if reason is None else None),
        member_fingerprints=(fp,), details=details)


def list_specs() -> list[ModelSpec]:
    """Every model spec: production, mean, rbf, each active member, each
    gate variant, each registered universal model (unavailable ones carry a
    ``reason``). Read-only; never opens a checkpoint (no TensorFlow)."""
    active = active_member_labels()
    fps = member_fingerprints(active)
    root = regime_dir()
    specs: list[ModelSpec] = []

    universal_production = production_model_name()
    if universal_production is not None:
        production = _universal_production_spec(universal_production)
    else:
        production = _combiner_spec(
            spec=SPEC_PRODUCTION, kind="production",
            label=f"Production · {COMBINER_MODELS[PRODUCTION_KIND].label}",
            directory=root / PRODUCTION_ARTIFACT_DIR, fitted_for_current=True,
            active=active, fps=fps)
        if not active:
            production = ModelSpec(SPEC_PRODUCTION, "production", production.label, (), (),
                                   False, reason="no active STARFULL members")
    specs.append(production)

    mean_reason = (_membership_reason(active, active, fps) if active
                   else "no active STARFULL members")
    specs.append(ModelSpec(
        SPEC_MEAN, "mean", f"Mean of {len(active)} STARFULL members",
        tuple(active), tuple(active), mean_reason is None, reason=mean_reason,
        fingerprint=(spec_fingerprint("mean", member_labels=active,
                                      member_fingerprints=[fps.get(lb) for lb in active])
                     if mean_reason is None else None),
        member_fingerprints=tuple(fps.get(lb) for lb in active)))

    specs.append(_combiner_spec(
        spec=SPEC_RBF, kind="rbf",
        label=f"RBF · {COMBINER_MODELS[RBF_KIND].label}",
        directory=root / COMBINER_MODELS[RBF_KIND].artifact_dir,
        fitted_for_current=False, active=active, fps=fps))

    for label in active:
        fp = fps.get(label)
        spec = f"{MEMBER_PREFIX}{member_name(label)}"
        specs.append(ModelSpec(
            spec, "member", spec_label(spec),
            (label,), (label,), bool(fp),
            reason=None if fp else "member has no checkpoint",
            fingerprint=spec_fingerprint("member", member_labels=(label,),
                                         member_fingerprints=(fp,)),
            member_fingerprints=(fp,),
            details=member_details(label, ensemble_registry.MEMBER_CLASS)))

    variants = sorted(path for path in root.glob(f"{GATE_DIR_PREFIX}*")
                      if path.is_dir() and path.name != PRODUCTION_ARTIFACT_DIR)
    for directory in variants:
        name = directory.name.removeprefix(GATE_DIR_PREFIX)
        if not _VARIANT.fullmatch(name):
            continue
        specs.append(_combiner_spec(
            spec=f"{GATE_PREFIX}{name}", kind="gate", label=f"Gate {name} · variant",
            directory=directory, fitted_for_current=False, active=active, fps=fps))

    # Standalone universal models — after the gates, so the first three specs
    # stay production, mean, rbf. The registry keeps the classes apart; the
    # filter only guards a hand-edited file.
    universal = [label for label in universal_labels() if label not in set(active)]
    universal_fps = member_fingerprints(universal)
    for label in universal:
        fp = universal_fps.get(label)
        reason = _universal_reason(label, fp)
        spec = f"{UNIVERSAL_PREFIX}{member_name(label)}"
        specs.append(ModelSpec(
            spec, "universal", spec_label(spec), (label,), (label,), reason is None,
            reason=reason,
            fingerprint=(spec_fingerprint("universal", member_labels=(label,),
                                          member_fingerprints=(fp,))
                         if reason is None else None),
            member_fingerprints=(fp,),
            details=member_details(label, ensemble_registry.UNIVERSAL_CLASS)))
    specs.extend(_milestone_specs())
    return specs


def _milestone_specs() -> list[ModelSpec]:
    """One spec per pulled milestone (:func:`milestones.local_steps`), by
    member then step; ``details`` = the member's recipe and the step."""
    out = []
    for member in milestones.local_members():
        origin = _origin(str(milestones.local_root() / member)) or {}
        for step in milestones.local_steps(member):
            label = milestone_label(member, step)
            fp = member_fingerprint(milestone_dir(label))
            spec = f"{MILESTONE_PREFIX}{member}@{step}"
            out.append(ModelSpec(
                spec, "milestone", spec_label(spec), (label,), (label,), fp is not None,
                reason=None if fp else "milestone has no checkpoint",
                fingerprint=spec_fingerprint("milestone", member_labels=(label,),
                                             member_fingerprints=(fp,)),
                member_fingerprints=(fp,),
                details={"milestone": True, "member": member, "step": step,
                         "target_steps": _int(origin.get("target_steps")),
                         "num_res_blocks": _int(origin.get("num_res_blocks"))}))
    return out


def resolve_spec(spec: str, specs: Sequence[ModelSpec] | None = None) -> ModelSpec:
    """The catalogue entry of one spec (:class:`KeyError` when unknown)."""
    wanted = canonical_spec(spec)
    for item in (list_specs() if specs is None else specs):
        if item.spec == wanted:
            return item
    raise KeyError(f"unknown model spec {wanted!r}")


def current_fingerprints(specs: Sequence[ModelSpec] | None = None) -> dict[str, str | None]:
    """``{spec: fingerprint}`` now (``None`` for unavailable specs)."""
    return {item.spec: item.fingerprint for item in (list_specs() if specs is None else specs)}


def needed_members(specs: Iterable[ModelSpec]) -> list[str]:
    """Union of the members the specs run, first-use order."""
    out: list[str] = []
    for item in specs:
        for label in item.reads:
            if label not in out:
                out.append(label)
    return out


def production_summary(specs: Sequence[ModelSpec] | None = None) -> dict[str, Any]:
    """What production is: ``{kind: "universal"|"gate", spec (the model it
    runs: "universal:member_238" | "production"), name|null, label,
    available, reason}``."""
    production = resolve_spec(SPEC_PRODUCTION, specs)
    name = production_model_name()
    if name is not None:
        model_spec = f"{UNIVERSAL_PREFIX}{member_name(member_label(name))}"
        return {"kind": "universal", "spec": model_spec, "name": name,
                "label": spec_label(model_spec), "available": production.available,
                "reason": production.reason}
    return {"kind": "gate", "spec": SPEC_PRODUCTION, "name": None,
            "label": COMBINER_MODELS[PRODUCTION_KIND].label,
            "available": production.available, "reason": production.reason}


def catalog_payload() -> dict[str, Any]:
    """``GET /api/models``."""
    specs = list_specs()
    active = active_member_labels()
    production = production_summary(specs)
    return {
        "regime": "starfull",
        # "universal" when production is a universal model, else the gate's kind
        "production_kind": "universal" if production["kind"] == "universal" else PRODUCTION_KIND,
        "production": production,
        "members": active,
        "models": [item.to_dict() for item in specs],
    }


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------

class MemberSource(Protocol):
    """Provides one member's SR ``(2H, 2W, 4)`` electrons for the current tile."""

    def get(self, label: str) -> np.ndarray: ...


def load_combiner_for(spec: ModelSpec):
    """The fitted combiner behind a combiner spec (``None`` if it vanished)."""
    if spec.combiner_dir is None or spec.combiner_kind is None:
        return None
    return _load_combiner(spec.combiner_dir, spec.member_labels, spec.combiner_kind)


def predict(spec: ModelSpec, lr_e: np.ndarray, members: MemberSource) -> np.ndarray:
    """SR ``(2H, 2W, 4)`` electrons of ``spec`` for one LR tile."""
    if not spec.available:
        raise ValueError(f"model {spec.spec} is unavailable: {spec.reason}")
    if is_single_model(spec):
        return np.asarray(members.get(spec.reads[0]), np.float32)
    if spec.kind == "mean":
        total = None
        for label in spec.reads:
            value = np.asarray(members.get(label), np.float64)
            total = value if total is None else total + value
        assert total is not None
        return (total / len(spec.reads)).astype(np.float32)
    combiner = load_combiner_for(spec)
    if combiner is None:
        raise ValueError(f"model {spec.spec} is unavailable: its artifact no longer loads")
    if isinstance(combiner, SpatialGateCombiner):
        labels = [spec.member_labels[i] for i in combiner.needed_member_indices()]
    else:
        labels = list(spec.member_labels)
    stack = np.stack([np.asarray(members.get(label), np.float32) for label in labels])
    return np.asarray(combiner.apply_field(stack, lr=np.asarray(lr_e, np.float32)),
                      np.float32)


def gate_weights(spec: ModelSpec, lr_e: np.ndarray, members: MemberSource) -> np.ndarray | None:
    """``(2H, 2W, M, 4)`` convex weights of a spatial-gate spec (else ``None``)."""
    combiner = load_combiner_for(spec) if spec.available else None
    if not isinstance(combiner, SpatialGateCombiner):
        return None
    labels = [spec.member_labels[i] for i in combiner.needed_member_indices()]
    stack = np.stack([np.asarray(members.get(label), np.float32) for label in labels])
    return combiner.weights_field(stack, lr=np.asarray(lr_e, np.float32))


class EnsembleMemberRunner:
    """Runs STARFULL members — and universal models — by label through one
    lazily loaded :class:`EnsembleModel` (only the requested models are
    evaluated).

    ``labels`` — the models a job will ask for — are exactly the models the
    ensemble restores (``EnsembleModel(labels=…)``, in that order; labels that
    are neither active STARFULL members nor runnable universal models
    (:func:`runnable_universal_labels`) are left out so :meth:`predict`
    reports them). A universal model among them adds
    ``include_universal=True`` (an ensemble of one each); an active-only job
    keeps the plain ensemble call. ``None`` loads every active member. A
    request for a runnable model outside ``labels`` (a job's plan changed,
    e.g. a reusable output turned out stale) rebuilds the ensemble once over
    the union; so does a planned model left out of the first build that can
    run now (a universal model mid-pull), without listing it twice.
    """

    def __init__(self, base_dir: str | None = None, factory=EnsembleModel, *,
                 labels: Iterable[str] | None = None) -> None:
        self._base_dir = base_dir
        self._factory = factory
        self._labels = None if labels is None else list(dict.fromkeys(labels))
        self._ensemble = None
        self._milestones: dict[str, Any] = {}

    @staticmethod
    def _runnable(labels: Sequence[str]) -> tuple[list[str], set[str]]:
        """``(runnable labels in order, each once; the universal ones among them)``."""
        labels = list(dict.fromkeys(labels))       # EnsembleModel refuses a label listed twice
        active = set(active_member_labels())
        universal = set(runnable_universal_labels(lb for lb in labels if lb not in active))
        return [lb for lb in labels if lb in active or lb in universal], universal

    @property
    def ensemble(self):
        if self._ensemble is None:
            kwargs: dict[str, Any] = {"starless": False}
            if self._labels is not None:
                labels, universal = self._runnable(self._labels)
                kwargs["labels"] = labels
                if universal:
                    kwargs["include_universal"] = True
            self._ensemble = self._factory(self._base_dir or ensemble_dir(), **kwargs)
        return self._ensemble

    def _milestone(self, label: str):
        """A milestone as an ensemble of one, from its own ensemble dir."""
        if label not in self._milestones:
            member, step = milestone_of(label)
            self._milestones[label] = self._factory(
                milestones.materialize(member, step), labels=[member],
                include_universal=True, starless=False)
        return self._milestones[label]

    def predict(self, lr_e: np.ndarray, label: str) -> np.ndarray:
        if is_milestone_label(label):
            stack = self._milestone(label).member_arrays(np.asarray(lr_e, np.float32))
            return np.asarray(stack[0], np.float32)
        labels = list(self.ensemble.member_labels)
        if (label not in labels and self._labels is not None
                and self._runnable([label])[0]):
            # A planned label left out of the first build (not runnable then,
            # e.g. mid-pull) is already in the plan: rebuild, never re-append.
            if label not in self._labels:
                self._labels = [*self._labels, label]
            self._ensemble = None
            labels = list(self.ensemble.member_labels)
        if label not in labels:
            raise KeyError(f"member {label} is not an active STARFULL member "
                           "or a runnable universal model")
        stack = self.ensemble.member_arrays(np.asarray(lr_e, np.float32),
                                            indices=[labels.index(label)])
        return np.asarray(stack[0], np.float32)


# ---------------------------------------------------------------------------
# the output store
# ---------------------------------------------------------------------------

def experiments_root() -> Path:
    return Path(Config.EUCLID_INFERENCE_DIR) / "experiments"


def outputs_root() -> Path:
    return experiments_root() / "outputs"


def member_cache_root() -> Path:
    """Per-tile member-SR cache of the experiments (bounded, LRU)."""
    return experiments_root() / "cache"


def check_id(value: str, what: str = "id") -> str:
    text = str(value or "")
    if not _SAFE_ID.fullmatch(text):
        raise ValueError(f"bad {what} {value!r}")
    return text


def output_dir(source: str, identifier: str) -> Path:
    return outputs_root() / check_id(source, "source") / check_id(identifier)


def member_cache_dir(source: str, identifier: str) -> Path:
    return member_cache_root() / check_id(source, "source") / check_id(identifier)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_files.write_json(path, payload, indent=2, sort_keys=True, allow_nan=False)


_STRUCTURAL_CARDS = frozenset({
    "SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "EXTEND", "BZERO",
    "BSCALE", "XTENSION", "PCOUNT", "GCOUNT", "EXTNAME", "EXTVER", "CHECKSUM",
    "DATASUM", "COMMENT", "HISTORY", "", "WCSAXES", "LONPOLE", "LATPOLE", "MJDREF",
})
_WCS_CARD = re.compile(r"^(CTYPE|CRVAL|CRPIX|CDELT|CUNIT|CROTA)\d$|^(CD|PC)\d_\d$")


def sr_header(lr_header: fits.Header | None, factor: int = SR_FACTOR) -> fits.Header:
    """Header of an SR grid: the LR celestial WCS magnified ``factor``×.

    ``CRPIX → factor·CRPIX − (factor − 1)/2`` and ``CD /= factor`` — the
    pixel-shuffle geometry of ``training.inference.scaled_wcs_header`` — always
    written in the compact 2-axis CD form, whatever mix of CD / PC+CDELT (or
    a cube's third axis) the LR header carries. Non-WCS cards are kept.
    """
    header = fits.Header()
    if lr_header is None:
        return header
    for key in list(lr_header.keys()):
        if key in _STRUCTURAL_CARDS or _WCS_CARD.match(key):
            continue
        try:
            header[key] = lr_header[key]
        except (ValueError, KeyError):
            continue
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wcs = WCS(lr_header).celestial
    except Exception:  # noqa: BLE001 - the SR simply carries no WCS then
        return header
    if not wcs.has_celestial:
        return header
    matrix = np.asarray(wcs.pixel_scale_matrix, np.float64) / float(factor)
    offset = (factor - 1) / 2.0
    header["CTYPE1"], header["CTYPE2"] = str(wcs.wcs.ctype[0]), str(wcs.wcs.ctype[1])
    header["CRVAL1"], header["CRVAL2"] = float(wcs.wcs.crval[0]), float(wcs.wcs.crval[1])
    header["CRPIX1"] = float(wcs.wcs.crpix[0]) * factor - offset
    header["CRPIX2"] = float(wcs.wcs.crpix[1]) * factor - offset
    header["CD1_1"], header["CD1_2"] = float(matrix[0, 0]), float(matrix[0, 1])
    header["CD2_1"], header["CD2_2"] = float(matrix[1, 0]), float(matrix[1, 1])
    return header


def save_output(source: str, identifier: str, spec: ModelSpec, sr: np.ndarray, *,
                lr_header: fits.Header | None, lr_sha: str | None,
                extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Persist one (tile, spec) SR + its sidecar; returns the sidecar."""
    directory = output_dir(source, identifier)
    directory.mkdir(parents=True, exist_ok=True)
    cube = np.asarray(sr, np.float32)
    header = sr_header(lr_header)
    header["BUNIT"] = ("electron", "SR electrons per SR pixel")
    header["SPEC"] = (spec.spec[:68], "model spec")
    header["SPECFP"] = (str(spec.fingerprint or "")[:68], "model spec fingerprint")
    header["BANDS"] = ",".join(Config.LR_INPUT_BAND_NAMES[:cube.shape[-1]])
    path = directory / f"{spec.slug}.fits"
    with atomic_files.temporary_sibling(path, ".fits") as temporary:
        fits.PrimaryHDU(np.moveaxis(cube, -1, 0), header=header).writeto(
            temporary, overwrite=True, output_verify="silentfix")
        os.replace(temporary, path)
    meta = {
        "spec": spec.spec, "slug": spec.slug, "kind": spec.kind, "label": spec.label,
        "fingerprint": spec.fingerprint,
        "member_labels": list(spec.member_labels),
        "member_fingerprints": list(spec.member_fingerprints),
        "combiner_kind": spec.combiner_kind,
        "combiner_fingerprint": spec.combiner_fingerprint,
        "lr_sha": lr_sha, "shape": [int(v) for v in cube.shape],
        "file": path.name, "created": datetime.now(UTC).isoformat(),
        "source": source, "id": identifier,
        **dict(extra or {}),
    }
    _atomic_json(directory / f"{spec.slug}.json", meta)
    return meta


def update_output_meta(source: str, identifier: str, spec: str,
                       patch: Mapping[str, Any]) -> dict[str, Any]:
    path = output_dir(source, identifier) / f"{spec_slug(spec)}.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.update(dict(patch))
    _atomic_json(path, meta)
    return meta


def list_outputs(source: str, identifier: str) -> dict[str, dict[str, Any]]:
    """``{spec: sidecar}`` of every cached output of one tile."""
    try:
        directory = output_dir(source, identifier)
    except ValueError:
        return {}
    out: dict[str, dict[str, Any]] = {}
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.json")):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (isinstance(meta, dict) and meta.get("spec")
                and (directory / str(meta.get("file", ""))).is_file()):
            out[str(meta["spec"])] = meta
    return out


def load_output(source: str, identifier: str, spec: str
                ) -> tuple[np.ndarray, fits.Header, dict[str, Any]]:
    """``(cube (2H, 2W, 4), header, sidecar)``; :class:`FileNotFoundError`."""
    directory = output_dir(source, identifier)
    slug = spec_slug(spec)
    meta_path, path = directory / f"{slug}.json", directory / f"{slug}.fits"
    if not (meta_path.is_file() and path.is_file()):
        raise FileNotFoundError(f"no {spec} output for {source}/{identifier}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with fits.open(path, memmap=False) as hdul:
        primary = hdul[0]
        data = np.asarray(primary.data, np.float32)
        header = primary.header.copy()
    return np.moveaxis(data, 0, -1), header, meta


def output_state(meta: Mapping[str, Any], current: Mapping[str, str | None]) -> str:
    """``current`` | ``stale`` | ``unavailable`` (the spec cannot run now)."""
    now = current.get(str(meta.get("spec")))
    if now is None:
        return "unavailable"
    return "current" if meta.get("fingerprint") == now else "stale"


def delete_outputs(source: str, identifier: str,
                   specs: Iterable[str] | None = None) -> list[str]:
    """Remove cached outputs (all, or ``specs``); returns removed paths."""
    directory = output_dir(source, identifier)
    if not directory.is_dir():
        return []
    # Exact file names: variant names may hold dots (``gate:v1`` vs ``gate:v1.2``).
    wanted = None if specs is None else {
        f"{spec_slug(spec)}{suffix}" for spec in specs for suffix in (".fits", ".json")}
    removed: list[str] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix not in (".fits", ".json") or path.name.startswith("."):
            continue
        if wanted is not None and path.name not in wanted:
            continue
        path.unlink()
        removed.append(str(path))
    if not any(directory.iterdir()):
        directory.rmdir()
    return removed


__all__ = [
    "EnsembleMemberRunner",
    "MemberSource",
    "ModelSpec",
    "PRODUCTION_KIND",
    "SPEC_MEAN",
    "SPEC_PRODUCTION",
    "SPEC_RBF",
    "UNIVERSAL_PREFIX",
    "active_member_labels",
    "array_sha",
    "canonical_spec",
    "catalog_payload",
    "current_fingerprints",
    "delete_outputs",
    "gate_weights",
    "list_outputs",
    "list_specs",
    "load_output",
    "member_fingerprints",
    "member_cache_dir",
    "member_cache_root",
    "member_details",
    "member_label",
    "member_name",
    "needed_members",
    "output_dir",
    "output_state",
    "parse_specs",
    "predict",
    "regime_dir",
    "resolve_spec",
    "runnable_universal_labels",
    "save_output",
    "spec_fingerprint",
    "spec_kind",
    "spec_label",
    "spec_member_label",
    "milestone_label",
    "is_milestone_label",
    "spec_slug",
    "sr_header",
    "training_log_summary",
    "universal_labels",
    "update_output_meta",
]
