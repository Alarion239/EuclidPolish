"""Runnable STARFULL model specs and their cached per-tile outputs (C9).

A **model spec** names one way of turning a real LR tile into an SR image:

==================  ==========================================================
``production``      the production combiner — the spatial gate
                    (``ACTIVE_COMBINER_KINDS[0]``) fitted for the current
                    STARFULL membership; unavailable otherwise (no fallback)
``mean``            plain mean of every active STARFULL member
``member:<name>``   one active member, e.g. ``member:member_170``
``gate:<variant>``  a named spatial-gate variant (``spatial_gate_<variant>/``
                    beside the production artifact), applied with its OWN
                    member labels; unavailable (with the reason) unless every
                    one of them is an active STARFULL member
``rbf``             the legacy RBF combiner, with its own member labels
==================  ==========================================================

Members are always run through :class:`~euclid_polish.ensemble.EnsembleModel`
(ensemble-only rule: a single model is an ensemble of one) and addressed by
their ``NN·psnr`` labels (:class:`EnsembleMemberRunner`).

Every spec has a **fingerprint**: a hash of the checkpoint identities of the
members it reads (:func:`euclid_polish.ensemble.member_fingerprint`) and, for
combiners, of the fitted artifact (``combiner.json`` + ``combiner.npz``). A
cached output is *current* while its recorded fingerprint equals the spec's
fingerprint now.

The **output store** keeps one SR per (real tile, spec) under
``<EUCLID_INFERENCE_DIR>/experiments/outputs/<source>/<id>/<slug>.fits`` —
``(4, 2H, 2W)`` electrons with the LR WCS magnified ×2 — plus a
``<slug>.json`` sidecar (spec, fingerprint, members, LR hash, metrics).
"""

from __future__ import annotations

import hashlib
import json
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

SPEC_PRODUCTION = "production"
SPEC_MEAN = "mean"
SPEC_RBF = "rbf"
MEMBER_PREFIX = "member:"
GATE_PREFIX = "gate:"
GATE_DIR_PREFIX = "spatial_gate_"
PRODUCTION_KIND = ACTIVE_COMBINER_KINDS[0]
PRODUCTION_ARTIFACT_DIR = COMBINER_MODELS[PRODUCTION_KIND].artifact_dir
RBF_KIND = RAW_INCREMENTAL_MINMEANMAX_RBF_KIND
#: SR grid magnification of every STARFULL member.
SR_FACTOR = int(Config.DEFAULT_REBIN_FACTOR)

_MEMBER_DIGITS = re.compile(r"^\d{1,6}$")
_VARIANT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")


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


def member_fingerprints(labels: Iterable[str]) -> dict[str, str | None]:
    """Checkpoint identity of each member (``None`` without a checkpoint)."""
    base = ensemble_dir()
    return {label: member_fingerprint(os.path.join(base, member_name(label)))
            for label in labels}


# ---------------------------------------------------------------------------
# spec names
# ---------------------------------------------------------------------------

def canonical_spec(raw: str) -> str:
    """Normalise a spec string (aliases: ``member:170``, ``member:170·psnr``,
    ``gate:spatial_gate_26m``); :class:`ValueError` when malformed."""
    text = str(raw or "").strip()
    lower = text.lower()
    if lower in (SPEC_PRODUCTION, SPEC_MEAN, SPEC_RBF):
        return lower
    if lower.startswith(MEMBER_PREFIX):
        name = text[len(MEMBER_PREFIX):].strip().removeprefix("member_").split("·")[0]
        if not _MEMBER_DIGITS.fullmatch(name):
            raise ValueError(f"bad member spec {raw!r} (use member:member_<N>)")
        return f"{MEMBER_PREFIX}member_{name}"
    if lower.startswith(GATE_PREFIX):
        name = text[len(GATE_PREFIX):].strip().removeprefix(GATE_DIR_PREFIX)
        if name == PRODUCTION_ARTIFACT_DIR.removeprefix(GATE_DIR_PREFIX):
            return SPEC_PRODUCTION
        if not _VARIANT.fullmatch(name):
            raise ValueError(f"bad gate spec {raw!r} (use gate:<variant>)")
        return f"{GATE_PREFIX}{name}"
    raise ValueError(
        f"unknown model spec {raw!r}: use production, mean, rbf, "
        "member:<name> or gate:<variant>")


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
    """Filesystem-safe name of a canonical spec (``gate:26m`` → ``gate-26m``)."""
    return canonical_spec(spec).replace(":", "-")


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
    if kind == "member":
        return _hash("member", labels[0], fps[0]) if len(labels) == 1 else None
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
    kind: str                       # production | mean | member | gate | rbf
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
            "n_members": len(self.member_labels),
            "available": self.available, "reason": self.reason,
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
                "installed_from", "fit_seconds", "complete"):
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
    details = _fit_summary(manifest, directory)
    if fitted_for_current and list(labels) != list(active):
        diff = sorted(set(active) ^ set(labels))
        reason = (f"the production gate was fitted for {len(labels)} members; the "
                  f"active STARFULL ensemble has {len(active)} (differs in "
                  f"{', '.join(diff[:6])}{'…' if len(diff) > 6 else ''})")
        return ModelSpec(spec, kind, label, labels, reads, False, reason=reason,
                         combiner_kind=combiner_kind, combiner_dir=str(directory),
                         details=details)
    member_fps = {**fps, **member_fingerprints([lb for lb in labels if lb not in fps])}
    reason = _membership_reason(labels, active, member_fps)
    combiner = _load_combiner(str(directory), labels, combiner_kind)
    if reason is None and combiner is None:
        reason = f"{directory.name} is not a loadable {combiner_kind} artifact"
    artifact_fp = combiner_artifact_fingerprint(str(directory.parent), directory.name)
    fingerprint = None
    if reason is None and artifact_fp:
        fingerprint = spec_fingerprint(
            kind, combiner_kind=combiner_kind, combiner_fingerprint=artifact_fp,
            member_labels=labels, member_fingerprints=[member_fps.get(lb) for lb in labels])
    elif reason is None:
        reason = f"{directory.name} artifact is incomplete"
    return ModelSpec(
        spec, kind, label, labels, reads, reason is None, reason=reason,
        fingerprint=fingerprint,
        member_fingerprints=tuple(member_fps.get(lb) for lb in labels),
        combiner_kind=combiner_kind, combiner_dir=str(directory),
        combiner_fingerprint=artifact_fp, details=details)


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


def list_specs() -> list[ModelSpec]:
    """Every model spec: production, mean, rbf, each active member, each
    gate variant (unavailable ones carry a ``reason``)."""
    active = active_member_labels()
    fps = member_fingerprints(active)
    root = regime_dir()
    specs: list[ModelSpec] = []

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
        specs.append(ModelSpec(
            f"{MEMBER_PREFIX}{member_name(label)}", "member", f"Member {label}",
            (label,), (label,), bool(fp),
            reason=None if fp else "member has no checkpoint",
            fingerprint=spec_fingerprint("member", member_labels=(label,),
                                         member_fingerprints=(fp,)),
            member_fingerprints=(fp,)))

    variants = sorted(path for path in root.glob(f"{GATE_DIR_PREFIX}*")
                      if path.is_dir() and path.name != PRODUCTION_ARTIFACT_DIR)
    for directory in variants:
        name = directory.name.removeprefix(GATE_DIR_PREFIX)
        if not _VARIANT.fullmatch(name):
            continue
        specs.append(_combiner_spec(
            spec=f"{GATE_PREFIX}{name}", kind="gate", label=f"Gate variant · {name}",
            directory=directory, fitted_for_current=False, active=active, fps=fps))
    return specs


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


def catalog_payload() -> dict[str, Any]:
    """``GET /api/models``."""
    specs = list_specs()
    active = active_member_labels()
    return {
        "regime": "starfull",
        "production_kind": PRODUCTION_KIND,
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
    if spec.kind == "member":
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
    """Runs STARFULL members by label through one lazily loaded
    :class:`EnsembleModel` (only the requested members are evaluated).

    ``labels`` — the members a job will ask for — caps the ensemble at the
    shortest prefix of the registry-active STARFULL order that holds them
    (``EnsembleModel(n_members=…)``), so a job over a few early members no
    longer restores every active checkpoint. ``EnsembleModel`` has no
    arbitrary-subset option, so a late member still restores the members
    before it; ``None`` loads every active member, and a request for a member
    past the prefix reloads the ensemble uncapped once.
    """

    def __init__(self, base_dir: str | None = None, factory=EnsembleModel, *,
                 labels: Iterable[str] | None = None) -> None:
        self._base_dir = base_dir
        self._factory = factory
        self._labels = None if labels is None else list(labels)
        self._ensemble = None

    def _prefix_length(self) -> int | None:
        if not self._labels:
            return None
        order = active_member_labels()
        positions = [order.index(label) for label in self._labels if label in order]
        if len(positions) != len(self._labels):
            return None                  # an unknown label: predict() reports it
        return max(positions) + 1

    @property
    def ensemble(self):
        if self._ensemble is None:
            kwargs: dict[str, Any] = {"starless": False}
            prefix = self._prefix_length()
            if prefix is not None:
                kwargs["n_members"] = prefix
            self._ensemble = self._factory(self._base_dir or ensemble_dir(), **kwargs)
        return self._ensemble

    def predict(self, lr_e: np.ndarray, label: str) -> np.ndarray:
        labels = list(self.ensemble.member_labels)
        if label not in labels and self._labels is not None:
            # Asked for a member past the prefix (a job's plan changed, e.g.
            # a reusable output turned out stale): load every member once.
            self._labels, self._ensemble = None, None
            labels = list(self.ensemble.member_labels)
        if label not in labels:
            raise KeyError(f"member {label} is not an active STARFULL member")
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
                         + "\n", encoding="utf-8")
    os.replace(temporary, path)


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
    temporary = directory / f".{spec.slug}.{os.getpid()}.tmp.fits"
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
    "member_label",
    "member_name",
    "needed_members",
    "output_dir",
    "output_state",
    "parse_specs",
    "predict",
    "regime_dir",
    "resolve_spec",
    "save_output",
    "spec_slug",
    "spec_fingerprint",
    "sr_header",
    "update_output_meta",
]
