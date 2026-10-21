"""Source of truth for which ensemble members are active.

The registry file lives at ``<ensemble parent>/ensemble_registry.json`` — one
level ABOVE the ensemble dir on purpose: the FASRC checkpoint mirror
(:mod:`euclid_polish.web.fasrc_mirror`, manual) rsyncs the ensemble dir with
``--delete-after``, which would delete any local-only file inside it.

It holds three keys (an older file without ``universal`` reads it as ``[]``)
and an optional production pointer::

    {"active": [...], "archived": [tombstone, ...], "universal": [...],
     "production": {"name": "member_238", "set_at": "..."}}

Bootstrap rule: any ``member_*`` directory with a checkpoint that the registry
has never seen (not active, archived or universal) is auto-added to ``active``
— so members trained on FASRC and pulled/mirrored down just work. Archived
tombstones are permanent: a member dir reappearing on disk (e.g. mirrored back
from FASRC after a local archive) is NOT re-activated.

``universal`` lists standalone models: single-image multi-knee members
(:func:`is_universal_origin`) pulled to compare on their own. They live in the
same ``member_NNN`` dirs and namespace as members but are NOT ensemble members
— not in the mean, the gates, Evaluate or any cache keyed on the active
labels (:func:`active_member_dirs`, :func:`active_labels` and
:func:`regime_labels` stay ensemble-only). A pull registers the name BEFORE its
rsync, so the bootstrap never activates the arriving dir, and an entry is
never dropped for a missing dir (mid-pull, or deleted by the mirror); a pull
whose rsync fails removes the registration it added
(:func:`remove_universal_entry`).

A tombstone records where the member went and which class it left::

    {"name": "member_02", "archived_at": "...", "zip": "models/…zip",
     "commit": "abc1234", "class": "member"}

(``"class": "universal"`` for a universal model; tombstones written before the
key existed are members.) Every read-modify-write of the file holds one
module-level lock: jobs are threads in the console process, and page reads
bootstrap-write too.

``production`` names the registered universal model that IS the production
model — the model catalogue's ``production`` spec and the evaluators' model
(:func:`production_name`, :func:`set_production`). Absent, production is the
legacy production gate (``<vis>/ensemble/starfull/spatial_gate_combiner``).
The production model cannot be archived or unregistered while it is
production.
"""

from __future__ import annotations

import glob
import json
import os
import re
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from euclid_polish.config import Config
from euclid_polish.tracking._utils import _read_json, _write_json

_MEMBER_GLOB = "member_*"
REGISTRY_FILENAME = "ensemble_registry.json"
_MEMBER_SPELLING = re.compile(r"^(?:member_)?(\d{1,6})(?:·psnr)?$")

#: Tombstone ``class`` values: an ensemble member or a universal model.
MEMBER_CLASS = "member"
UNIVERSAL_CLASS = "universal"
#: The registry key of the production pointer (see the module docstring).
PRODUCTION_KEY = "production"

#: Guards every read-modify-write of the registry file (see the module doc).
_LOCK = threading.Lock()


def member_name(label: str) -> str:
    """``"196·psnr"`` / ``"196"`` / ``"member_196"`` → ``"member_196"``.

    The one normaliser for every spelling the UI, the registry, the cube
    manifests (``NN·psnr``) and the model catalogue use; anything else (a
    ``·loss`` track, a path) raises :class:`ValueError`. The number is
    zero-padded to two digits like every member directory (``"2"`` →
    ``"member_02"``)."""
    m = _MEMBER_SPELLING.fullmatch(str(label).strip())
    if m is None:
        raise ValueError(f"not a member name or label: {label!r}")
    return f"member_{int(m.group(1)):02d}"


def member_label(name: str) -> str:
    """Any member spelling → its ensemble label ``"NN·psnr"``."""
    return member_name(name).removeprefix("member_") + "·psnr"


def member_is_starless(member_dir: str) -> bool:
    """Whether a member trained in the STARLESS regime (erase stars), read from
    its ``origin.json``. Members predating the star knob have no field → they
    are STARFULL (the original reconstruct-stars behavior). Lives here (no
    TensorFlow) so :func:`regime_labels` needs no import of
    :mod:`euclid_polish.ensemble`, which re-exports it."""
    try:
        with open(os.path.join(member_dir, "origin.json")) as f:
            return bool(json.load(f).get("starless", False))
    except (OSError, ValueError, AttributeError):
        return False


def is_universal_origin(origin: Mapping[str, Any] | None) -> bool:
    """Whether a member's recipe makes it a UNIVERSAL model: a multi-knee
    member (non-empty ``asinh_knees``) that outputs ONE image — at a fixed
    ``output_knee`` (members 195/196) or at a learned one
    (``learned_output_knee``, a dict and no ``output_knee`` key; 206+).
    Six-head multi-knee members (197/198: one image per knee) and single-knee
    members are not.

    Takes an ``origin.json`` dict; also accepts a FASRC job-ledger
    ``member_spec`` entry, whose learned-knee spelling is
    ``learn_output_knee: true`` (and an ``output_knee`` init value)."""
    if not isinstance(origin, Mapping):
        return False
    knees = origin.get("asinh_knees")
    if not (isinstance(knees, Sequence) and not isinstance(knees, str) and len(knees) > 0):
        return False
    return (origin.get("output_knee") is not None
            or bool(origin.get("learned_output_knee"))
            or bool(origin.get("learn_output_knee")))


def default_ensemble_dir() -> str:
    """THE model location: ``<ckpt parent>/ensemble`` (members in
    ``member_NN/``). Lives here (not in :mod:`euclid_polish.ensemble`) so
    cheap status paths can resolve it without importing TensorFlow;
    ``euclid_polish.ensemble`` re-exports it."""
    return os.path.join(
        os.path.dirname(Config.DEFAULT_CHECKPOINT_DIR.rstrip("/")) or ".",
        "ensemble")


def _checkpoint_exists(d: str) -> bool:
    # Mirrors sky_records.checkpoint_present — cheap, no TensorFlow import.
    return (os.path.isfile(os.path.join(d, "checkpoint"))
            or bool(glob.glob(os.path.join(d, "*.index"))))


def registry_path(base_dir: str) -> str:
    parent = os.path.dirname(os.path.abspath(base_dir).rstrip("/")) or "."
    return os.path.join(parent, REGISTRY_FILENAME)


def _member_dirs_on_disk(base_dir: str) -> list[str]:
    return sorted(
        d for d in glob.glob(os.path.join(base_dir, _MEMBER_GLOB))
        if os.path.isdir(d) and _checkpoint_exists(d))


def _tombstoned(reg: Mapping[str, Any]) -> set[str]:
    return {str(t.get("name")) for t in reg["archived"]}


def _production_entry(raw: Any) -> dict[str, Any] | None:
    """The stored production pointer when well formed (``{"name": ...}``)."""
    if not isinstance(raw, Mapping) or not _MEMBER_SPELLING.fullmatch(str(raw.get("name") or "")):
        return None
    return {"name": member_name(str(raw["name"])), "set_at": raw.get("set_at")}


def _load_locked(base_dir: str) -> dict[str, Any]:
    """:func:`load_registry` for a caller already holding :data:`_LOCK`."""
    reg = _read_json(registry_path(base_dir)) or {}
    active = [str(n) for n in reg.get("active", [])]
    archived = list(reg.get("archived", []))
    tombstoned = {str(t.get("name")) for t in archived}
    listed = [str(n) for n in reg.get("universal") or []]
    # One class per name: a tombstone is permanent and an ensemble member
    # outranks a universal registration (only a hand edit makes either clash).
    universal = sorted(set(listed) - set(active) - tombstoned)
    changed = universal != listed
    seen = set(active) | tombstoned | set(universal)
    on_disk = {os.path.basename(d) for d in _member_dirs_on_disk(base_dir)}
    for name in sorted(on_disk - seen):          # bootstrap new members
        active.append(name)
        changed = True
    kept = [n for n in active if n in on_disk]   # drop vanished actives
    if kept != active:
        active, changed = kept, True
    # Universal entries are never dropped for a missing dir: the pull
    # registers before its rsync, and the mirror may delete the dir.
    out: dict[str, Any] = {"active": sorted(active), "archived": archived, "universal": universal}
    production = _production_entry(reg.get(PRODUCTION_KEY))
    if production is not None:
        out[PRODUCTION_KEY] = production         # carried through every write
    if changed and (on_disk or archived or universal):
        _write_json(registry_path(base_dir), out)
    return out


def load_registry(base_dir: str) -> dict[str, Any]:
    """Load + bootstrap the registry; persists any change it makes. Always
    returns ``{"active", "archived", "universal"}`` (plus ``production`` when
    set); a write always carries all of them."""
    with _LOCK:
        return _load_locked(base_dir)


def production_name(base_dir: str) -> str | None:
    """The universal model named production (``"member_238"``), or ``None``
    when production is the legacy production gate. The name is returned even
    when it is no longer a loadable universal model, so callers report why
    production is unavailable instead of silently using something else."""
    entry = load_registry(base_dir).get(PRODUCTION_KEY)
    return str(entry["name"]) if entry else None


def set_production(base_dir: str, name: str | None) -> str | None:
    """Make the registered universal model ``name`` (any :func:`member_name`
    spelling) production, or clear the pointer (``None``: production is the
    production gate again). Returns the stored name. Raises
    :class:`ValueError` for a name that is not a registered universal model
    with a checkpoint on disk."""
    with _LOCK:
        reg = _load_locked(base_dir)
        if name is None:
            reg.pop(PRODUCTION_KEY, None)
        else:
            name = member_name(name)
            if name not in reg["universal"]:
                raise ValueError(f"{name} is not a registered universal model")
            if not _checkpoint_exists(os.path.join(base_dir, name)):
                raise ValueError(f"{name} has no checkpoint on disk")
            reg[PRODUCTION_KEY] = {"name": name,
                                   "set_at": datetime.now(UTC).isoformat(timespec="seconds")}
        _write_json(registry_path(base_dir), reg)
    return name


def active_member_dirs(base_dir: str) -> list[str]:
    return [os.path.join(base_dir, n)
            for n in load_registry(base_dir)["active"]]


def active_labels(base_dir: str) -> list[str]:
    """Model labels of every registry-active member (both star regimes),
    aligned with member order: one ``NN·psnr`` per active member. Caches
    validate against the per-regime :func:`regime_labels`, not this list.
    Evaluation uses only each member's
    PSNR-best checkpoint — the ``loss_best/`` sub-track is still saved during
    training (and can seed forks) but is NOT part of the ensemble."""
    return [os.path.basename(d).removeprefix("member_") + "·psnr"
            for d in active_member_dirs(base_dir)]


def regime_labels(base_dir: str, starless: bool) -> list[str]:
    """Active-member labels of ONE star regime, matching exactly what
    :class:`~euclid_polish.ensemble.EnsembleModel` loads for that regime:
    registry-active ∩ has-checkpoint ∩ matching star regime, in member order.

    The starfull and starless reconstructions are fully detached — each keeps
    its own cube/eval/combiner artifacts — so this (NOT :func:`active_labels`,
    which spans both regimes) is the per-regime membership fingerprint those
    caches validate against."""
    out = []
    for d in active_member_dirs(base_dir):
        if (os.path.isdir(d) and _checkpoint_exists(d)
                and member_is_starless(d) == bool(starless)):
            out.append(os.path.basename(d).removeprefix("member_") + "·psnr")
    return out


def universal_names(base_dir: str) -> list[str]:
    """Every registered universal model (sorted), whether or not its dir is
    on disk yet."""
    return list(load_registry(base_dir)["universal"])


def universal_member_dirs(base_dir: str) -> list[str]:
    """The loadable universal models: registered ∩ dir exists ∩ has a
    checkpoint, in name order. Never part of :func:`active_member_dirs`."""
    out = []
    for n in universal_names(base_dir):
        d = os.path.join(base_dir, n)
        if os.path.isdir(d) and _checkpoint_exists(d):
            out.append(d)
    return out


def universal_labels(base_dir: str) -> list[str]:
    """``NNN·psnr`` labels of :func:`universal_member_dirs` (PSNR-best
    checkpoint, like the members)."""
    return [os.path.basename(d).removeprefix("member_") + "·psnr"
            for d in universal_member_dirs(base_dir)]


def add_universal_entry(base_dir: str, name: str) -> str:
    """Register ``name`` (any :func:`member_name` spelling) as a universal
    model and return its normalised name. Idempotent (no write when already
    registered). Call it BEFORE the pull's rsync so the arriving dir is never
    bootstrapped into the ensemble. Raises :class:`ValueError` for a
    malformed name, an active ensemble member or a tombstoned name (archived
    names are never reused)."""
    name = member_name(name)
    with _LOCK:
        reg = _load_locked(base_dir)
        if name in reg["active"]:
            raise ValueError(f"{name} is an active ensemble member")
        if name in _tombstoned(reg):
            raise ValueError(f"{name} is archived (tombstoned)")
        if name not in reg["universal"]:
            reg["universal"] = sorted([*reg["universal"], name])
            _write_json(registry_path(base_dir), reg)
    return name


def remove_universal_entry(base_dir: str, name: str) -> str:
    """Unregister a universal model (any :func:`member_name` spelling) and
    return its normalised name — the rollback of a pull whose rsync failed
    after :func:`add_universal_entry`. Raises :class:`ValueError` for a
    malformed name or one that is not a registered universal model (an
    active member, a tombstone or an unknown name is never touched). The
    caller makes sure no checkpoint of it is left on disk first, or the
    bootstrap would activate that dir."""
    name = member_name(name)
    with _LOCK:
        reg = _load_locked(base_dir)
        if name not in reg["universal"]:
            raise ValueError(f"{name} is not a registered universal model")
        _refuse_production(reg, name)
        reg["universal"] = [n for n in reg["universal"] if n != name]
        _write_json(registry_path(base_dir), reg)
    return name


def _refuse_production(reg: Mapping[str, Any], name: str) -> None:
    if (reg.get(PRODUCTION_KEY) or {}).get("name") == name:
        raise ValueError(f"{name} is the production model — make another model "
                         "production first")


def tombstone_class(tombstone: Mapping[str, Any]) -> str:
    """The class a tombstone left: :data:`UNIVERSAL_CLASS` or
    :data:`MEMBER_CLASS` (tombstones predating the key are members)."""
    return UNIVERSAL_CLASS if tombstone.get("class") == UNIVERSAL_CLASS else MEMBER_CLASS


def _member_index(name: str) -> int | None:
    tail = str(name).removeprefix("member_")
    return int(tail) if tail.isdigit() else None


def next_member_names(base_dir: str, k: int) -> list[str]:
    """``k`` fresh consecutive member names that can never collide.

    Starts at max(index over active ∪ archived tombstones ∪ universal ∪
    ``member_*`` on disk) + 1 — tombstoned indices are skipped FOREVER (the
    bootstrap ignores a reincarnated archived name, so reuse would create a
    ghost member), and a registered universal model owns its index before
    its dir is pulled. Called locally at FASRC submit time; the job receives
    explicit names.
    """
    reg = load_registry(base_dir)
    names = set(reg["active"]) | _tombstoned(reg) | set(reg["universal"])
    names |= {os.path.basename(d)
              for d in glob.glob(os.path.join(base_dir, _MEMBER_GLOB))}
    used = [i for i in (_member_index(n) for n in names) if i is not None]
    start = (max(used) + 1) if used else 0
    return [f"member_{i:02d}" for i in range(start, start + int(k))]


def archive_member_entry(base_dir: str, name: str, *, zip_path: str,
                         commit: str | None) -> dict[str, Any]:
    """Move ``name`` — an active member or a universal model — to an archived
    tombstone recording its ``class``. Returns the registry."""
    with _LOCK:
        reg = _load_locked(base_dir)
        _refuse_production(reg, name)
        if name in reg["active"]:
            key, cls = "active", MEMBER_CLASS
        elif name in reg["universal"]:
            key, cls = "universal", UNIVERSAL_CLASS
        else:
            raise ValueError(f"{name!r} is not an active ensemble member or a universal model")
        reg[key] = [n for n in reg[key] if n != name]
        reg["archived"].append({
            "name": name,
            "archived_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "zip": zip_path,
            "commit": commit,
            "class": cls,
        })
        _write_json(registry_path(base_dir), reg)
    return reg


def restore_member_entry(base_dir: str, name: str) -> dict[str, Any]:
    """Move an archived tombstone back into the class it left — ``active`` for
    a member, ``universal`` for a universal model (the inverse of
    :func:`archive_member_entry`). The member directory must already be back
    on disk with a checkpoint, else the bootstrap would drop it again."""
    name = member_name(name)
    with _LOCK:
        reg = _load_locked(base_dir)
        if name in reg["active"]:
            raise ValueError(f"{name} is already active")
        if name in reg["universal"]:
            raise ValueError(f"{name} is already a universal model")
        tomb = next((t for t in reg["archived"] if str(t.get("name")) == name), None)
        if tomb is None:
            raise ValueError(f"{name} is not archived")
        d = os.path.join(base_dir, name)
        if not (os.path.isdir(d) and _checkpoint_exists(d)):
            raise ValueError(f"{name} has no checkpoint on disk at {d}")
        reg["archived"] = [t for t in reg["archived"] if str(t.get("name")) != name]
        key = "universal" if tombstone_class(tomb) == UNIVERSAL_CLASS else "active"
        reg[key] = sorted([*reg[key], name])
        _write_json(registry_path(base_dir), reg)
    return reg


def _campaign_dirs(tracking_root: str) -> list[tuple[str, str]]:
    """``(campaign, dir)`` for the active campaign then every archived one."""
    root = os.path.abspath(tracking_root)
    out = [("current", os.path.join(root, "current"))]
    archive = os.path.join(root, "archive")
    if os.path.isdir(archive):
        out += [(n, os.path.join(archive, n)) for n in sorted(os.listdir(archive))
                if os.path.isdir(os.path.join(archive, n))]
    return out


def find_archived_zip(tombstone: dict[str, Any], tracking_root: str
                      ) -> tuple[str, str] | None:
    """``(campaign, abs zip path)`` of a tombstone's archive zip, or ``None``.

    The tombstone records the zip relative to the campaign that was active
    when the member was archived (``models/<file>.zip``), not which campaign;
    campaigns are saved into ``archive/<slug>`` later, so every campaign's
    ``models/`` is searched (the active one first). Only the file NAME is
    used, so a crafted tombstone can never point outside the tracking tree."""
    rel = str(tombstone.get("zip") or "").replace("\\", "/")
    fname = os.path.basename(rel)
    if not fname.endswith(".zip") or rel not in (fname, f"models/{fname}"):
        return None
    for campaign, d in _campaign_dirs(tracking_root):
        path = os.path.join(d, "models", fname)
        if os.path.isfile(path):
            return campaign, path
    return None


def archived_members(base_dir: str, tracking_root: str | None = None
                     ) -> list[dict[str, Any]]:
    """The tombstones, newest first, each with where its zip is now:
    ``zip_found``, ``zip_path`` (absolute or ``None``), ``campaign`` and
    ``size_bytes`` — the Models › Members archived table and the restore
    action read this."""
    root = tracking_root or Config.TRACKING_DIR
    rows = []
    for t in load_registry(base_dir)["archived"]:
        found = find_archived_zip(t, root)
        size = None
        if found is not None:
            try:
                size = os.path.getsize(found[1])
            except OSError:
                found = None
        rows.append({**t, "zip_found": found is not None,
                     "zip_path": found[1] if found else None,
                     "campaign": found[0] if found else None,
                     "size_bytes": size})
    rows.sort(key=lambda r: str(r.get("archived_at") or ""), reverse=True)
    return rows
