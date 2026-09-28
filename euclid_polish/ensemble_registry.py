"""Source of truth for which ensemble members are active.

The registry file lives at ``<ensemble parent>/ensemble_registry.json`` — one
level ABOVE the ensemble dir on purpose: the FASRC checkpoint auto-mirror
rsyncs the ensemble dir with ``--delete-after``, which would delete any
local-only file inside it.

Bootstrap rule: any ``member_*`` directory with a checkpoint that the registry
has never seen (neither active nor archived) is auto-added to ``active`` — so
members trained on FASRC and pulled/mirrored down just work. Archived
tombstones are permanent: a member dir reappearing on disk (e.g. mirrored back
from FASRC after a local archive) is NOT re-activated.

A tombstone records where the member went::

    {"name": "member_02", "archived_at": "...", "zip": "models/…zip",
     "commit": "abc1234"}
"""

from __future__ import annotations

import glob
import json
import os
import re
from datetime import UTC, datetime
from typing import Any

from euclid_polish.config import Config
from euclid_polish.tracking._utils import _read_json, _write_json

_MEMBER_GLOB = "member_*"
REGISTRY_FILENAME = "ensemble_registry.json"
_MEMBER_SPELLING = re.compile(r"^(?:member_)?(\d{1,6})(?:·psnr)?$")


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


def load_registry(base_dir: str) -> dict[str, Any]:
    """Load + bootstrap the registry; persists any change it makes."""
    reg = _read_json(registry_path(base_dir)) or {}
    active = [str(n) for n in reg.get("active", [])]
    archived = list(reg.get("archived", []))
    seen = set(active) | {str(t.get("name")) for t in archived}
    on_disk = {os.path.basename(d) for d in _member_dirs_on_disk(base_dir)}
    changed = False
    for name in sorted(on_disk - seen):          # bootstrap new members
        active.append(name)
        changed = True
    kept = [n for n in active if n in on_disk]   # drop vanished actives
    if kept != active:
        active, changed = kept, True
    out = {"active": sorted(active), "archived": archived}
    if changed and (on_disk or archived):
        _write_json(registry_path(base_dir), out)
    return out


def active_member_dirs(base_dir: str) -> list[str]:
    return [os.path.join(base_dir, n)
            for n in load_registry(base_dir)["active"]]


def active_labels(base_dir: str) -> list[str]:
    """Model labels the ensemble will load, aligned with member order:
    one ``NN·psnr`` per active member. This is the membership fingerprint
    caches are validated against. Evaluation uses only each member's
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


def _member_index(name: str) -> int | None:
    tail = str(name).removeprefix("member_")
    return int(tail) if tail.isdigit() else None


def next_member_names(base_dir: str, k: int) -> list[str]:
    """``k`` fresh consecutive member names that can never collide.

    Starts at max(index over active ∪ archived tombstones ∪ ``member_*`` on
    disk) + 1 — tombstoned indices are skipped FOREVER (the bootstrap ignores
    a reincarnated archived name, so reuse would create a ghost member).
    Called locally at FASRC submit time; the job receives explicit names.
    """
    reg = load_registry(base_dir)
    names = set(reg["active"]) | {str(t.get("name")) for t in reg["archived"]}
    names |= {os.path.basename(d)
              for d in glob.glob(os.path.join(base_dir, _MEMBER_GLOB))}
    used = [i for i in (_member_index(n) for n in names) if i is not None]
    start = (max(used) + 1) if used else 0
    return [f"member_{i:02d}" for i in range(start, start + int(k))]


def archive_member_entry(base_dir: str, name: str, *, zip_path: str,
                         commit: str | None) -> dict[str, Any]:
    """Move ``name`` from active → archived tombstone. Returns the registry."""
    reg = load_registry(base_dir)
    if name not in reg["active"]:
        raise ValueError(f"{name!r} is not an active ensemble member")
    reg["active"] = [n for n in reg["active"] if n != name]
    reg["archived"].append({
        "name": name,
        "archived_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "zip": zip_path,
        "commit": commit,
    })
    _write_json(registry_path(base_dir), reg)
    return reg


def restore_member_entry(base_dir: str, name: str) -> dict[str, Any]:
    """Move an archived tombstone back to ``active`` (the inverse of
    :func:`archive_member_entry`). The member directory must already be back
    on disk with a checkpoint, else the bootstrap would drop it again."""
    name = member_name(name)
    reg = load_registry(base_dir)
    if name in reg["active"]:
        raise ValueError(f"{name} is already active")
    if not any(str(t.get("name")) == name for t in reg["archived"]):
        raise ValueError(f"{name} is not archived")
    d = os.path.join(base_dir, name)
    if not (os.path.isdir(d) and _checkpoint_exists(d)):
        raise ValueError(f"{name} has no checkpoint on disk at {d}")
    reg["archived"] = [t for t in reg["archived"] if str(t.get("name")) != name]
    reg["active"] = sorted([*reg["active"], name])
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
