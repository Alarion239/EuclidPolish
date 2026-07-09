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
import os
from datetime import UTC, datetime
from typing import Any

from euclid_polish.config import Config
from euclid_polish.tracking._utils import _read_json, _write_json

_MEMBER_GLOB = "member_*"
REGISTRY_FILENAME = "ensemble_registry.json"


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
    from euclid_polish.ensemble import member_is_starless  # lazy: avoid cycle
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
