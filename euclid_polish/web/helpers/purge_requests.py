"""Durable "a stale-cube purge is due" flag.

Jobs that make cached cubes stale (archiving or pulling members, the FASRC
ensemble mirror, a records sync) call :func:`request_stale_purge`; the job
registry's finish hook (:mod:`euclid_polish.web.helpers.stale_purge`) starts
the purge once no job runs. A file under ``<vis>/ensemble`` so a request
survives a console restart. Config-only imports: the modules that request a
purge must not import the purge itself (an import cycle via ensemble_viz)."""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from typing import Any

from euclid_polish.config import Config

PENDING_NAME = "stale_purge_pending.json"
_MAX_REASONS = 20
_LOCK = threading.Lock()


def pending_path() -> str:
    return os.path.join(os.path.abspath(Config.VIS_DIR), "ensemble", PENDING_NAME)


def read_pending() -> dict[str, Any] | None:
    """The pending request (``reasons``, ``requested_at``) or ``None``."""
    try:
        with open(pending_path()) as handle:
            value = json.load(handle)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def request_stale_purge(reason: str) -> None:
    """Record that cached cubes may have gone stale (``reason``: what
    happened). Each request gets a fresh ``requested_at`` token."""
    with _LOCK:
        reasons = [str(r) for r in (read_pending() or {}).get("reasons", [])]
        payload = {"reasons": [*reasons, str(reason)][-_MAX_REASONS:],
                   "requested_at": time.time_ns()}
        path = pending_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w") as handle:
            json.dump(payload, handle)
        os.replace(tmp, path)


def clear_pending(token: int | None) -> bool:
    """Remove the request if it is still the one stamped ``token``: one made
    while the purge ran survives it. Whether it was removed."""
    with _LOCK:
        pending = read_pending()
        if pending is None or pending.get("requested_at") != token:
            return False
        with contextlib.suppress(FileNotFoundError):
            os.remove(pending_path())
        return True
