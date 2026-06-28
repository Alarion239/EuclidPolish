"""Shared low-level helpers for the tracking package.

Extracted here so :mod:`~euclid_polish.tracking.store` and
:mod:`~euclid_polish.tracking.timetravel` can import them without
duplicating definitions.
"""

from __future__ import annotations

import datetime
import json
import os
from typing import Any, Dict, Optional


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _write_json(path: str, obj: Any) -> None:
    """Atomic JSON write (makedirs + tmp + os.replace)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fp:
        json.dump(obj, fp, indent=2, sort_keys=True)
        fp.write("\n")
    os.replace(tmp, path)


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as fp:
            return json.load(fp)
    except (OSError, json.JSONDecodeError):
        return None
