"""Shared low-level helpers for the provenance package."""

from __future__ import annotations

import json
import os
from typing import Any


def _atomic_write_json(path: str, obj: Any) -> None:
    """Atomic JSON write (tmp + os.replace) — the provenance-store idiom."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fp:
        json.dump(obj, fp, indent=2, sort_keys=True)
        fp.write("\n")
    os.replace(tmp, path)
