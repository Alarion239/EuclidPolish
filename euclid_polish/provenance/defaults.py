"""Project-wired default :class:`ProvStore`.

Kept out of ``store.py`` so the store itself stays decoupled from project paths.
The index lives under ``Config.PROV_DIR``; sidecars are discovered wherever the
data lives (records dir, data dir, checkpoints).
"""

from __future__ import annotations

import os

from euclid_polish.config import Config
from euclid_polish.provenance.store import ProvStore


def default_store() -> ProvStore:
    """A :class:`ProvStore` rooted at the project's data + checkpoint dirs."""
    candidates = [
        Config.RECORDS_DIR_V2,
        Config.DATA_DIR,
        os.path.join(os.getcwd(), "ckpt"),
    ]
    roots = [r for r in candidates if os.path.isdir(r)]
    return ProvStore(Config.PROV_DIR, data_roots=roots or [Config.PROV_DIR])
