"""Project-wired default :class:`ProvStore`.

Kept out of ``store.py`` so the store itself stays decoupled from project paths.
The index lives under ``Config.PROV_DIR``; sidecars and stamp files are
discovered wherever the data lives (records dir, data dir, checkpoints and the
ensemble's member checkpoints).
"""

from __future__ import annotations

import os

from euclid_polish.config import Config
from euclid_polish.provenance.ids import ProvId
from euclid_polish.provenance.store import ProvStore

_default_store_cache = None


def default_store() -> ProvStore:
    """A :class:`ProvStore` rooted at the project's data + checkpoint dirs.

    Returns the same instance on repeated calls within a process (cached).
    """
    global _default_store_cache
    if _default_store_cache is None:
        # The ensemble's members (``<ckpt parent>/ensemble``, as
        # ``ensemble_registry.default_ensemble_dir`` resolves it — not imported
        # here: provenance sits below the tracking layer) carry the checkpoint
        # stamps a fresh mint must avoid.
        ensemble_dir = os.path.join(
            os.path.dirname(Config.DEFAULT_CHECKPOINT_DIR.rstrip("/")) or ".", "ensemble")
        candidates = [
            Config.RECORDS_DIR_V2,
            Config.DATA_DIR,
            Config.DEFAULT_CHECKPOINT_DIR,
            ensemble_dir,
        ]
        roots = [r for r in candidates if os.path.isdir(r)]
        _default_store_cache = ProvStore(Config.PROV_DIR, data_roots=roots or [Config.PROV_DIR])
    return _default_store_cache


def mint_id(store=None) -> ProvId:
    """Mint a fresh :class:`ProvId`, degrading to the sentinel on any failure.

    The guarded mint shared by every operator (simulator, forward model,
    archive, trained model) so a provenance-store hiccup never blocks the
    actual work — the artifact is produced unstamped-but-correct instead.
    """
    try:
        s = store if store is not None else default_store()
        return s.mint()
    except Exception:   # noqa: BLE001 — provenance is best-effort
        return ProvId.sentinel()
