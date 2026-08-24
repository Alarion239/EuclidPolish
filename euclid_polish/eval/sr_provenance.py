"""Provenance for super-resolved FITS outputs (the eval / web SR path).

An ``SR.fits`` gets PROVID/PRODBY/PROVPAR header cards plus a persisted
SR-cutout :class:`Artifact`, parented on the producing model (read from the
checkpoint's ``provenance.json``). All best-effort: callers wrap the call so a
provenance hiccup never blocks reconstruction.
"""

from __future__ import annotations

import os
from typing import Any

from euclid_polish.provenance.checkpoint import model_id_of_checkpoint
from euclid_polish.provenance.defaults import default_store
from euclid_polish.provenance.fits import write_stamp_cards
from euclid_polish.provenance.gitinfo import capture_git
from euclid_polish.provenance.records import (
    Artifact,
    Format,
    Process,
    Stamp,
)
from euclid_polish.provenance.store import ProvStore


def write_sr_provenance(header, *, checkpoint_dir: str, sr_fits_path: str,
                        store: ProvStore | None = None,
                        git: dict[str, Any] | None = None,
                        descriptors: dict[str, Any] | None = None) -> Stamp:
    """Write provenance cards into ``header`` and persist the SR artifact.

    Returns the :class:`Stamp` written (caller may ignore it). The model id —
    when the checkpoint carries one — becomes the SR's parent, so staleness can
    be checked later.

    """
    store = store if store is not None else default_store()
    if git is None:
        git = capture_git()

    model_id = model_id_of_checkpoint(checkpoint_dir)
    parents = tuple(m for m in (model_id,) if m is not None)

    run = Process.inference(id=store.mint(), git=git, status="ok", inputs=parents)
    store.put(run)

    sr_id = store.mint()
    stamp = Stamp(id=sr_id, produced_by=run.id, parents=parents, schema_version=3)
    write_stamp_cards(header, stamp)

    art = Artifact.sr_cutout(
        id=sr_id, git=git, produced_by=run.id, format=Format.FITS,
        path=sr_fits_path, parents=parents, descriptors=descriptors or {},
    )
    store.put(art, sidecar_dir=os.path.dirname(sr_fits_path) or None)
    return stamp
