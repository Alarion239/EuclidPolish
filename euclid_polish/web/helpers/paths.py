"""paths helpers for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

from euclid_polish.config import Config
from euclid_polish.web import fasrc_config
from euclid_polish.web.fasrc_fetcher import _local_path_for
from flask import abort
from typing import List
import os


def _sky_records_remote_dir() -> str:
    cfg = fasrc_config.load()
    return f"{cfg.data_dir}/images/records_v2"


def _sky_records_local_dir() -> str:
    """Local cache dir mirroring the remote synthetic records dir.

    Same convention as :func:`fasrc_fetcher._local_path_for`, so the
    viewer reads exactly what ``/api/sky/sync`` (fetch_one_file)
    writes. The synthetic generator runs on FASRC, so the preview
    renders the synced shards — not a stale local copy."""
    any_path = f"{_sky_records_remote_dir()}/clean_validate.tfrecord"
    return os.path.dirname(_local_path_for(any_path))


def _inspectable_roots() -> List[str]:
    """Real paths under which a user may request any FITS file via /inspect.

    Anything outside this set is rejected with HTTP 403 — this prevents a
    crafted ``?fits=../../../etc/passwd`` from escaping the data tree.
    All roots are normalised via :func:`os.path.realpath` so symlinks
    can't bypass the check either.
    """
    roots = [
        Config.DEFAULT_OUTPUT_DIR,        # Euclid star cutouts
        Config.EUCLID_PSF_DIR,             # band PSFs
        Config.EUCLID_INFERENCE_DIR,       # real-Euclid inference outputs
        Config.VIS_DIR,                     # reconstruction PNG/FITS, demos, sky_fits
        Config.RECORDS_DIR_V2,              # TFRecords (not FITS) — kept for completeness
        Config.FASRC_CACHE_DIR,            # rsync'd-from-FASRC FITS files
    ]
    out = []
    for p in roots:
        if not p:
            continue
        try:
            out.append(os.path.realpath(p))
        except OSError:
            pass
    return out


def _resolve_inspectable_fits(raw_path: str) -> str:
    """Validate ``raw_path`` and return its real absolute path.

    Aborts the request with the appropriate HTTP error:

    * 400 — empty / non-FITS extension
    * 403 — resolves outside the allowed roots
    * 404 — does not exist on disk
    """
    if not raw_path:
        abort(400)
    p = raw_path if os.path.isabs(raw_path) else os.path.normpath(
        os.path.join(os.getcwd(), raw_path)
    )
    real = os.path.realpath(p)
    if not real.lower().endswith((".fits", ".fit", ".fits.gz")):
        abort(400)
    if not os.path.isfile(real):
        abort(404)
    for root in _inspectable_roots():
        if real == root or real.startswith(root + os.sep):
            return real
    abort(403)


def _resolve_trackable_file(raw_path: str) -> str:
    """Validate a file the user wants to back up into the tracking store.

    Same allow-listing as :func:`_resolve_inspectable_fits` (so a crafted
    ``../../etc/passwd`` can't be copied out of the data tree) but without
    the FITS-extension constraint, since images (PNG) are trackable too.
    Aborts 400 (empty), 403 (outside roots), or 404 (missing).
    """
    if not raw_path:
        abort(400)
    p = raw_path if os.path.isabs(raw_path) else os.path.normpath(
        os.path.join(os.getcwd(), raw_path)
    )
    real = os.path.realpath(p)
    if not os.path.isfile(real):
        abort(404)
    for root in _inspectable_roots():
        if real == root or real.startswith(root + os.sep):
            return real
    abort(403)


def _resolve_trackable_ckpt(raw_path: str) -> str:
    """Validate a checkpoint *directory* for a model backup.

    Constrained to live under the checkpoint root (``./ckpt`` by default)
    so the model-backup endpoint can't be pointed at an arbitrary tree.
    Aborts 403 (outside the ckpt root) or 404 (not a directory).
    """
    ckpt_root = os.path.realpath(
        os.path.dirname(os.path.realpath(Config.DEFAULT_CHECKPOINT_DIR))
    )
    real = os.path.realpath(raw_path)
    if not os.path.isdir(real):
        abort(404)
    if real == ckpt_root or real.startswith(ckpt_root + os.sep):
        return real
    abort(403)


def _safe_relpath(real_abs: str) -> str:
    """Return the project-rooted relative form of an absolute path.

    Used to build ``?fits=`` query strings that work regardless of the
    server's CWD. Falls back to the absolute path on failure.
    """
    try:
        rel = os.path.relpath(real_abs, os.getcwd())
        return rel if not rel.startswith("..") else real_abs
    except ValueError:
        return real_abs
