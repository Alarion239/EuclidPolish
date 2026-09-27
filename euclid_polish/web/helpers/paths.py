"""paths helpers for the EuclidPolish web UI (extracted from app.py).

The Inspect workspace (spec §8.6) reads FITS files from a fixed set of
*inspectable roots* (:func:`inspect_roots`): every path the file browser lists
or the inspector opens must resolve (symlinks expanded) inside one of them,
so a crafted ``?fits=../../etc/passwd`` can never escape the data tree.
"""
from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
from typing import Any, NoReturn

from flask import abort, jsonify
from werkzeug.exceptions import default_exceptions

from euclid_polish.config import Config
from euclid_polish.web import fasrc_config
from euclid_polish.web.fasrc_fetcher import _local_path_for

#: The repository root (``poster/``, ``output/`` live here).
REPO_ROOT = Path(__file__).resolve().parents[3]

#: File names the inspector opens (plain, gzip and tile-compressed FITS).
FITS_SUFFIXES = (".fits", ".fit", ".fts", ".fits.gz", ".fit.gz", ".fts.gz", ".fits.fz", ".fz")

#: Browse / search budgets: a listing never returns more entries than this, a
#: search visits at most ``SEARCH_MAX_DIRS`` directories for at most
#: ``SEARCH_MAX_SECONDS`` and returns at most ``SEARCH_MAX_RESULTS`` hits.
BROWSE_MAX_ENTRIES = 5000
SEARCH_MAX_RESULTS = 500
SEARCH_MAX_DIRS = 20_000
SEARCH_MAX_SECONDS = 4.0


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


def _viewer_results_dir() -> str:
    """The saved viewer-results root (``helpers.viewer_results.results_root``).

    Duplicated here (three lines) because ``viewer_results`` imports
    ``viewer_data``, which imports this module: importing it would be a cycle.
    """
    explicit = os.environ.get("EUCLID_POLISH_RESULTS_DIR")
    if explicit:
        return os.path.expanduser(explicit)
    data_root = os.environ.get("EUCLID_POLISH_DATA_DIR") or os.fspath(Config.DATA_DIR)
    return os.path.join(os.path.expanduser(data_root), "viewer_results")


def _root_specs() -> list[tuple[str, str, str]]:
    """``(id, label, configured path)`` of every inspectable root, in browse order."""
    data = os.fspath(Config.DATA_DIR)
    return [
        ("eval", "Evaluation results", Config.EVAL_RESULTS_DIR),
        ("inference", "Euclid inference", Config.EUCLID_INFERENCE_DIR),
        ("jwst", "JWST × Euclid", os.path.join(data, "jwst_euclid_overlap")),
        ("sky", "Euclid sky (archive fields)", Config.EUCLID_SKY_DIR),
        ("stars", "Euclid star cutouts", Config.DEFAULT_OUTPUT_DIR),
        ("psf", "Band PSFs", Config.EUCLID_PSF_DIR),
        ("viewer-results", "Viewer results", _viewer_results_dir()),
        ("population", "Population comparison", os.path.join(data, "population_comparison")),
        ("tng", "TNG SKIRT", Config.TNG_SKIRT_DIR),
        ("vis", "Figures & reconstructions", Config.VIS_DIR),
        ("fasrc-cache", "FASRC cache", Config.FASRC_CACHE_DIR),
        ("records", "Synthetic records", Config.RECORDS_DIR_V2),
        ("poster", "Poster (repo)", os.fspath(REPO_ROOT / "poster")),
        ("output", "Output (repo)", os.fspath(REPO_ROOT / "output")),
        ("tracking", "Tracking", Config.TRACKING_DIR),
    ]


def inspect_roots() -> list[dict[str, Any]]:
    """Every inspectable root: ``{id, label, path (real), rel, exists}``.

    Computed per call (tests and the time-travel sandbox re-point ``Config``).
    Duplicate real paths collapse into the first entry.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root_id, label, configured in _root_specs():
        if not configured:
            continue
        try:
            real = os.path.realpath(configured)
        except OSError:
            continue
        if real in seen:
            continue
        seen.add(real)
        out.append({
            "id": root_id, "label": label, "path": real,
            "rel": _safe_relpath(real), "exists": os.path.isdir(real),
        })
    return out


def _inspectable_roots() -> list[str]:
    """Real paths under which a user may request any FITS file via /inspect.

    Anything outside this set is rejected with HTTP 403 — this prevents a
    crafted ``?fits=../../../etc/passwd`` from escaping the data tree.
    All roots are normalised via :func:`os.path.realpath` so symlinks
    can't bypass the check either.
    """
    return [root["path"] for root in inspect_roots()]


def root_of(real: str, roots: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """The inspectable root containing the real path ``real`` (deepest wins);
    ``roots`` reuses an :func:`inspect_roots` result across many lookups."""
    best: dict[str, Any] | None = None
    for root in inspect_roots() if roots is None else roots:
        path = root["path"]
        if (real == path or real.startswith(path + os.sep)) and (
                best is None or len(path) > len(best["path"])):
            best = root
    return best


def _real_from_raw(raw_path: str) -> str:
    p = raw_path if os.path.isabs(raw_path) else os.path.normpath(
        os.path.join(os.getcwd(), raw_path)
    )
    return os.path.realpath(p)


def is_fits_name(name: str) -> bool:
    return name.lower().endswith(FITS_SUFFIXES)


def _resolve_inspectable_fits(raw_path: str) -> str:
    """Validate ``raw_path`` and return its real absolute path.

    Aborts the request with the appropriate HTTP error (the description is
    the JSON ``error`` under ``/api/inspect``):

    * 400 — empty / non-FITS extension
    * 403 — resolves outside the allowed roots
    * 404 — does not exist on disk
    """
    if not raw_path:
        abort(400, description="pass ?fits=<project-relative FITS path>")
    real = _real_from_raw(raw_path)
    if not is_fits_name(real):
        abort(400, description=f"not a FITS file name: {os.path.basename(raw_path)}")
    if not os.path.isfile(real):
        abort(404, description=f"no such FITS file: {raw_path}")
    if root_of(real) is not None:
        return real
    abort(403, description=f"{raw_path} is outside the inspectable data roots")


def resolve_inspect_dir(raw_path: str) -> str:
    """Validate a directory to browse; 403 outside the roots, 404 missing."""
    real = _real_from_raw(raw_path)
    if root_of(real) is None:
        abort(403, description=f"{raw_path} is outside the inspectable data roots")
    if not os.path.isdir(real):
        abort(404, description=f"no such directory: {raw_path}")
    return real


def _crumbs(real: str, root: dict[str, Any]) -> list[dict[str, str]]:
    """Breadcrumbs from the root down to ``real``: ``[{name, rel}]``."""
    crumbs = [{"name": root["label"], "rel": root["rel"]}]
    tail = os.path.relpath(real, root["path"])
    if tail == ".":
        return crumbs
    current = root["path"]
    for part in tail.split(os.sep):
        current = os.path.join(current, part)
        crumbs.append({"name": part, "rel": _safe_relpath(current)})
    return crumbs


def _entry(path: str, name: str, is_dir: bool, st: os.stat_result | None) -> dict[str, Any]:
    return {
        "name": name, "rel": _safe_relpath(path), "kind": "dir" if is_dir else "fits",
        "size": None if is_dir or st is None else int(st.st_size),
        "mtime": None if st is None else float(st.st_mtime),
    }


def list_inspect_dir(real: str) -> dict[str, Any]:
    """One directory of the file browser: sub-directories and FITS files.

    Hidden names (``.…``) are skipped; entries resolving outside every root
    (a symlink out of the tree) are dropped. Directories first, then files,
    each by name. ``other`` counts the non-FITS files not listed.
    """
    dirs: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    other = 0
    truncated = False
    roots = inspect_roots()
    try:
        with os.scandir(real) as it:
            for item in it:
                if item.name.startswith("."):
                    continue
                if len(dirs) + len(files) >= BROWSE_MAX_ENTRIES:
                    truncated = True
                    break
                try:
                    is_dir = item.is_dir()
                    if not is_dir and not is_fits_name(item.name):
                        other += 1
                        continue
                    target = os.path.realpath(item.path)
                    if root_of(target, roots) is None:
                        continue
                    st = item.stat()
                except OSError:
                    continue
                (dirs if is_dir else files).append(_entry(item.path, item.name, is_dir, st))
    except OSError as exc:
        abort(403, description=f"cannot list {os.path.basename(real)}: {exc.strerror or exc}")
    dirs.sort(key=lambda e: e["name"].lower())
    files.sort(key=lambda e: e["name"].lower())
    return {"entries": dirs + files, "other": other, "truncated": truncated}


def search_inspect_tree(bases: list[str], query: str) -> dict[str, Any]:
    """FITS files under ``bases`` whose path (relative to the base) contains
    every whitespace-separated token of ``query`` (case-insensitive).

    Bounded: at most :data:`SEARCH_MAX_RESULTS` hits, :data:`SEARCH_MAX_DIRS`
    directories and :data:`SEARCH_MAX_SECONDS`; ``truncated`` says when a bound
    stopped the walk. Symlinked directories are not followed.
    """
    tokens = [t for t in query.lower().split() if t]
    results: list[dict[str, Any]] = []
    visited = 0
    truncated = False
    deadline = time.monotonic() + SEARCH_MAX_SECONDS
    for base in bases:
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            visited += 1
            if visited > SEARCH_MAX_DIRS or time.monotonic() > deadline:
                truncated = True
                break
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            rel_dir = os.path.relpath(dirpath, base)
            for name in sorted(filenames):
                if not is_fits_name(name):
                    continue
                hay = (name if rel_dir == "." else os.path.join(rel_dir, name)).lower()
                if all(t in hay for t in tokens):
                    path = os.path.join(dirpath, name)
                    with contextlib.suppress(OSError):
                        results.append(_entry(path, name, False, os.stat(path)))
                    if len(results) >= SEARCH_MAX_RESULTS:
                        truncated = True
                        break
            if truncated:
                break
        if truncated:
            break
    return {"entries": results, "truncated": truncated, "visited": visited}


def browse_inspectable(raw_dir: str, query: str = "") -> dict[str, Any]:
    """The file browser payload (``GET /api/inspect/browse``).

    No ``dir``: the roots (or a search across every root). A ``dir``: its
    listing with breadcrumbs (or a search under it).
    """
    roots = inspect_roots()
    query = query.strip()
    if not raw_dir:
        if query:
            found = search_inspect_tree([r["path"] for r in roots if r["exists"]], query)
            return {"dir": None, "root": None, "crumbs": [], "query": query, "roots": roots, **found}
        entries = [{
            "name": r["label"], "rel": r["rel"], "kind": "root", "root_id": r["id"],
            "exists": True, "size": None, "mtime": None,
        } for r in roots if r["exists"]]
        return {"dir": None, "root": None, "crumbs": [], "query": "", "roots": roots,
                "entries": entries, "other": 0, "truncated": False}
    real = resolve_inspect_dir(raw_dir)
    root = root_of(real)
    assert root is not None  # resolve_inspect_dir guarantees it
    payload = {"dir": _safe_relpath(real), "root": root, "crumbs": _crumbs(real, root),
               "query": query, "roots": roots}
    if query:
        return {**payload, **search_inspect_tree([real], query)}
    return {**payload, **list_inspect_dir(real)}


def _abort_json(code: int, message: str) -> NoReturn:
    """Abort with ``code`` and a JSON ``{ok: false, error}`` body on ANY path.

    A route outside the JSON-error prefixes (``/api/tracking``) would
    otherwise answer with Flask's HTML page, and the SPA's toast could only
    show the status line. The exception keeps ``description`` too, so a
    JSON-prefix handler (and callers) still read the message.
    """
    response = jsonify({"ok": False, "error": message})
    response.status_code = code
    raise default_exceptions[code](description=message, response=response)


def _resolve_trackable_file(raw_path: str) -> str:
    """Validate a file the user wants to back up into the tracking store.

    Same allow-listing as :func:`_resolve_inspectable_fits` (so a crafted
    ``../../etc/passwd`` can't be copied out of the data tree) but without
    the FITS-extension constraint, since images (PNG) are trackable too.
    Aborts 400 (empty), 403 (outside roots), or 404 (missing), each with a
    JSON ``{ok: false, error}`` body (:func:`_abort_json`).
    """
    if not raw_path:
        _abort_json(400, "pass path=<project-relative file path>")
    real = _real_from_raw(raw_path)
    if not os.path.isfile(real):
        _abort_json(404, f"no such file: {raw_path}")
    if root_of(real) is not None:
        return real
    _abort_json(403, f"{raw_path} is outside the inspectable data roots")


def _resolve_trackable_ckpt(raw_path: str) -> str:
    """Validate a checkpoint *directory* for a model backup.

    Constrained to live under the checkpoint root (``./ckpt`` by default)
    so the model-backup endpoint can't be pointed at an arbitrary tree.
    Aborts 403 (outside the ckpt root) or 404 (not a directory), each with a
    JSON ``{ok: false, error}`` body (:func:`_abort_json`).
    """
    ckpt_root = os.path.realpath(
        os.path.dirname(os.path.realpath(Config.DEFAULT_CHECKPOINT_DIR))
    )
    real = os.path.realpath(raw_path)
    if not os.path.isdir(real):
        _abort_json(404, f"no such checkpoint directory: {raw_path}")
    if real == ckpt_root or real.startswith(ckpt_root + os.sep):
        return real
    _abort_json(403, f"{raw_path} is outside the checkpoint root")


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
