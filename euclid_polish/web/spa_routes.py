"""SPA page matcher driven by the committed route manifest (contract C1).

``euclid_polish/web/spa_routes.json`` is the single source of truth for the
console's page URLs: the frontend router imports it, and Flask uses this
module to decide which requests get the SPA shell (``static/dist/index.html``)
and which legacy URLs are permanently redirected.

A **page path** is a workspace ``path`` (``:param`` segments substituted from
their allowed values) optionally followed by ``/<tab>`` with ``tab`` in that
workspace's ``tabs``. Anything else — ``/ensemble/status.json``,
``/inspect/preview.png``, ``/api/...`` — is not a page and reaches its normal
Flask handler.

Redirects are exact-path matches (trailing slash ignored) and preserve the
query string. ``/app/<rest>`` (the pre-rework SPA prefix) maps to ``/<rest>``
with every leading slash/backslash collapsed, so it never leaves this host.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path(__file__).with_name("spa_routes.json")

_APP_PREFIX = "/app"

Manifest = Mapping[str, Any]


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Parse the route manifest (the committed one when ``path`` is None)."""
    if path is None:
        return _default_manifest()
    return _read(Path(path))


def _read(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("workspaces"), list
    ):
        raise ValueError(f"{path}: not a route manifest (no workspaces list)")
    return manifest


@functools.lru_cache(maxsize=1)
def _default_manifest() -> dict[str, Any]:
    return _read(MANIFEST_PATH)


def _normalise(path: str) -> str:
    """``/sky/`` → ``/sky``; the root stays ``/``."""
    return path.rstrip("/") or "/"


def _workspace_paths(workspace: Mapping[str, Any]) -> list[str]:
    """Concrete paths of one workspace (every allowed ``:param`` value)."""
    paths = [str(workspace["path"])]
    for name, values in (workspace.get("params") or {}).items():
        paths = [
            path.replace(f":{name}", str(value))
            for path in paths
            for value in values
        ]
    return paths


def _page_paths(manifest: Manifest) -> frozenset[str]:
    pages: set[str] = set()
    for workspace in manifest["workspaces"]:
        tabs = [str(tab) for tab in workspace.get("tabs") or ()]
        for path in _workspace_paths(workspace):
            base = _normalise(path)
            pages.add(base)
            prefix = "" if base == "/" else base
            pages.update(f"{prefix}/{tab}" for tab in tabs)
    return frozenset(pages)


@functools.lru_cache(maxsize=1)
def _default_page_paths() -> frozenset[str]:
    return _page_paths(_default_manifest())


def page_paths(manifest: Manifest | None = None) -> frozenset[str]:
    """Every exact page path (normalised, no trailing slash) of a manifest."""
    if manifest is None:
        return _default_page_paths()
    return _page_paths(manifest)


def is_page_path(path: str, manifest: Manifest | None = None) -> bool:
    """True when ``path`` is served by the SPA shell."""
    if not path:
        return False
    return _normalise(path) in page_paths(manifest)


def _query_suffix(query: str | bytes | None) -> str:
    if not query:
        return ""
    text = query.decode("utf-8", "replace") if isinstance(query, bytes) else query
    return f"?{text}" if text else ""


# A browser reads ``//host`` and ``/\\host`` as protocol-relative URLs and
# drops ASCII tab/newline anywhere in a URL, so none of these may lead the
# rest of an ``/app/<rest>`` path.
_UNSAFE_LEADING = "/\\" + "".join(chr(code) for code in range(0x21))


def _same_host_path(rest: str) -> str:
    """``rest`` as a path on this host: ``//evil.example`` → ``/evil.example``.

    Every leading slash, backslash, space or control character is collapsed
    into one ``/`` so the ``/app/<rest>`` redirect can never become an open
    redirect to another host.
    """
    return "/" + rest.lstrip(_UNSAFE_LEADING)


def redirect_target(
    path: str,
    query: str | bytes | None = "",
    manifest: Manifest | None = None,
) -> str | None:
    """Where a legacy URL permanently moves to (query preserved), else None."""
    if not path:
        return None
    manifest = _default_manifest() if manifest is None else manifest
    normalised = _normalise(path)
    target = (manifest.get("redirects") or {}).get(normalised)
    if target is None and path.startswith(_APP_PREFIX + "/"):
        target = _same_host_path(path[len(_APP_PREFIX):])
    if target is None:
        return None
    return f"{target}{_query_suffix(query)}"


__all__ = [
    "MANIFEST_PATH",
    "is_page_path",
    "load_manifest",
    "page_paths",
    "redirect_target",
]
