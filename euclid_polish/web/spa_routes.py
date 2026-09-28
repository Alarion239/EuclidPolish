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

Redirects come in two kinds, tried in this order (first match wins):

- ``redirectRules``: query-aware rules. ``from`` is a path pattern whose
  ``:name`` segments bind one segment each, restricted to ``params[name]``;
  ``query`` requires keys (``"*"`` any value, a string, or a list of allowed
  values; a repeated key is judged by its last value). The target is ``to``
  with every ``:name`` substituted; its query is the original pairs (order
  kept) after ``drop``, then ``rename``, then ``map`` (value per key), then
  ``prefix`` (value per key), then ``set`` (replace every occurrence or
  append; ``:name`` substituted), form-encoded byte for byte like the
  browser's ``URLSearchParams`` (``spa_redirect_cases.json`` pins this module
  and ``app/manifest.ts`` to the same output).
- ``redirects``: exact-path matches (trailing slash ignored) that append the
  original query string untouched.

``/app/<rest>`` (the pre-rework SPA prefix) maps to ``/<rest>`` with every
leading slash/backslash collapsed, so it never leaves this host.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

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


# WHATWG application/x-www-form-urlencoded keeps these bytes as they are.
_FORM_SAFE = frozenset(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789*-._"
)


def _form_encode(text: str) -> str:
    """WHATWG application/x-www-form-urlencoded, byte for byte like URLSearchParams."""
    out = []
    for byte in text.encode("utf-8"):
        if byte in _FORM_SAFE:
            out.append(chr(byte))
        elif byte == 0x20:
            out.append("+")
        else:
            out.append(f"%{byte:02X}")
    return "".join(out)


def _match_path(rule: Mapping[str, Any], path: str) -> dict[str, str] | None:
    """The ``:name`` bindings of ``path`` against the rule's ``from``, or None."""
    want = _normalise(str(rule["from"])).split("/")
    got = _normalise(path).split("/")
    if len(want) != len(got):
        return None
    allowed = rule.get("params") or {}
    bound: dict[str, str] = {}
    for w, g in zip(want, got, strict=True):
        if w.startswith(":"):
            name = w[1:]
            if not g or (name in allowed and g not in allowed[name]):
                return None
            bound[name] = g
        elif w != g:
            return None
    return bound


def _query_matches(cond: Mapping[str, Any] | None, pairs: list[tuple[str, str]]) -> bool:
    if not cond:
        return True
    have = dict(pairs)
    for key, want in cond.items():
        if key not in have:
            return False
        if want == "*":
            continue
        options = want if isinstance(want, list) else [want]
        if have[key] not in options:
            return False
    return True


def _substitute(text: str, bound: Mapping[str, str]) -> str:
    for name, value in bound.items():
        text = text.replace(f":{name}", value)
    return text


def _apply_rule(
    rule: Mapping[str, Any], bound: Mapping[str, str], pairs: list[tuple[str, str]]
) -> str:
    target = _substitute(str(rule["to"]), bound)
    drop = set(rule.get("drop") or [])
    rename = rule.get("rename") or {}
    mapping = rule.get("map") or {}
    prefix = rule.get("prefix") or {}
    out = [(rename.get(k, k), v) for k, v in pairs if k not in drop]
    out = [(k, (mapping.get(k) or {}).get(v, v)) for k, v in out]
    out = [(k, f"{prefix[k]}{v}" if k in prefix else v) for k, v in out]
    for key, value in (rule.get("set") or {}).items():
        text = _substitute(str(value), bound)
        if any(k == key for k, _ in out):
            out = [(k, text if k == key else v) for k, v in out]
        else:
            out.append((key, text))
    if not out:
        return target
    return target + "?" + "&".join(f"{_form_encode(k)}={_form_encode(v)}" for k, v in out)


def redirect_target(
    path: str,
    query: str | bytes | None = "",
    manifest: Manifest | None = None,
) -> str | None:
    """Where a legacy URL permanently moves to, else None.

    A ``redirectRules`` match rewrites the query; an exact ``redirects``
    entry (and ``/app/<rest>``) keeps it as it was. ``/app/<rest>`` resolves
    ``<rest>`` once more, so an old path under ``/app`` lands in one hop.
    """
    if not path:
        return None
    manifest = _default_manifest() if manifest is None else manifest
    rules = manifest.get("redirectRules") or []
    if rules:
        text = query.decode("utf-8", "replace") if isinstance(query, bytes) else (query or "")
        pairs = parse_qsl(text, keep_blank_values=True)
        for rule in rules:
            bound = _match_path(rule, path)
            if bound is not None and _query_matches(rule.get("query"), pairs):
                return _apply_rule(rule, bound, pairs)
    normalised = _normalise(path)
    target = (manifest.get("redirects") or {}).get(normalised)
    if target is None and path.startswith(_APP_PREFIX + "/"):
        rest = _same_host_path(path[len(_APP_PREFIX):])
        # Resolve the old path in one hop: /app/ensemble → /models/…, not /ensemble.
        onward = redirect_target(rest, query, manifest)
        if onward is not None:
            return onward
        target = rest
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
