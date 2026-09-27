"""HTTP caching and compression of the built SPA (``/static/dist/``).

Vite writes every bundle chunk as ``assets/<name>-<content hash>.<ext>``: a
given URL never changes content, so those files are served with
``Cache-Control: public, max-age=31536000, immutable`` (no revalidation round
trip per chunk on every navigation). The shell (``index.html``, served by the
SPA hook for page paths) and unhashed files keep Flask's ``no-cache``.

Text assets (JS, CSS, JSON, SVG, WASM, …) larger than :data:`MIN_GZIP_BYTES`
are gzip-compressed when the client accepts it — the Aladin chunk shrinks
from ~2.4 MB to ~0.8 MB. The compressed bytes are memoised per
``(path, mtime, size)`` (bounded by :data:`GZIP_CACHE_BYTES`), so each file
is compressed once per build. The gzip variant carries its own ETag
(``"<etag>-gzip"``) and a matching ``If-None-Match`` is answered 304. Range
requests (206) are never compressed.
"""

from __future__ import annotations

import gzip
import os
import re
import threading
from collections import OrderedDict

from flask import Flask, Response, request
from werkzeug.security import safe_join

#: URL prefix of the built SPA and of its content-hashed chunks.
DIST_PREFIX = "/static/dist/"
ASSETS_PREFIX = "/static/dist/assets/"
IMMUTABLE = "public, max-age=31536000, immutable"
#: Files smaller than this are not worth a gzip frame.
MIN_GZIP_BYTES = 1024
#: Upper bound of the memoised compressed bytes (LRU by insertion/use).
GZIP_CACHE_BYTES = 64 * 1024 * 1024
GZIP_LEVEL = 6

# ``index-3tEk6LPN.js``, ``Cutouts-L84R61E-.js``: Vite's 8-character
# base64url content hash right before the extension.
_HASHED = re.compile(r"-[A-Za-z0-9_-]{8,}\.[A-Za-z0-9]+$")
_COMPRESSIBLE = ("application/javascript", "text/javascript", "application/json",
                 "application/manifest+json", "application/wasm", "application/xml",
                 "image/svg+xml")

_LOCK = threading.Lock()
_CACHE: OrderedDict[tuple[str, int, int], bytes] = OrderedDict()
_CACHE_BYTES = [0]


def clear_cache() -> None:
    """Forget every memoised compressed file (tests)."""
    with _LOCK:
        _CACHE.clear()
        _CACHE_BYTES[0] = 0


def is_hashed_asset(path: str) -> bool:
    """Whether a request path names a content-hashed Vite chunk."""
    return path.startswith(ASSETS_PREFIX) and bool(_HASHED.search(path))


def _compressible(mimetype: str | None) -> bool:
    kind = (mimetype or "").lower()
    return kind.startswith("text/") or kind in _COMPRESSIBLE


def _gzipped(path: str) -> bytes | None:
    """The gzip bytes of one file (memoised on its mtime and size)."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    key = (path, stat.st_mtime_ns, stat.st_size)
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            _CACHE.move_to_end(key)
            return cached
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError:
        return None
    packed = gzip.compress(raw, compresslevel=GZIP_LEVEL, mtime=0)
    with _LOCK:
        _CACHE[key] = packed
        _CACHE_BYTES[0] += len(packed)
        while _CACHE_BYTES[0] > GZIP_CACHE_BYTES and len(_CACHE) > 1:
            _old, dropped = _CACHE.popitem(last=False)
            _CACHE_BYTES[0] -= len(dropped)
    return packed


def _accepts_gzip() -> bool:
    return request.accept_encodings["gzip"] > 0 or request.accept_encodings["x-gzip"] > 0


def _compress(app: Flask, response: Response) -> Response:
    if (response.status_code != 200 or "Content-Encoding" in response.headers
            or "Content-Range" in response.headers or not _compressible(response.mimetype)
            or not _accepts_gzip() or app.static_folder is None):
        return response
    relative = request.path[len("/static/"):]
    source = safe_join(app.static_folder, relative)
    if source is None or not os.path.isfile(source) or os.path.getsize(source) < MIN_GZIP_BYTES:
        return response
    packed = _gzipped(source)
    if packed is None:
        return response
    etag, _weak = response.get_etag()
    gzip_etag = f"{etag}-gzip" if etag else None
    if gzip_etag and request.if_none_match.contains(gzip_etag):
        unchanged = Response(status=304)
        for name in ("Cache-Control", "Last-Modified", "Expires"):
            if name in response.headers:
                unchanged.headers[name] = response.headers[name]
        unchanged.set_etag(gzip_etag)
        unchanged.vary.add("Accept-Encoding")
        response.close()
        return unchanged
    response.close()
    response.direct_passthrough = False
    response.set_data(packed)
    response.headers["Content-Encoding"] = "gzip"
    if gzip_etag:
        response.set_etag(gzip_etag)
    return response


def register_static_caching(app: Flask) -> None:
    """Immutable caching + gzip for ``/static/dist/`` (see the module doc)."""

    @app.after_request
    def _static_dist_caching(response: Response) -> Response:
        if not request.path.startswith(DIST_PREFIX):
            return response
        response.vary.add("Accept-Encoding")
        if is_hashed_asset(request.path) and response.status_code in (200, 206, 304):
            response.headers["Cache-Control"] = IMMUTABLE
        return _compress(app, response)


__all__ = [
    "ASSETS_PREFIX",
    "DIST_PREFIX",
    "IMMUTABLE",
    "clear_cache",
    "is_hashed_asset",
    "register_static_caching",
]
