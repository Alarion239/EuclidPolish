"""Security boundary for the zero-login, loopback-only Web UI.

Four layers, wired in :mod:`euclid_polish.web.app` (``main`` checks the bind
host; :func:`~euclid_polish.web.app.create_app` registers the other three):

* :func:`validate_bind_host` — the server binds to loopback only.
* :func:`register_host_allowlist` — the ``Host`` header must name the
  loopback interface (Flask/Werkzeug ``TRUSTED_HOSTS``), which defeats DNS
  rebinding: a hostile page on a rebinding domain sends its own name as
  ``Host`` and is refused with 400 before any handler (or later
  ``before_request`` hook such as the SPA shell) runs.
* :func:`register_mutation_guard` — unsafe methods reject cross-site
  browser requests (``Sec-Fetch-Site`` / ``Origin``). Every state-changing
  endpoint must therefore be POST (or PUT/PATCH/DELETE), never GET.
* :func:`register_security_headers` — every response forbids framing
  (``X-Frame-Options: DENY`` + CSP ``frame-ancestors 'none'``: a hostile page
  could otherwise embed the console, and a POST made inside that frame is
  same-origin), MIME sniffing and cross-origin referrers.

:func:`is_same_origin_request` is the fetch-metadata test for the few GETs
that may still do work on the user's behalf (a ``?fresh=1`` re-render, a
FASRC file pull behind a link): a cross-site ``<img src>`` is refused.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

from flask import Flask, abort, jsonify, request
from werkzeug.exceptions import SecurityError

_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# Werkzeug compares the Host header without its port. "[::1]" matches a
# bracketed IPv6 loopback literal (what browsers send); "::1" is kept for
# clients that send the bare form.
TRUSTED_HOSTS: tuple[str, ...] = ("localhost", "127.0.0.1", "[::1]", "::1")

#: Set on every response (``setdefault``: a handler may set its own).
SECURITY_HEADERS: dict[str, str] = {
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": "frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
}


def _origin_key(value: str) -> tuple[str, str, int] | None:
    """Return a normalized (scheme, host, port) for an HTTP origin."""
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            return None
        port = parsed.port
    except ValueError:
        return None
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return parsed.scheme, parsed.hostname.rstrip(".").lower(), port


def validate_bind_host(host: str) -> str:
    """Accept only loopback bind hosts for the unauthenticated Web UI."""
    candidate = str(host).strip()
    if candidate.lower() == "localhost" or candidate in {"127.0.0.1", "::1"}:
        return host
    raise ValueError(
        "the zero-login Web UI may bind only to a loopback host "
        "(127.0.0.1, ::1, or localhost)"
    )


def _rejection(message: str, code: str, status: int):
    if request.path.startswith("/api/") or request.is_json:
        return jsonify({"ok": False, "error": message, "code": code}), status
    return message, status, {"Content-Type": "text/plain; charset=utf-8"}


def register_host_allowlist(
    app: Flask, hosts: tuple[str, ...] = TRUSTED_HOSTS,
) -> None:
    """Refuse requests whose ``Host`` header is not a loopback name.

    Sets ``TRUSTED_HOSTS`` so Werkzeug validates the header, and registers a
    ``before_request`` hook that surfaces the resulting ``SecurityError`` as
    a 400 *before* any other hook. Register it first: Flask stores a failed
    host check as the routing exception, which is only raised at dispatch —
    after ``before_request`` hooks that could otherwise answer on their own.
    """
    app.config["TRUSTED_HOSTS"] = list(hosts)

    @app.before_request
    def _enforce_trusted_host():
        if isinstance(request.routing_exception, SecurityError):
            return _rejection("untrusted Host header", "untrusted_host", 400)
        return None


def is_same_origin_request() -> bool:
    """Whether the request comes from this origin (or no browser context).

    ``Sec-Fetch-Site`` must be ``same-origin`` or ``none`` (typed URL,
    bookmark) — ``same-site`` is another localhost port — and an ``Origin``
    header, when sent, must be ours. Clients that send neither (curl, the
    test client, old browsers) count as same-origin: the Host allowlist
    already keeps rebinding pages out.
    """
    fetch_site = request.headers.get("Sec-Fetch-Site", "").strip().lower()
    if fetch_site and fetch_site not in {"same-origin", "none"}:
        return False
    supplied_origin = request.headers.get("Origin")
    if supplied_origin is None:
        return True
    return _origin_key(supplied_origin) == _origin_key(f"{request.scheme}://{request.host}")


def fresh_requested() -> bool:
    """``?fresh=1`` — honoured only for a same-origin request (the SPA's own
    re-render button); a cross-site ``<img src=…?fresh=1>`` gets the cached
    render. The state-changing way to re-render is the route's POST."""
    fresh = request.args.get("fresh", "").strip().lower() in ("1", "true", "yes")
    return fresh and is_same_origin_request()


def refuse_cross_site_get() -> None:
    """Abort 403 (JSON) unless :func:`is_same_origin_request` — for the GETs
    behind links that still do work (a FASRC file pull into the cache)."""
    if not is_same_origin_request():
        response = jsonify({"ok": False, "error": "cross-origin request rejected"})
        response.status_code = 403
        abort(response)


def refuse_cross_site_cache_fill(path: str) -> None:
    """Abort 404 (JSON) when ``path`` is not cached and the request is not
    :func:`is_same_origin_request` — for the GETs that render or recompute a
    missing cache (seconds to minutes): a cross-site ``<img src>`` gets what
    is cached, never a render."""
    if request.method not in ("GET", "HEAD"):
        return
    if not is_same_origin_request() and not os.path.isfile(path):
        response = jsonify({"ok": False,
                            "error": "not rendered yet — open it from the console"})
        response.status_code = 404
        abort(response)


def register_security_headers(app: Flask) -> None:
    """Add :data:`SECURITY_HEADERS` to every response (errors and refusals too)."""

    @app.after_request
    def _security_headers(response):
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response


def register_mutation_guard(app: Flask) -> None:
    """Reject browser cross-origin requests before unsafe route handlers."""

    @app.before_request
    def _enforce_same_origin_mutations():
        if request.method not in _UNSAFE_METHODS:
            return None

        fetch_site = request.headers.get("Sec-Fetch-Site", "").strip().lower()
        supplied_origin = request.headers.get("Origin")
        request_origin = f"{request.scheme}://{request.host}"
        origin_mismatch = supplied_origin is not None and (
            _origin_key(supplied_origin) != _origin_key(request_origin)
        )
        if fetch_site != "cross-site" and not origin_mismatch:
            return None

        if request.path.startswith("/api/") or request.is_json:
            return jsonify({"ok": False, "error": "cross-origin request rejected"}), 403
        return "cross-origin request rejected", 403


__all__ = [
    "SECURITY_HEADERS",
    "TRUSTED_HOSTS",
    "fresh_requested",
    "is_same_origin_request",
    "refuse_cross_site_cache_fill",
    "refuse_cross_site_get",
    "register_host_allowlist",
    "register_mutation_guard",
    "register_security_headers",
    "validate_bind_host",
]
