"""Security boundary for the zero-login, loopback-only Web UI.

Three layers, all registered by :func:`euclid_polish.web.app.create_app`:

* :func:`validate_bind_host` — the server binds to loopback only.
* :func:`register_host_allowlist` — the ``Host`` header must name the
  loopback interface (Flask/Werkzeug ``TRUSTED_HOSTS``), which defeats DNS
  rebinding: a hostile page on a rebinding domain sends its own name as
  ``Host`` and is refused with 400 before any handler (or later
  ``before_request`` hook such as the SPA shell) runs.
* :func:`register_mutation_guard` — unsafe methods reject cross-site
  browser requests (``Sec-Fetch-Site`` / ``Origin``). Every state-changing
  endpoint must therefore be POST (or PUT/PATCH/DELETE), never GET.
"""

from __future__ import annotations

from urllib.parse import urlsplit

from flask import Flask, jsonify, request
from werkzeug.exceptions import SecurityError

_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# Werkzeug compares the Host header without its port. "[::1]" matches a
# bracketed IPv6 loopback literal (what browsers send); "::1" is kept for
# clients that send the bare form.
TRUSTED_HOSTS: tuple[str, ...] = ("localhost", "127.0.0.1", "[::1]", "::1")


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
    "TRUSTED_HOSTS",
    "register_host_allowlist",
    "register_mutation_guard",
    "validate_bind_host",
]
