"""Security boundary for the zero-login, loopback-only Web UI."""

from __future__ import annotations

from urllib.parse import urlsplit

from flask import Flask, jsonify, request

_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


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
