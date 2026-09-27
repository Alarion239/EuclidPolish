"""JSON error responses by path prefix — the app's single HTTPException handler.

Flask keeps ONE error handler per exception class per app, so two route
modules that each register ``@app.errorhandler(HTTPException)`` silently
replace one another (whichever ``register`` runs last wins). Route modules
therefore never register their own; they call :func:`json_errors_for` with
the path prefixes whose errors must be JSON. The first call installs the
shared, path-dispatching :func:`json_http_error`; later calls only add
prefixes. Under a registered prefix every HTTP error — routing 404/405
included, and an unhandled exception's 500 — is
``{"ok": false, "error": <description>}`` with its status code (the same
shape the route helpers return); every other path keeps Flask's default
response. ``create_app`` registers the whole ``/api/`` prefix.

:func:`int_arg` is the one integer query-argument parser of the routes: a
malformed value is a 400 (never a silent fallback to the default).
"""

from __future__ import annotations

from flask import Flask, abort, current_app, jsonify, request
from werkzeug.exceptions import HTTPException

#: ``app.extensions`` key holding the registered JSON-error path prefixes.
EXTENSION_KEY = "euclid_polish.json_error_prefixes"


def json_errors_for(app: Flask, *prefixes: str) -> None:
    """Answer HTTP errors under each of ``prefixes`` with JSON ``{error}``.

    Idempotent; installs :func:`json_http_error` on the first call.
    """
    registered: list[str] | None = app.extensions.get(EXTENSION_KEY)
    if registered is None:
        registered = []
        app.extensions[EXTENSION_KEY] = registered
        app.register_error_handler(HTTPException, json_http_error)
    for prefix in prefixes:
        if prefix not in registered:
            registered.append(prefix)


def json_prefixes(app: Flask) -> list[str]:
    """The path prefixes whose errors are JSON (registration order)."""
    return list(app.extensions.get(EXTENSION_KEY) or [])


def json_http_error(error: HTTPException):
    """JSON ``{ok: false, error}`` under a registered prefix, else Flask's default."""
    prefixes = tuple(current_app.extensions.get(EXTENSION_KEY) or ())
    if prefixes and request.path.startswith(prefixes):
        return (jsonify({"ok": False, "error": error.description or error.name}),
                error.code or 500)
    return error


def int_arg(name: str, default: int, *, lo: int | None = None, hi: int | None = None,
            clamp: bool = False) -> int:
    """Integer query argument ``name`` (``default`` when absent or blank).

    A malformed value aborts 400. Out of ``[lo, hi]`` it is clamped when
    ``clamp`` (page offsets / limits), else it aborts 400 too.
    """
    raw = request.args.get(name, "").strip()
    if raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        abort(400, description=f"{name} must be an integer, got {raw!r}")
    if lo is not None and value < lo:
        if not clamp:
            abort(400, description=f"{name} must be at least {lo}, got {value}")
        value = lo
    if hi is not None and value > hi:
        if not clamp:
            abort(400, description=f"{name} must be at most {hi}, got {value}")
        value = hi
    return value
