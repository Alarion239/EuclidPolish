"""JSON error responses by path prefix — the app's single HTTPException handler.

Flask keeps ONE error handler per exception class per app, so two route
modules that each register ``@app.errorhandler(HTTPException)`` silently
replace one another (whichever ``register`` runs last wins). Route modules
therefore never register their own; they call :func:`json_errors_for` with
the path prefixes whose errors must be JSON. The first call installs the
shared, path-dispatching :func:`json_http_error`; later calls only add
prefixes. Under a registered prefix every HTTP error — routing 404/405
included, and an unhandled exception's 500 — is ``{"error": <description>}``
with its status code; every other path keeps Flask's default response.
"""

from __future__ import annotations

from flask import Flask, current_app, jsonify, request
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
    """JSON ``{error}`` under a registered prefix, else Flask's default."""
    prefixes = tuple(current_app.extensions.get(EXTENSION_KEY) or ())
    if prefixes and request.path.startswith(prefixes):
        return (jsonify({"error": error.description or error.name}),
                error.code or 500)
    return error
