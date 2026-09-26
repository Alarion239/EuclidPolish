"""Per-route FASRC gate (contract C4).

The console is offline-first: every endpoint works without the FASRC SSH
session except the handlers explicitly marked with :func:`requires_fasrc`.
While FASRC is disconnected a marked handler is never entered; the request
gets ``503`` with :data:`FASRC_OFFLINE_PAYLOAD`, which the SPA recognises by
``code == "fasrc_offline"``.

Usage (decorator order does not matter; the flag lives on the function)::

    @app.post("/api/fasrc/cancel")
    @requires_fasrc
    def api_fasrc_cancel(): ...

``tests/test_fasrc_gate.py`` audits every route handler: one that references
the SSH session (``STATE.ssh``, ``.run(``, rsync, ``fetch_one_file`` …) must
be marked or be listed there as knowingly degrading gracefully.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from flask import Flask, jsonify, request

from euclid_polish.web.remote import STATE

FASRC_OFFLINE_PAYLOAD: dict[str, Any] = {
    "ok": False,
    "error": "FASRC not connected",
    "code": "fasrc_offline",
}

_FLAG = "_requires_fasrc"


def requires_fasrc[ViewT: Callable[..., Any]](view: ViewT) -> ViewT:
    """Mark a view as needing the FASRC SSH session (returns ``view``)."""
    setattr(view, _FLAG, True)
    return view


def fasrc_connected() -> bool:
    """True when the shared SSH ControlMaster session is alive."""
    session = STATE.ssh
    return session is not None and bool(session.is_connected())


def register_fasrc_gate(app: Flask) -> None:
    """Refuse marked views with the C4 503 while FASRC is disconnected."""

    @app.before_request
    def _fasrc_gate():
        if request.method == "OPTIONS" or request.endpoint is None:
            return None
        view = app.view_functions.get(request.endpoint)
        if view is None or not getattr(view, _FLAG, False):
            return None
        if fasrc_connected():
            return None
        return jsonify(FASRC_OFFLINE_PAYLOAD), 503


__all__ = [
    "FASRC_OFFLINE_PAYLOAD",
    "fasrc_connected",
    "register_fasrc_gate",
    "requires_fasrc",
]
