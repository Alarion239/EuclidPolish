"""JSON error responses by path prefix (``euclid_polish.web.errors``).

Flask keeps ONE handler per exception class, so route modules must not each
register ``@app.errorhandler(HTTPException)`` — the last one would silently
replace the others. They share one path-dispatching handler instead
(``errors.json_errors_for(app, prefix)``); ``/viewer/`` is registered (C6).
"""

from __future__ import annotations

import pytest
from werkzeug.exceptions import HTTPException

from euclid_polish.web import errors
from euclid_polish.web.app import create_app
from euclid_polish.web.helpers import viewer_data


@pytest.fixture
def app():
    application = create_app()
    application.config["TESTING"] = True
    return application


def _http_exception_handlers(app) -> list:
    return [handler
            for by_code in app.error_handler_spec[None].values()
            for exc_class, handler in by_code.items()
            if issubclass(exc_class, HTTPException)]


def test_one_shared_http_exception_handler_for_the_whole_app(app):
    """Any module registering its own HTTPException handler would break the
    others' JSON errors — there must be exactly the shared one."""
    assert _http_exception_handlers(app) == [errors.json_http_error]
    assert "/viewer/" in errors.json_prefixes(app)


def test_viewer_routing_errors_are_json(app):
    client = app.test_client()
    not_allowed = client.post("/viewer/meta/sky")
    assert not_allowed.status_code == 405
    assert not_allowed.is_json and not_allowed.get_json()["error"]
    missing = client.get("/viewer/no/such/route")
    assert missing.status_code == 404
    assert missing.is_json and "error" in missing.get_json()


def test_a_viewer_loader_crash_is_a_json_500(app, monkeypatch):
    app.config["TESTING"] = False           # let Flask turn the crash into a 500
    app.config["PROPAGATE_EXCEPTIONS"] = False

    def boom(_collection, _params):
        raise RuntimeError("loader exploded")

    monkeypatch.setattr(viewer_data, "get_meta", boom)
    response = app.test_client().get("/viewer/meta/sky")
    assert response.status_code == 500
    assert response.is_json and "error" in response.get_json()


def test_other_paths_keep_flask_default_errors(app):
    response = app.test_client().get("/api/definitely-not-a-route")
    assert response.status_code == 404
    assert not response.is_json


def test_a_second_prefix_joins_instead_of_replacing(app):
    """A later module (e.g. real-results JSON errors) adds its prefix to the
    same handler; ``/viewer/`` keeps answering JSON."""
    errors.json_errors_for(app, "/api/example-json/")
    errors.json_errors_for(app, "/api/example-json/")          # idempotent
    assert _http_exception_handlers(app) == [errors.json_http_error]
    assert errors.json_prefixes(app).count("/api/example-json/") == 1
    client = app.test_client()
    other = client.get("/api/example-json/missing")
    assert other.status_code == 404 and other.is_json
    viewer = client.post("/viewer/meta/sky")
    assert viewer.status_code == 405 and viewer.is_json
