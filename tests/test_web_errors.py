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
    response = app.test_client().get("/definitely-not-a-route.txt")
    assert response.status_code == 404
    assert not response.is_json


def test_the_whole_api_prefix_answers_json_errors(app):
    """Every ``/api/*`` error is ``{ok: false, error}`` with its status — the
    SPA's client shows the message instead of "HTTP 404 NOT FOUND"."""
    assert "/api/" in errors.json_prefixes(app)
    client = app.test_client()
    for method, path, status in [
        ("get", "/api/definitely-not-a-route", 404),
        ("get", "/api/cutouts/NOPE/list.json", 404),
        ("get", "/api/tng/nope", 404),
        ("get", "/api/git/nope", 404),
        ("get", "/api/tracking/backup", 405),
        ("get", "/api/fasrc/queue/clear", 405),
        ("get", "/api/tracking/campaign/..%2F..%2Fetc", 404),
    ]:
        response = getattr(client, method)(path)
        assert response.status_code == status, path
        assert response.is_json, path
        body = response.get_json()
        assert body["ok"] is False and isinstance(body["error"], str) and body["error"], path


def test_json_error_bodies_carry_ok_false(app):
    body = app.test_client().post("/viewer/meta/sky").get_json()
    assert body["ok"] is False and body["error"]


@pytest.mark.parametrize("path", [
    "/api/provenance/records?limit=abc",
    "/api/git/log?limit=abc",
    "/api/git/log?skip=1.5",
    "/api/tracking/jobs?offset=x",
])
def test_a_malformed_integer_argument_is_a_400(app, path):
    response = app.test_client().get(path)
    assert response.status_code == 400, response.get_data(as_text=True)[:200]
    assert response.is_json and "integer" in response.get_json()["error"]


def test_the_shared_int_arg_clamps_or_refuses_out_of_range(app):
    with app.test_request_context("/?n=5000&m=-3&k=7"):
        assert errors.int_arg("n", 1, lo=1, hi=100, clamp=True) == 100
        assert errors.int_arg("m", 1, lo=0, clamp=True) == 0
        assert errors.int_arg("missing", 42) == 42
        with pytest.raises(HTTPException) as refused:
            errors.int_arg("k", 1, lo=0, hi=5)
        assert refused.value.code == 400


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
