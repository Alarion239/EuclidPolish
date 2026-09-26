"""The route-module registry: ``routes.MODULES`` drives ``create_app``.

Adding a route group is one line in ``euclid_polish/web/routes/__init__.py``;
these tests keep that list complete and make sure the app factory registers
exactly those modules.
"""

from __future__ import annotations

import types
from pathlib import Path

from flask import Flask

from euclid_polish.web import app as web_app
from euclid_polish.web import routes

ROUTES_DIR = Path(routes.__file__).parent


def test_modules_is_a_tuple_of_route_modules():
    assert isinstance(routes.MODULES, tuple)
    assert routes.MODULES
    for module in routes.MODULES:
        assert isinstance(module, types.ModuleType)
        assert module.__name__.startswith("euclid_polish.web.routes.")
        assert callable(module.register)


def test_every_route_file_is_registered_exactly_once():
    on_disk = {
        path.stem for path in ROUTES_DIR.glob("*.py")
        if path.stem != "__init__"
    }
    names = [module.__name__.rsplit(".", 1)[-1] for module in routes.MODULES]
    assert len(names) == len(set(names))
    assert set(names) == on_disk


def test_create_app_registers_every_module(monkeypatch):
    calls = []

    def fake(name):
        return types.SimpleNamespace(
            __name__=f"euclid_polish.web.routes.{name}",
            register=lambda app: calls.append((name, app)),
        )

    fakes = (fake("alpha"), fake("beta"))
    monkeypatch.setattr(web_app, "ROUTE_MODULES", fakes)

    app = web_app.create_app()

    assert isinstance(app, Flask)
    assert [name for name, _ in calls] == ["alpha", "beta"]
    assert all(registered is app for _, registered in calls)


def test_real_app_exposes_routes_from_each_module():
    app = web_app.create_app()
    endpoints = set(app.view_functions)
    # One well-known endpoint per module family (spot check that the loop
    # really ran over the whole tuple).
    for endpoint in ("api_jobs", "api_fasrc_status", "api_noise",
                     "viewer_meta", "api_git_status", "api_tracking_state"):
        assert endpoint in endpoints
