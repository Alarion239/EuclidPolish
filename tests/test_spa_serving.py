"""Flask serves the SPA shell for manifest page paths and 308s legacy URLs.

Page paths come from ``euclid_polish/web/spa_routes.json`` (contract C1);
data endpoints that share a prefix with a page (``/ensemble/status.json``,
``/inspect/preview.png``) must keep reaching their own handlers.
"""

from __future__ import annotations

import json
import re

import pytest

from euclid_polish.web.app import create_app
from euclid_polish.web.spa_routes import MANIFEST_PATH, page_paths

MANIFEST = json.loads(MANIFEST_PATH.read_text())


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _assert_shell(response):
    assert response.status_code == 200, response.get_data(as_text=True)[:400]
    assert response.mimetype == "text/html"
    assert 'id="root"' in response.get_data(as_text=True)


@pytest.mark.parametrize("path", sorted(page_paths()))
def test_every_manifest_page_serves_the_shell(client, path):
    _assert_shell(client.get(path))


@pytest.mark.parametrize("path", ["/sky/atlas", "/", "/ensemble/starless/train",
                                  "/settings/about", "/inspect"])
def test_new_workspace_urls_serve_the_shell(client, path):
    _assert_shell(client.get(path))


def test_page_path_with_query_and_trailing_slash_serves_the_shell(client):
    _assert_shell(client.get("/sky/atlas/?ra=150.1&dec=2.2"))
    _assert_shell(client.get("/inspect?fits=data/foo.fits"))


def test_head_on_a_page_path_serves_the_shell_headers(client):
    response = client.head("/sky/atlas")
    assert response.status_code == 200
    assert response.mimetype == "text/html"


def test_legacy_config_url_redirects_permanently(client):
    response = client.get("/config")
    assert response.status_code == 308
    assert response.headers["Location"] == "/settings/config"


@pytest.mark.parametrize("source,target", sorted(MANIFEST["redirects"].items()))
def test_every_legacy_url_redirects_with_its_query(client, source, target):
    response = client.get(f"{source}?a=1&b=x%2Fy")
    assert response.status_code == 308
    assert response.headers["Location"] == f"{target}?a=1&b=x%2Fy"


def test_head_follows_the_same_redirect(client):
    response = client.head("/noise")
    assert response.status_code == 308
    assert response.headers["Location"] == "/realism/noise"


def test_app_prefix_redirects_to_the_bare_route(client):
    response = client.get("/app/sky/atlas?x=1")
    assert response.status_code == 308
    assert response.headers["Location"] == "/sky/atlas?x=1"


@pytest.mark.parametrize("path,location", [
    ("/app//evil.example", "/evil.example"),
    ("/app//evil.example/x?y=1", "/evil.example/x?y=1"),
    ("/app/%5Cevil.example", "/evil.example"),
    ("/app/%2F%2Fevil.example", "/evil.example"),
    ("/app/%5C%2Fevil.example", "/evil.example"),
])
def test_app_prefix_redirect_is_not_an_open_redirect(client, path, location):
    """``Location: //evil.example`` (or ``/\\evil.example``) is protocol-
    relative to a browser and would leave localhost; the redirect must stay
    on this host."""
    response = client.get(path)
    assert response.status_code == 308
    assert response.headers["Location"] == location


def test_redirect_chain_ends_on_the_shell(client):
    response = client.get("/app/inference", follow_redirects=True)
    _assert_shell(response)
    assert response.request.path == "/sky/results"


def test_data_endpoint_under_a_page_prefix_stays_json(client):
    response = client.get("/ensemble/status.json")
    assert response.status_code == 200
    assert response.is_json


def test_unknown_tab_is_not_served_as_a_page(client):
    assert client.get("/ensemble/foo").status_code == 404
    assert client.get("/sky/unknown").status_code == 404


def test_posts_to_page_paths_are_not_given_the_shell(client):
    response = client.post("/sky/atlas")
    assert response.status_code in (404, 405)
    assert 'id="root"' not in response.get_data(as_text=True)


def test_legacy_post_is_not_redirected(client):
    """Only GET/HEAD navigations move; a stray POST must not be converted
    into a 308 that browsers would replay against the new page."""
    for legacy in ("/config", "/noise", "/connection-error"):
        response = client.post(legacy)
        assert response.status_code != 308, legacy
        assert "Location" not in response.headers or (
            response.headers["Location"] != MANIFEST["redirects"][legacy]
        ), legacy


def test_missing_build_returns_a_build_hint(tmp_path):
    app = create_app()
    app.config["SPA_INDEX_PATH"] = str(tmp_path / "missing" / "index.html")
    response = app.test_client().get("/sky/atlas")
    assert response.status_code == 503
    assert "npm" in response.get_data(as_text=True)
    assert response.mimetype == "text/plain"


def test_shell_references_only_existing_bundle_files(client):
    """The committed ``index.html`` must point at files that exist in the
    committed build (checked by status only; the bundle's contents are the
    frontend's business)."""
    body = client.get("/").get_data(as_text=True)
    refs = re.findall(r'(?:src|href)="(/static/dist/[^"]+)"', body)
    assert refs, "the shell references no bundle files"
    for ref in refs:
        assert client.get(ref).status_code == 200, ref
