"""Serving of the built SPA: content-hashed Vite chunks (``ASSETS_PREFIX``)
are cached as immutable and gzip-compressed on request; the shell
(``index.html``) is always revalidated (``no-cache``)."""

from __future__ import annotations

import gzip

import pytest

from euclid_polish.web import static_assets
from euclid_polish.web.app import create_app

#: The content-hashed chunk under test (the URL prefix comes from the module:
#: pytest never reads the real bundle).
JS_URL = static_assets.ASSETS_PREFIX + "index-3tEk6LPN.js"
_JS = ("export const x = " + "'the quick brown fox jumps over the lazy dog ' + " * 400
       + "'';\n").encode()


@pytest.fixture
def client(tmp_path):
    static = tmp_path / "static"
    (static / "dist" / "assets").mkdir(parents=True)
    (static / "dist" / "assets" / "index-3tEk6LPN.js").write_bytes(_JS)
    (static / "dist" / "assets" / "logo-AbCdEf12.png").write_bytes(b"\x89PNG" + b"\0" * 4000)
    (static / "dist" / "assets" / "tiny-Q1w2E3r4.css").write_bytes(b"a{}")
    (static / "dist" / "robots.txt").write_bytes(b"User-agent: *\n" * 200)
    index = static / "dist" / "index.html"
    index.write_text('<!doctype html><div id="root"></div>', encoding="utf-8")
    app = create_app()
    app.config["TESTING"] = True
    app.static_folder = str(static)
    app.config["SPA_INDEX_PATH"] = str(index)
    static_assets.clear_cache()
    with app.test_client() as c:
        yield c


def test_hashed_assets_are_immutable(client):
    response = client.get(JS_URL)
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "public, max-age=31536000, immutable"


def test_an_unhashed_static_file_keeps_revalidating(client):
    response = client.get(static_assets.DIST_PREFIX + "robots.txt")
    assert response.status_code == 200
    assert "immutable" not in response.headers.get("Cache-Control", "")


def test_the_spa_shell_is_never_cached_as_immutable(client):
    response = client.get("/sky/atlas")
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-cache"


def test_assets_are_gzipped_when_the_client_accepts_it(client):
    response = client.get(JS_URL,
                          headers={"Accept-Encoding": "gzip, deflate, br"})
    assert response.status_code == 200
    assert response.headers["Content-Encoding"] == "gzip"
    assert "Accept-Encoding" in response.headers["Vary"]
    body = response.get_data()
    assert int(response.headers["Content-Length"]) == len(body) < len(_JS) / 4
    assert gzip.decompress(body) == _JS
    assert response.headers["Cache-Control"] == "public, max-age=31536000, immutable"
    assert response.headers["ETag"].endswith('-gzip"')


def test_no_gzip_without_accept_encoding(client):
    response = client.get(JS_URL,
                          headers={"Accept-Encoding": "identity"})
    assert "Content-Encoding" not in response.headers
    assert response.get_data() == _JS
    assert "Accept-Encoding" in response.headers.get("Vary", "")


@pytest.mark.parametrize("name", ["logo-AbCdEf12.png", "tiny-Q1w2E3r4.css"])
def test_binary_and_tiny_files_are_sent_as_is(client, name):
    response = client.get(static_assets.ASSETS_PREFIX + name, headers={"Accept-Encoding": "gzip"})
    assert response.status_code == 200
    assert "Content-Encoding" not in response.headers


def test_a_range_request_is_never_compressed(client):
    response = client.get(JS_URL,
                          headers={"Accept-Encoding": "gzip", "Range": "bytes=0-99"})
    assert response.status_code == 206
    assert "Content-Encoding" not in response.headers
    assert response.get_data() == _JS[:100]


def test_a_revalidation_of_the_gzip_variant_is_a_304(client):
    first = client.get(JS_URL,
                       headers={"Accept-Encoding": "gzip"})
    again = client.get(JS_URL,
                       headers={"Accept-Encoding": "gzip",
                                "If-None-Match": first.headers["ETag"]})
    assert again.status_code == 304
    assert again.get_data() == b""
