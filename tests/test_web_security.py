from __future__ import annotations

import pytest
from flask import Flask, jsonify

from euclid_polish.web.security import (
    TRUSTED_HOSTS,
    register_host_allowlist,
    register_mutation_guard,
    validate_bind_host,
)


@pytest.fixture
def client_and_calls():
    app = Flask(__name__)
    app.config.update(TESTING=True)
    register_mutation_guard(app)
    calls = []

    @app.post("/mutate")
    def mutate():
        calls.append("called")
        return jsonify({"ok": True})

    return app.test_client(), calls


def test_cross_origin_post_cannot_reach_mutation(client_and_calls):
    client, calls = client_and_calls
    response = client.post("/mutate", headers={"Origin": "https://attacker.example"})

    assert response.status_code == 403
    assert calls == []


def test_same_origin_post_reaches_mutation(client_and_calls):
    client, calls = client_and_calls
    response = client.post("/mutate", headers={"Origin": "http://localhost"})

    assert response.status_code == 200
    assert calls == ["called"]


def test_cross_site_fetch_metadata_is_rejected(client_and_calls):
    client, calls = client_and_calls
    response = client.post("/mutate", headers={"Sec-Fetch-Site": "cross-site"})

    assert response.status_code == 403
    assert calls == []


def test_headerless_local_client_remains_supported(client_and_calls):
    client, calls = client_and_calls
    response = client.post("/mutate")

    assert response.status_code == 200
    assert calls == ["called"]


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_loopback_bind_hosts_are_allowed(host):
    assert validate_bind_host(host) == host


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "127.0.0.2", "192.168.1.20", "example.test"],
)
def test_non_loopback_bind_hosts_are_rejected(host):
    with pytest.raises(ValueError, match="loopback"):
        validate_bind_host(host)


# ---------------------------------------------------------------------------
# Host allowlist (DNS-rebinding guard)
# ---------------------------------------------------------------------------

@pytest.fixture
def host_guarded():
    app = Flask(__name__)
    app.config.update(TESTING=True)
    register_host_allowlist(app)
    register_mutation_guard(app)
    calls = []

    @app.before_request
    def later_hook_that_would_answer():
        # A later hook that returns a response (like the SPA shell) must not
        # be able to bypass the Host check.
        if calls == ["shortcut"]:
            return "shortcut", 200
        return None

    @app.get("/read")
    def read():
        calls.append("read")
        return jsonify({"ok": True})

    @app.post("/api/mutate")
    def mutate():
        calls.append("mutate")
        return jsonify({"ok": True})

    return app, calls


def test_trusted_hosts_are_the_loopback_names():
    assert list(TRUSTED_HOSTS) == ["localhost", "127.0.0.1", "[::1]", "::1"]


def test_register_host_allowlist_sets_the_flask_config(host_guarded):
    app, _calls = host_guarded
    assert app.config["TRUSTED_HOSTS"] == list(TRUSTED_HOSTS)


# Browsers always send the host lower-cased (Werkzeug compares exactly).
@pytest.mark.parametrize("host", [
    "localhost", "localhost:9777", "127.0.0.1", "127.0.0.1:8765",
    "[::1]:9777",
])
def test_loopback_host_headers_are_served(host_guarded, host):
    app, calls = host_guarded
    response = app.test_client().get("/read", headers={"Host": host})
    assert response.status_code == 200
    assert calls == ["read"]


@pytest.mark.parametrize("host", [
    "evil.example", "evil.example:9777", "localhost.evil.example",
    "127.0.0.1.nip.io:9777", "192.168.1.20:9777",
])
def test_foreign_host_headers_are_rejected_before_any_handler(host_guarded, host):
    app, calls = host_guarded
    client = app.test_client()
    read = client.get("/read", headers={"Host": host})
    mutate = client.post("/api/mutate", headers={"Host": host})
    assert read.status_code == 400
    assert mutate.status_code == 400
    assert mutate.get_json() == {
        "ok": False, "error": "untrusted Host header", "code": "untrusted_host",
    }
    assert calls == []


def test_later_hooks_cannot_bypass_the_host_check(host_guarded):
    app, calls = host_guarded
    calls.append("shortcut")
    response = app.test_client().get("/read", headers={"Host": "evil.example"})
    assert response.status_code == 400
