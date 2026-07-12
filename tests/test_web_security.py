from __future__ import annotations

import pytest
from flask import Flask, jsonify

from euclid_polish.web.security import register_mutation_guard, validate_bind_host


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
