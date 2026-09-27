"""The assembled app enforces the Host allowlist ahead of every other hook,
and state-changing FASRC actions are POST-only."""

from __future__ import annotations

import pytest

from euclid_polish.web.app import create_app
from euclid_polish.web.security import TRUSTED_HOSTS


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_app_trusts_only_loopback_hosts(client):
    assert client.application.config["TRUSTED_HOSTS"] == list(TRUSTED_HOSTS)


@pytest.mark.parametrize("path", ["/", "/sky/atlas", "/api/jobs", "/config",
                                  "/api/fasrc/status"])
def test_rebinding_host_gets_nothing(client, path):
    response = client.get(path, headers={"Host": "evil.example:9777"})
    assert response.status_code == 400
    assert 'id="root"' not in response.get_data(as_text=True)


def test_rebinding_host_cannot_post(client):
    response = client.post("/api/connection/retry",
                           headers={"Host": "evil.example"})
    assert response.status_code == 400
    assert response.get_json()["code"] == "untrusted_host"


@pytest.mark.parametrize("host", ["localhost:9777", "127.0.0.1:8765", "[::1]:9777"])
def test_loopback_hosts_get_the_shell(client, host):
    response = client.get("/sky/atlas", headers={"Host": host})
    assert response.status_code == 200
    assert 'id="root"' in response.get_data(as_text=True)


def test_env_update_is_no_longer_a_get(client):
    """``mamba env update`` on the cluster is a mutation: a cross-site
    ``<img src>`` must not be able to trigger it."""
    assert client.get("/api/fasrc/env-update").status_code == 405


# --------------------------------------------------------------------------- #
# anti-framing / nosniff headers (clickjacking would otherwise bypass the
# same-origin mutation guard: a POST made inside a hostile frame is
# same-origin)
# --------------------------------------------------------------------------- #

_SECURITY_HEADERS = {
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": "frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
}


@pytest.mark.parametrize("method,path", [
    ("get", "/sky/atlas"),                 # the SPA shell
    ("get", "/api/jobs"),                  # JSON
    ("get", "/api/definitely-not-a-route"),  # an error page
    ("post", "/api/connection/retry"),     # a mutation
])
def test_every_response_forbids_framing_and_sniffing(client, method, path):
    response = getattr(client, method)(path)
    for name, value in _SECURITY_HEADERS.items():
        assert response.headers.get(name) == value, (path, name)


def test_a_refused_host_still_carries_the_security_headers(client):
    response = client.get("/sky/atlas", headers={"Host": "evil.example"})
    assert response.status_code == 400
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Content-Security-Policy"] == "frame-ancestors 'none'"
