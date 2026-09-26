"""FASRC connection state (contract C4): ``last_error`` and offline settings.

``GET /api/fasrc/status`` explains *why* FASRC is disconnected — the startup
auto-connect error or the last failed connect — and the connection settings
(``/api/fasrc/config``) plus ``/api/connection/retry`` work offline.
"""

from __future__ import annotations

import pytest

from euclid_polish.web import app as web_app
from euclid_polish.web import fasrc_config, remote
from euclid_polish.web.fasrc_config import FasrcConfig
from euclid_polish.web.remote import SSHError


class _FakeSession:
    """Stands in for SSHSession; never touches ssh."""

    fail_with: str | None = None
    instances: list[_FakeSession] = []

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.connected = False
        _FakeSession.instances.append(self)

    def connect(self) -> None:
        if _FakeSession.fail_with:
            raise SSHError(_FakeSession.fail_with)
        self.connected = True

    def is_connected(self) -> bool:
        return self.connected

    def disconnect(self) -> None:
        self.connected = False

    def run(self, cmd, timeout=None, binary=False):
        return (0, "", "")


@pytest.fixture
def fake_ssh(monkeypatch):
    cfg = FasrcConfig(ssh_user="astro", ssh_host="cluster.example",
                      control_socket="/tmp/test-connection.sock")
    monkeypatch.setattr(fasrc_config, "load", lambda: cfg)
    monkeypatch.setattr(remote, "SSHSession", _FakeSession)
    monkeypatch.setattr(remote.STATE, "ssh", None)
    monkeypatch.setattr(remote.STATE, "connected_at", None)
    monkeypatch.setattr(remote.STATE, "last_error", None)
    _FakeSession.fail_with = None
    _FakeSession.instances = []
    yield _FakeSession
    _FakeSession.fail_with = None


@pytest.fixture
def client(fake_ssh):
    app = web_app.create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_public_status_reports_last_error(monkeypatch):
    monkeypatch.setattr(remote.STATE, "ssh", None)
    monkeypatch.setattr(remote.STATE, "last_error", "Permission denied (publickey).")
    status = remote.STATE.public_status()
    assert status["ssh_connected"] is False
    assert status["last_error"] == "Permission denied (publickey)."


def test_startup_error_is_reported_by_status(client):
    """Tests run with the auto-connect kill switch, which is itself the
    startup "error" the status explains."""
    status = client.get("/api/fasrc/status").get_json()
    assert status["ssh_connected"] is False
    assert "auto-connect disabled" in status["last_error"]


def test_failed_startup_connect_is_kept_as_last_error(fake_ssh, monkeypatch):
    monkeypatch.setenv("EUCLID_POLISH_DISABLE_AUTO_SSH", "0")
    fake_ssh.fail_with = "Permission denied (publickey)."

    app = web_app.create_app()

    assert remote.STATE.ssh is None
    assert "Permission denied" in remote.STATE.last_error
    status = app.test_client().get("/api/fasrc/status").get_json()
    assert "Permission denied" in status["last_error"]


def test_startup_error_has_a_single_source_of_truth(fake_ssh, monkeypatch):
    """``STATE.last_error`` is the only record of why FASRC is down; the app
    keeps no second copy in ``app.config`` that could drift from it."""
    monkeypatch.setenv("EUCLID_POLISH_DISABLE_AUTO_SSH", "0")
    fake_ssh.fail_with = "Permission denied (publickey)."

    app = web_app.create_app()

    assert "FASRC_STARTUP_ERROR" not in app.config


def test_legacy_connection_error_form_is_gone(client):
    """The classic connect form was deleted: a GET moves to the Settings ›
    Connections workspace (C1) and a POST no longer reaches any handler, so
    ``POST /api/connection/retry`` / ``POST /api/fasrc/connect`` are the only
    connect actions."""
    get = client.get("/connection-error")
    assert get.status_code == 308
    assert get.headers["Location"] == "/settings/connections"
    assert client.post("/connection-error", data={}).status_code == 404


def test_successful_startup_connect_clears_the_error(fake_ssh, monkeypatch):
    monkeypatch.setenv("EUCLID_POLISH_DISABLE_AUTO_SSH", "0")
    monkeypatch.setattr(remote.STATE, "last_error", "stale")
    monkeypatch.setattr(web_app.fasrc_jobs, "sync_pending_on_connect",
                        lambda _ssh: None)

    app = web_app.create_app()

    assert isinstance(remote.STATE.ssh, fake_ssh)
    assert remote.STATE.last_error is None
    status = app.test_client().get("/api/fasrc/status").get_json()
    assert status["last_error"] is None
    assert status["ssh_connected"] is True


def test_connect_failure_then_success_updates_last_error(client, fake_ssh, monkeypatch):
    monkeypatch.setattr(web_app.fasrc_jobs, "sync_pending_on_connect",
                        lambda _ssh: None)
    fake_ssh.fail_with = "ssh: connect to host cluster.example: timed out"

    failed = client.post("/api/fasrc/connect")
    assert failed.status_code == 400
    assert "timed out" in failed.get_json()["error"]
    status = client.get("/api/fasrc/status").get_json()
    assert status["ssh_connected"] is False
    assert "timed out" in status["last_error"]

    fake_ssh.fail_with = None
    ok = client.post("/api/fasrc/connect")
    assert ok.status_code == 200
    assert ok.get_json()["status"]["last_error"] is None
    assert client.get("/api/fasrc/status").get_json()["ssh_connected"] is True


def test_connect_without_a_user_explains_itself(client, monkeypatch):
    monkeypatch.setattr(fasrc_config, "load", lambda: FasrcConfig(ssh_user=""))
    response = client.post("/api/fasrc/connect")
    assert response.status_code == 400
    assert "ssh_user" in response.get_json()["error"]
    assert "ssh_user" in client.get("/api/fasrc/status").get_json()["last_error"]


def test_retry_reports_and_records_its_own_error_offline(client):
    response = client.post("/api/connection/retry")
    assert response.status_code == 502
    body = response.get_json()
    assert body["ok"] is False
    assert "auto-connect disabled" in body["error"]
    assert client.get("/api/fasrc/status").get_json()["last_error"] == body["error"]


def test_fasrc_settings_are_reachable_offline(client):
    response = client.get("/api/fasrc/config")
    assert response.status_code == 200
    assert response.get_json()["ssh_user"] == "astro"


def test_manual_disconnect_leaves_no_error(client, fake_ssh, monkeypatch):
    monkeypatch.setattr(web_app.fasrc_jobs, "sync_pending_on_connect",
                        lambda _ssh: None)
    assert client.post("/api/fasrc/connect").status_code == 200
    status = client.post("/api/fasrc/disconnect").get_json()["status"]
    assert status["ssh_connected"] is False
    assert status["last_error"] is None


def test_ensure_ssh_connected_records_a_failure(fake_ssh):
    fake_ssh.fail_with = "Host key verification failed."
    with pytest.raises(SSHError, match="Host key"):
        remote.ensure_ssh_connected()
    assert remote.STATE.ssh is None
    assert "Host key" in remote.STATE.last_error


def test_ensure_ssh_connected_reuses_a_live_session(fake_ssh):
    session = remote.ensure_ssh_connected()
    assert remote.ensure_ssh_connected() is session
    assert len(fake_ssh.instances) == 1
    assert remote.STATE.last_error is None
