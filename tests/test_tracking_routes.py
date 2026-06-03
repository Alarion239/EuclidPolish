"""Route-level tests for the Tracking tab (euclid_polish.web.app)."""

from __future__ import annotations

import os

import pytest

from euclid_polish.config import Config
from euclid_polish.web.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_tracking_page_renders_when_empty(client):
    r = client.get("/tracking")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "Start a campaign" in body


def test_tracking_page_reachable_without_ssh(client, monkeypatch):
    # The gate must NOT redirect /tracking even when SSH is down.
    from euclid_polish.web import remote
    monkeypatch.setattr(remote.STATE, "ssh", None)
    r = client.get("/tracking")
    assert r.status_code == 200


def test_new_then_state_then_save(client):
    r = client.post("/api/tracking/new",
                    data={"title": "Route Run", "description": "via http"})
    assert r.status_code == 200 and r.get_json()["ok"]
    meta = r.get_json()["metadata"]
    assert meta["title"] == "Route Run"

    # Second create is rejected while one is active.
    r2 = client.post("/api/tracking/new", data={"title": "second"})
    assert r2.status_code == 400 and not r2.get_json()["ok"]

    # State reflects the active campaign.
    st = client.get("/api/tracking/state").get_json()
    assert st["active"]["title"] == "Route Run"

    # The page now shows the title + the Save button.
    body = client.get("/tracking").get_data(as_text=True)
    assert "Route Run" in body and "Save &amp; start new" in body

    # Save → archived, no active.
    rs = client.post("/api/tracking/save")
    assert rs.status_code == 200 and rs.get_json()["ok"]
    st2 = client.get("/api/tracking/state").get_json()
    assert st2["active"] is None
    assert any(c["title"] == "Route Run" for c in st2["archived"])


def test_new_requires_title(client):
    r = client.post("/api/tracking/new", data={"title": "   "})
    assert r.status_code == 400
    assert "title" in r.get_json()["error"]


def test_log_append_and_replace(client):
    client.post("/api/tracking/new", data={"title": "notes"})
    r = client.post("/api/tracking/log",
                    data={"text": "first observation", "mode": "append"})
    assert r.status_code == 200
    assert "first observation" in r.get_json()["log_md"]

    r = client.post("/api/tracking/log",
                    data={"text": "# wiped\n", "mode": "replace"})
    assert r.get_json()["log_md"] == "# wiped\n"

    # empty append is rejected
    r = client.post("/api/tracking/log", data={"text": "  ", "mode": "append"})
    assert r.status_code == 400


def test_backup_fits_route(client, tmp_path, monkeypatch):
    # Make tmp_path an allowed root, then back up a file living under it.
    monkeypatch.setattr(Config, "DEFAULT_OUTPUT_DIR", str(tmp_path))
    src = tmp_path / "result.fits"
    src.write_bytes(b"SIMPLE = T" + b" " * 80)

    client.post("/api/tracking/new", data={"title": "bk"})
    r = client.post("/api/tracking/backup",
                    data={"kind": "fits", "path": str(src),
                          "comment": "the SR output", "name": "sr"})
    assert r.status_code == 200, r.get_data(as_text=True)
    j = r.get_json()
    assert j["ok"]
    assert j["record"]["kind"] == "fits"
    assert j["record"]["comment"] == "the SR output"
    # the synthetic null-SSH stub makes sync a no-op success
    assert "sync" in j

    st = client.get("/api/tracking/state").get_json()
    assert len(st["backups"]["fits"]) == 1


def test_backup_rejects_path_outside_roots(client, tmp_path, monkeypatch):
    # Allowed root is a *different* dir, so a file elsewhere is 403.
    monkeypatch.setattr(Config, "DEFAULT_OUTPUT_DIR", str(tmp_path / "allowed"))
    os.makedirs(Config.DEFAULT_OUTPUT_DIR, exist_ok=True)
    outside = tmp_path / "outside.fits"
    outside.write_bytes(b"x")
    client.post("/api/tracking/new", data={"title": "bk"})
    r = client.post("/api/tracking/backup",
                    data={"kind": "fits", "path": str(outside)})
    assert r.status_code == 403


def test_backup_unknown_kind(client):
    client.post("/api/tracking/new", data={"title": "bk"})
    r = client.post("/api/tracking/backup", data={"kind": "bogus"})
    assert r.status_code == 400
