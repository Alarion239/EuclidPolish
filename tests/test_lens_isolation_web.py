from __future__ import annotations

import os

os.environ["EUCLID_POLISH_DISABLE_AUTO_SSH"] = "1"

from euclid_polish.web.app import create_app


def test_classic_page_and_status_are_registered():
    app = create_app()
    client = app.test_client()
    page = client.get("/lens-isolation")
    assert page.status_code == 200
    body = page.get_data(as_text=True)
    assert "lens_isolation_generate" in body
    assert "lens_isolation_train" in body
    assert "lens_isolation_evaluate" in body
    assert "Pure TNG" in body
    assert "20 lenses / arcmin²" in body
    assert "random, block-aligned crops" in body
    assert "balanced" not in body.lower()
    status = client.get("/api/lens-isolation/status")
    assert status.status_code == 200
    assert "experiments/lens_isolation" in status.get_json()["root"]


def test_sync_defaults_to_evaluation_only_when_offline(monkeypatch):
    from euclid_polish.web.remote import STATE

    monkeypatch.setattr(STATE, "ssh", None)
    app = create_app()
    response = app.test_client().post("/api/lens-isolation/sync")
    assert response.status_code == 400
    assert response.get_json()["error"] == "not connected"
