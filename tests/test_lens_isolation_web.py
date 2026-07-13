from __future__ import annotations

import os
from pathlib import Path

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


def test_react_generation_card_uses_only_fasrc_cpu_resources():
    page = Path("euclid_polish/web/frontend/src/pages/LensIsolation.tsx").read_text(encoding="utf-8")

    assert 'label="workers"' not in page
    assert "setWorkers" not in page


def test_classic_generation_card_uses_only_fasrc_cpu_resources():
    page = Path("euclid_polish/web/static/fasrc_step_card.js").read_text(encoding="utf-8")
    card = page.split("case 'lens_isolation_generate':", 1)[1].split(
        "case 'lens_isolation_train':", 1
    )[0]

    assert 'name="workers"' not in card
