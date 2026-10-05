"""``/api/sky/generate-sr`` refusals name places that exist in the console.

Members are trained in Models › Train and pulled from FASRC in
Models › Members; ``/ensemble`` only redirects to the Leaderboard.
No model, no records on disk: the record and member probes are stubbed.
"""

from __future__ import annotations

import pytest

from euclid_polish.web.app import create_app
from euclid_polish.web.routes import views as views_routes


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_generate_sr_without_members_points_at_models_train_and_members(
        client, monkeypatch, tmp_path):
    monkeypatch.setattr(views_routes, "_sky_records_local_dir", lambda: str(tmp_path))
    monkeypatch.setattr(views_routes.sky_records, "present_subsets", lambda _d: ["test"])
    monkeypatch.setattr(views_routes.sky_records, "checkpoint_present", lambda *_a: False)

    r = client.post("/api/sky/generate-sr")

    assert r.status_code == 400
    error = r.get_json()["error"]
    assert error.startswith("no active ensemble members")
    assert "Models › Train" in error and "Models › Members" in error
    assert "/ensemble" not in error
