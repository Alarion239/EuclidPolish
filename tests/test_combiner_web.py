"""Web contract for the sole incremental raw combiner."""

import pytest

from euclid_polish.web.app import create_app


class _Cap:
    def tick(self, *_args, **_kwargs):
        pass


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_combiner_fit_defaults_to_incremental_raw_k128(client, monkeypatch):
    from euclid_polish.web.routes import ensemble as routes

    seen = {}

    def fake_job(_cap, **kwargs):
        seen.update(kwargs)

    def fake_spawn(_description, target):
        target(_Cap())
        return "raw-incremental"

    monkeypatch.setattr(routes, "job_combiner_fit", fake_job)
    monkeypatch.setattr(routes.REGISTRY, "spawn", fake_spawn)
    response = client.post(
        "/ensemble/combiner/fit", data={"mode": "starless"})

    assert response.status_code == 200
    assert response.get_json()["job_id"] == "raw-incremental"
    assert seen["model_kind"] == "raw_incremental_minmeanmax_rbf"
    assert seen["n_kernels"] == 128
    assert seen["starless"] is True


@pytest.mark.parametrize("retired", [
    "rbf_gate",
    "stats_rbf_gate",
    "minmax_rbf_gate",
    "stacked_rbf_gate",
    "shared_rbf_gate",
    "shared_zero_rbf_gate",
    "shared_minmeanstdmax_rbf_gate",
    "identity_soft_selector",
])
def test_combiner_fit_rejects_retired_models(client, retired):
    response = client.post("/ensemble/combiner/fit", data={
        "mode": "starfull", "model_kind": retired,
    })
    assert response.status_code == 400


def test_combiner_json_defaults_to_incremental_raw(client, monkeypatch):
    from euclid_polish.web.routes import ensemble as routes

    seen = {}

    def fake_payload(starless, model_kind):
        seen.update(starless=starless, model_kind=model_kind)
        return None

    monkeypatch.setattr(routes, "compute_combiner_payload", fake_payload)
    response = client.get("/ensemble/combiner.json?mode=starfull")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["available"] is False
    assert payload["kind"] == "raw_incremental_minmeanmax_rbf"
    assert seen == {
        "starless": False,
        "model_kind": "raw_incremental_minmeanmax_rbf",
    }


def test_removed_combined_combiner_routes_are_404(client):
    assert client.get("/ensemble/combined-combiner.json").status_code == 404
    assert client.post("/ensemble/combined-combiner/fit").status_code == 404
