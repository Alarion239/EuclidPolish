"""The combiner metric block, and the removed legacy combiner routes (404)."""

import numpy as np
import pytest

from euclid_polish.web.app import create_app
from euclid_polish.web.helpers.ensemble_viz import _CombinerMetricAcc
from euclid_polish.web.routes import ensemble as routes


def test_combiner_metric_block_reports_l1_and_psnr():
    hr = np.zeros((2, 2), np.float32)
    members = np.stack([
        np.full_like(hr, 50.0),
        np.full_like(hr, 200.0),
    ])
    acc = _CombinerMetricAcc()
    acc.add(hr, np.full_like(hr, 100.0), members, np.full_like(hr, 25.0))
    block = acc.block(["near", "far"])

    assert block is not None
    assert block["available"] is True
    assert block["asinh_l1"] < block["ensemble_mean_asinh_l1"]
    assert block["best_member_l1_label"] == "near"
    assert block["best_member_label"] == "near"
    assert block["psnr"] > block["ensemble_mean_psnr"]


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_removed_combined_combiner_routes_are_404(client):
    assert client.get("/ensemble/combined-combiner.json").status_code == 404
    assert client.post("/ensemble/combined-combiner/fit").status_code == 404


def test_removed_rbf_combiner_routes_are_404(client, monkeypatch):
    """The legacy in-place RBF fit and its Combiner-card dataset are gone (no
    SPA caller): the variant registry (/ensemble/combiners.json, …/fit,
    …/promote) is the only combiner API."""
    spawned = []
    monkeypatch.setattr(routes.REGISTRY, "spawn",
                        lambda *a, **k: spawned.append(a) or "never")
    assert client.post("/ensemble/combiner/fit", data={"mode": "starfull"}).status_code == 404
    assert client.get("/ensemble/combiner.json?mode=starfull").status_code == 404
    assert spawned == []
