"""GETs that still do work for the user (a ``?fresh=1`` re-render, a FASRC
file pull behind a link) refuse cross-site requests: a hostile page's
``<img src=http://localhost:…>`` carries our Host, so only fetch metadata
(``Sec-Fetch-Site``) tells it from the SPA. The state-changing variants are
POSTs (``POST /api/evaluation/angular-power-spectrum`` re-renders;
``POST /api/fasrc/file/fetch`` pulls into the cache)."""

from __future__ import annotations

import os

import pytest

from euclid_polish.config import Config
from euclid_polish.eval import power_spectrum
from euclid_polish.web import fasrc_fetcher, remote
from euclid_polish.web.app import create_app
from euclid_polish.web.fasrc_fetcher import FetchResult
from euclid_polish.web.routes import ensemble as ensemble_routes

CROSS_SITE = {"Sec-Fetch-Site": "cross-site"}
SAME_SITE_OTHER_PORT = {"Sec-Fetch-Site": "same-site"}
SAME_ORIGIN = {"Sec-Fetch-Site": "same-origin"}


class _Up:
    def is_connected(self) -> bool:
        return True


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def cached_spectrum(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(tmp_path / "res"))
    os.makedirs(Config.EVAL_RESULTS_DIR)
    png = os.path.join(Config.EVAL_RESULTS_DIR, "angular_power_spectrum.png")
    with open(png, "wb") as handle:
        handle.write(b"\x89PNG cached")
    renders: list[str] = []

    def render(out_png, *, out_json=None):
        renders.append(out_png)
        with open(out_png, "wb") as handle:
            handle.write(b"\x89PNG fresh")
        return out_png

    monkeypatch.setattr(power_spectrum, "render_power_spectrum_summary", render)
    return renders


@pytest.mark.parametrize("headers", [CROSS_SITE, SAME_SITE_OTHER_PORT])
def test_a_cross_site_fresh_request_is_served_from_the_cache(client, cached_spectrum, headers):
    response = client.get("/api/evaluation/angular-power-spectrum?fresh=1", headers=headers)
    assert response.status_code == 200 and response.data == b"\x89PNG cached"
    assert cached_spectrum == []


def test_the_spa_can_still_force_a_re_render(client, cached_spectrum):
    response = client.get("/api/evaluation/angular-power-spectrum?fresh=1", headers=SAME_ORIGIN)
    assert response.data == b"\x89PNG fresh" and len(cached_spectrum) == 1


def test_a_post_re_renders(client, cached_spectrum):
    response = client.post("/api/evaluation/angular-power-spectrum")
    assert response.status_code == 200 and response.get_json()["ok"] is True
    assert len(cached_spectrum) == 1
    assert client.post("/api/evaluation/angular-power-spectrum",
                       headers={"Origin": "http://evil.example"}).status_code == 403


@pytest.mark.parametrize("path", ["/fasrc/file/inspect", "/fasrc/file/download"])
def test_a_cross_site_link_cannot_pull_a_fasrc_file(client, monkeypatch, path):
    monkeypatch.setattr(remote.STATE, "ssh", _Up())
    pulls: list[str] = []
    monkeypatch.setattr(fasrc_fetcher, "fetch_one_file",
                        lambda remote_path: pulls.append(remote_path) or FetchResult(ok=False))
    response = client.get(f"{path}?remote_path=/n/x.fits", headers=CROSS_SITE)
    assert response.status_code == 403
    assert response.get_json()["ok"] is False
    assert pulls == []


def test_the_fetch_post_pulls_into_the_cache(client, monkeypatch, tmp_path):
    monkeypatch.setattr(remote.STATE, "ssh", _Up())
    local = tmp_path / "x.fits"
    local.write_bytes(b"SIMPLE")
    monkeypatch.setattr(fasrc_fetcher, "fetch_one_file",
                        lambda remote_path: FetchResult(ok=True, local_path=str(local)))
    monkeypatch.setattr("euclid_polish.web.routes.files._safe_relpath", lambda path: "data/x.fits")
    body = client.post("/api/fasrc/file/fetch", data={"remote_path": "/n/x.fits"}).get_json()
    assert body["ok"] is True
    assert body["inspect_url"] == "/files?fits=data%2Fx.fits"
    assert body["download_url"] == "/fasrc/file/download?remote_path=%2Fn%2Fx.fits"
    assert client.post("/api/fasrc/file/fetch", data={}).status_code == 400
    monkeypatch.setattr(fasrc_fetcher, "fetch_one_file",
                        lambda remote_path: FetchResult(ok=False, error="too big"))
    failed = client.post("/api/fasrc/file/fetch", data={"remote_path": "/n/x.fits"})
    assert failed.status_code == 502 and failed.get_json() == {"ok": False, "error": "too big"}


def test_a_cross_site_get_never_renders_a_missing_figure(client, cached_spectrum):
    os.remove(os.path.join(Config.EVAL_RESULTS_DIR, "angular_power_spectrum.png"))
    response = client.get("/api/evaluation/angular-power-spectrum", headers=CROSS_SITE)
    assert response.status_code == 404 and cached_spectrum == []
    assert "not rendered yet" in response.get_json()["error"]
    response = client.get("/api/evaluation/angular-power-spectrum", headers=SAME_ORIGIN)
    assert response.data == b"\x89PNG fresh" and len(cached_spectrum) == 1


@pytest.fixture
def evals_cache(tmp_path, monkeypatch):
    path = tmp_path / "ensemble_evals.json"
    computed: list[str] = []

    def compute(starless):
        computed.append("payload")
        path.write_text('{"coherence": {}, "fresh": true}', encoding="utf-8")
        return str(path)

    def refresh(starless):
        computed.append("diagnostics")
        return str(path)

    monkeypatch.setattr(ensemble_routes, "_evals_payload_path", lambda starless: str(path))
    monkeypatch.setattr(ensemble_routes, "compute_evaluation_payload", compute)
    monkeypatch.setattr(ensemble_routes, "refresh_evaluation_diagnostics", refresh)
    return path, computed


def test_a_cross_site_get_never_computes_the_evaluation_payload(client, evals_cache):
    path, computed = evals_cache
    response = client.get("/ensemble/evals.json", headers=CROSS_SITE)
    assert response.status_code == 404 and computed == []
    path.write_text('{"old": true}', encoding="utf-8")        # predates the diagnostics
    response = client.get("/ensemble/evals.json", headers=CROSS_SITE)
    assert response.status_code == 200 and response.get_json() == {"old": True}
    assert computed == []                                     # served as cached
    response = client.get("/ensemble/evals.json", headers=SAME_ORIGIN)
    assert response.get_json()["fresh"] is True and "payload" in computed
