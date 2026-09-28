"""Home's cached real-SR previews (routes/figures.py ``/api/figures/real-sr``):
read the C9 output store only, newest first, never run a model."""
from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest
from flask import Flask
from PIL import Image

from euclid_polish.web.helpers import model_catalog
from euclid_polish.web.routes import figures


def _spec(fingerprint: str) -> SimpleNamespace:
    return SimpleNamespace(
        spec="production", slug=model_catalog.spec_slug("production"), kind="production",
        label="Production · spatial gate", fingerprint=fingerprint, member_labels=["m1"],
        member_fingerprints=["f1"], combiner_kind="spatial_gate", combiner_fingerprint="c1")


def _sr(side: int) -> np.ndarray:
    y, x = np.mgrid[:side, :side]
    blob = np.exp(-((x - side / 2) ** 2 + (y - side / 2) ** 2) / (2 * (side / 8) ** 2))
    return np.stack([200.0 * (i + 1) * blob + 1.0 for i in range(4)], axis=-1).astype(np.float32)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(model_catalog, "outputs_root", lambda: tmp_path / "outputs")
    monkeypatch.setattr(model_catalog, "current_fingerprints", lambda specs=None: {"production": "fp-now"})
    figures._REAL_THUMBS.clear()
    older = model_catalog.save_output("nexus", "f200w-0040", _spec("fp-old"), _sr(256),
                                      lr_header=None, lr_sha=None)
    model_catalog.update_output_meta("nexus", "f200w-0040", "production",
                                     {"created": "2026-09-01T00:00:00+00:00"})
    model_catalog.save_output("tile", "ra0273.07050_decp066.36241", _spec("fp-now"), _sr(128),
                              lr_header=None, lr_sha=None)
    # a non-production output and a sidecar without its FITS are ignored
    model_catalog.save_output("nexus", "f200w-0042", SimpleNamespace(**{**vars(_spec("x")), "spec": "mean", "slug": "mean"}),
                              _sr(32), lr_header=None, lr_sha=None)
    (tmp_path / "outputs" / "pair" / "p1").mkdir(parents=True)
    (tmp_path / "outputs" / "pair" / "p1" / "production.json").write_text('{"spec": "production", "file": "production.fits"}')
    app = Flask(__name__)
    app.config.update(TESTING=True)
    figures.register(app)
    return app.test_client(), older


def test_lists_the_newest_production_outputs_with_their_state(store):
    http, _older = store
    body = http.get("/api/figures/real-sr").get_json()
    assert body["total"] == 2
    assert [item["ref"] for item in body["items"]] == ["tile/ra0273.07050_decp066.36241", "nexus/f200w-0040"]
    assert [item["state"] for item in body["items"]] == ["current", "stale"]
    first = body["items"][0]
    assert first["source_label"] == "Cached 25.6″ tiles"
    assert first["thumb"] == "/api/figures/real-sr/tile/ra0273.07050_decp066.36241.jpg"
    assert http.get("/api/figures/real-sr?limit=1").get_json()["items"][0]["source"] == "tile"
    assert http.get("/api/figures/real-sr?limit=x").status_code == 400


def test_thumbnail_is_a_memoised_colour_jpeg(store):
    http, _older = store
    response = http.get("/api/figures/real-sr/nexus/f200w-0040.jpg?size=96")
    assert response.status_code == 200 and response.mimetype == "image/jpeg"
    with Image.open(io.BytesIO(response.data)) as image:
        assert image.mode == "RGB" and max(image.size) == 96
    assert figures.real_sr_thumbnail("nexus", "f200w-0040", 96) is figures.real_sr_thumbnail("nexus", "f200w-0040", 96)


def test_thumbnail_refuses_missing_and_unsafe_tiles(store):
    http, _older = store
    assert http.get("/api/figures/real-sr/nexus/f200w-0042.jpg").status_code == 404   # only a mean output
    assert http.get("/api/figures/real-sr/pair/p1.jpg").status_code == 404
    assert http.get("/api/figures/real-sr/-bad/x.jpg").status_code == 400
    assert http.get("/api/figures/real-sr/nexus/f200w-0040.jpg?size=big").status_code == 400


def test_opening_the_list_never_runs_a_model(store, monkeypatch):
    http, _older = store

    def refuse(*args, **kwargs):
        raise AssertionError("a model ran")

    monkeypatch.setattr(model_catalog, "predict", refuse)
    monkeypatch.setattr(model_catalog, "save_output", refuse)
    assert http.get("/api/figures/real-sr").status_code == 200
    assert http.get("/api/figures/real-sr/tile/ra0273.07050_decp066.36241.jpg").status_code == 200
