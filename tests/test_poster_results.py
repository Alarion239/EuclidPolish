"""Poster cutout result routes: an explicit (FASRC) pull, offline-first GETs
of the last pulled copy that never write, and the data/vis archive."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from flask import Flask

from euclid_polish.config import Config
from euclid_polish.web.routes import poster

PNG = b"\x89PNG\r\n\x1a\n" + b"\x01" * 32
FITS = b"SIMPLE  =                    T" + b" " * 50


@pytest.fixture
def client(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setattr(Config, "FASRC_CACHE_DIR", str(cache))
    monkeypatch.setattr(Config, "VIS_DIR", str(tmp_path / "vis"))
    monkeypatch.setattr(poster.fasrc_config, "load", lambda: SimpleNamespace(data_dir="/n/remote/data"))
    remote_bodies = {"poster_cutout.png": PNG, "poster_cutout.fits": FITS}
    pulls = []

    def fake_fetch(remote_path, *, force=False, **_kw):
        pulls.append((remote_path, force))
        name = remote_path.rsplit("/", 1)[-1]
        if name not in remote_bodies:
            return SimpleNamespace(ok=False, local_path=None, error="no such file")
        local = poster._local_path_for(remote_path)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        with open(local, "wb") as fh:
            fh.write(remote_bodies[name])
        return SimpleNamespace(ok=True, local_path=local, error=None)

    monkeypatch.setattr(poster, "fetch_one_file", fake_fetch)
    app = Flask(__name__)
    app.config.update(TESTING=True)
    poster.register(app)
    return app.test_client(), tmp_path, pulls, remote_bodies


def test_gets_serve_nothing_before_a_pull_and_never_write(client):
    http, tmp_path, pulls, _bodies = client
    status = http.get("/poster/result/status").get_json()
    assert status["ok"] is True and status["available"] is False
    assert status["png"] is None and status["fits"] is None
    response = http.get("/poster/result/cutout.png")
    assert response.status_code == 404
    assert "pull its result" in response.get_json()["error"]
    assert http.get("/poster/result/cutout.fits").status_code == 404
    assert pulls == []
    assert not (tmp_path / "vis").exists() and not (tmp_path / "cache").exists()


def test_pull_then_serve_locally_and_archive_once(client):
    http, tmp_path, pulls, bodies = client
    response = http.post("/poster/result/pull")
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["ok"] is True and body["errors"] == {}
    assert body["png"]["size"] == len(PNG) and body["fits"]["size"] == len(FITS)
    assert {path for path, _force in pulls} == {
        "/n/remote/data/_poster/poster_cutout.png", "/n/remote/data/_poster/poster_cutout.fits"}
    assert all(force for _path, force in pulls)
    archived = sorted((tmp_path / "vis" / "poster").iterdir())
    assert len(archived) == 1 and archived[0].read_bytes() == PNG

    png = http.get("/poster/result/cutout.png")
    assert png.status_code == 200 and png.data == PNG and png.mimetype == "image/png"
    fits = http.get("/poster/result/cutout.fits")
    assert fits.status_code == 200 and fits.headers["Content-Disposition"].startswith("attachment")
    assert http.get("/poster/result/status").get_json()["available"] is True

    # an identical re-pull archives nothing new; a changed PNG is archived
    assert http.post("/poster/result/pull").get_json()["archived"] is None
    bodies["poster_cutout.png"] = PNG + b"\x02"
    assert http.post("/poster/result/pull").get_json()["archived"]
    assert len(list((tmp_path / "vis" / "poster").iterdir())) == 2


def test_pull_reports_a_missing_remote_result(client):
    http, _tmp, _pulls, bodies = client
    bodies.clear()
    response = http.post("/poster/result/pull")
    assert response.status_code == 404
    body = response.get_json()
    assert body["ok"] is False and body["errors"]["png"] == "no such file"
    assert http.get("/poster/result/pull").status_code == 405
