from __future__ import annotations

import os

import pytest
from flask import Flask

from euclid_polish.config import Config
from euclid_polish.web.routes import evaluation


@pytest.fixture
def client(tmp_path, monkeypatch):
    root = tmp_path / "eval_results"
    root.mkdir()
    monkeypatch.setattr(Config, "EVAL_RESULTS_DIR", str(root))
    app = Flask(__name__)
    app.config.update(TESTING=True)
    evaluation.register(app)
    return app.test_client(), root


def _manifest(directory, object_id):
    directory.mkdir(exist_ok=True)
    (directory / "manifest.csv").write_text(f"id,ok\n{object_id},True\n")


def test_runs_api_reads_the_selected_child_manifest(client):
    http, root = client
    _manifest(root, "root-object")
    _manifest(root / "child", "child-object")

    response = http.get("/api/evaluation/runs?run=child")

    assert response.status_code == 200
    assert response.get_json()["run"] == "child"
    assert response.get_json()["rows"][0]["id"] == "child-object"


@pytest.mark.parametrize("run", ["../escape", ".", "a/b", r"a\b"])
def test_runs_api_rejects_unsafe_run_names(client, run):
    http, _root = client
    assert http.get("/api/evaluation/runs", query_string={"run": run}).status_code == 400


def test_runs_api_returns_404_for_missing_child(client):
    http, _root = client
    assert http.get("/api/evaluation/runs?run=missing").status_code == 404


def test_runs_api_rejects_child_symlink_escape(client, tmp_path):
    http, root = client
    outside = tmp_path / "outside"
    _manifest(outside, "outside-object")
    os.symlink(outside, root / "escaped")

    assert http.get("/api/evaluation/runs?run=escaped").status_code == 400


def test_rerender_only_removes_selected_run_images(client):
    http, root = client
    child = root / "child"
    child.mkdir()
    root_png = root / "eye.png"
    child_png = child / "eye.png"
    root_png.write_bytes(b"root")
    child_png.write_bytes(b"child")

    response = http.post("/api/evaluation/rerender", data={"run": "child"})

    assert response.status_code == 200
    assert response.get_json()["removed"] == 1
    assert root_png.exists()
    assert not child_png.exists()
