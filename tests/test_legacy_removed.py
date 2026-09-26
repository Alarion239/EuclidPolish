"""The classic (Jinja) console and its dead endpoints are gone (spec §10).

The React SPA serves every page (``spa_routes.json``, contract C1); the
server keeps only the JSON/PNG/FITS endpoints the SPA (or a wired-in
feature) calls. These tests pin the removal so no legacy page handler,
template or superseded endpoint creeps back.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from euclid_polish.web import app as web_app
from euclid_polish.web import fasrc_config, fasrc_fetcher, remote
from euclid_polish.web.fasrc_config import FasrcConfig
from euclid_polish.web.fasrc_fetcher import FetchResult
from euclid_polish.web.helpers import (
    ensemble_viz,
    fits_render,
    jobs_impl,
    status,
    viewer_data,
)
from euclid_polish.web.routes import evaluation as evaluation_routes

WEB = Path(web_app.__file__).parent

# Rules that must no longer exist in ``app.url_map`` (page handlers the SPA
# shell always answered first, unreferenced and superseded endpoints).
REMOVED_RULES = [
    # page handlers (GET answered by the SPA shell / C1 redirects)
    "/", "/connection-error", "/catalog", "/sky", "/config", "/cutouts",
    "/cutouts/<band_name>", "/ensemble", "/fasrc", "/git", "/inference",
    "/training", "/psfs", "/tng", "/tracking", "/visualization", "/inspect",
    "/evaluation",
    # unreferenced
    "/ensemble/render", "/ensemble/eval-plot/<plot>.png", "/view/star-cutout",
    "/api/jwst-euclid/saved", "/api/jwst-euclid/nexus/options",
    "/api/jwst-euclid/field/<identifier>/<kind>", "/api/fasrc/eta",
    "/api/fasrc/jobs", "/api/fasrc/mirror/start", "/api/fasrc/mirror/stop",
    "/api/fasrc/runs/ckpt-bundle.tar",
    # superseded
    "/ensemble/power-spectrum.png", "/api/euclid-psf/preview",
    "/api/sky/totals",
    "/api/fasrc/runs/training-plot.png", "/api/fasrc/training-status",
    "/api/fasrc/log/<jobid>", "/api/fasrc/stages/<jobid>", "/api/fasrc/submit",
    "/star-cutout/inspect", "/sky/inspect", "/sky/fits",
]

# Wired-not-deleted endpoints (spec §10) stay.
KEPT_RULES = [
    "/api/fasrc/queue/remove", "/api/fasrc/refresh-accounting",
    "/api/tracking/backup", "/api/fasrc/config",
    "/api/jwst-euclid/nexus/download",
    "/api/jwst-euclid/field/<identifier>/download/<kind>",
    "/inference-files/<path:relpath>", "/viewer/results/<result_id>",
    "/viewer/results/<result_id>/panel.png", "/poster/result/cutout.png",
    "/poster/result/cutout.fits", "/api/connection/retry",
    "/fasrc/file/inspect", "/fasrc/file/download",
    # only its PNG renderer was superseded; the FITS download stays
    "/eval-files/<path:relpath>",
]


@pytest.fixture(scope="module")
def rules() -> set[str]:
    return {rule.rule for rule in web_app.create_app().url_map.iter_rules()}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(remote.STATE, "ssh", None)
    application = web_app.create_app()
    application.config["TESTING"] = True
    with application.test_client() as c:
        yield c


@pytest.mark.parametrize("rule", REMOVED_RULES)
def test_legacy_rule_is_gone(rules, rule):
    assert rule not in rules


@pytest.mark.parametrize("rule", KEPT_RULES)
def test_wired_rule_is_kept(rules, rule):
    assert rule in rules


def test_templates_and_classic_assets_are_deleted():
    assert not (WEB / "templates").exists()
    for name in ("ensemble_combiner.js", "ensemble_evals.js",
                 "ensemble_train_curves.js", "fasrc_step_card.js",
                 "job_status.js", "style.css"):
        assert not (WEB / "static" / name).exists(), name
    # static/cutout_viewer.js is WP-V's to port and delete; not pinned here.


def test_no_python_module_renders_a_template():
    offenders = [
        str(path.relative_to(WEB))
        for path in WEB.rglob("*.py")
        if "frontend" not in path.parts
        and ("render_template" in path.read_text(encoding="utf-8")
             or "url_for(" in path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_dead_route_modules_are_deleted():
    assert not (WEB / "routes" / "catalog.py").exists()
    assert not (WEB / "routes" / "sky.py").exists()


@pytest.mark.parametrize("module,name", [
    (jobs_impl, "_job_generate_reconstruct"),
    (jobs_impl, "_forward_model_sr_residual"),
    (jobs_impl, "_job_reconstruct_euclid_cutout"),
    (viewer_data, "_jwst_filter_tint"),
    (viewer_data, "_pair_asinh"),
    (fits_render, "_arrays_to_fits_bytes"),
    (fits_render, "_psf_preview_payload"),
    (status, "_cutout_layout_status"),
    (ensemble_viz, "job_ensemble_render"),
    (ensemble_viz, "regenerate_power_spectrum"),
    (ensemble_viz, "regenerate_eval_diagnostics"),
    (evaluation_routes, "_list_catalogs"),
    (evaluation_routes, "_render_object_png"),
])
def test_orphan_function_is_deleted(module, name):
    assert not hasattr(module, name)


def test_kept_live_helpers_survive():
    """Verified live (spec §10 KEEP): do not delete these with the legacy."""
    assert (WEB / "helpers" / "tng_prior.py").is_file()
    assert hasattr(jobs_impl, "reconstruct_cutout_at")
    assert hasattr(status, "_fasrc_catalog_dir")


def test_fasrc_config_has_no_science_fields():
    fields = set(FasrcConfig().to_dict())
    for name in ("n_train", "n_valid", "n_test", "image_size", "batch_size",
                 "steps"):
        assert name not in fields
    for name in ("ssh_user", "ssh_host", "repo_path", "data_dir", "ckpt_dir",
                 "conda_env_path", "logs_subdir"):
        assert name in fields


def test_fasrc_config_ignores_persisted_legacy_science_keys(tmp_path, monkeypatch):
    path = tmp_path / "fasrc.json"
    path.write_text('{"ssh_user": "astro", "n_train": 6400, "steps": 1}')
    monkeypatch.setattr(fasrc_config, "CONFIG_PATH", str(path))
    cfg = fasrc_config.load()
    assert cfg.ssh_user == "astro"
    assert not hasattr(cfg, "n_train")


class _Up:
    def is_connected(self) -> bool:
        return True


def test_fasrc_file_inspect_fetch_failure_is_json_502(client, monkeypatch):
    monkeypatch.setattr(remote.STATE, "ssh", _Up())
    monkeypatch.setattr(fasrc_fetcher, "fetch_one_file",
                        lambda remote_path: FetchResult(ok=False, error="too big"))
    response = client.get("/fasrc/file/inspect?remote_path=/n/x.fits")
    assert response.status_code == 502
    assert response.get_json() == {"ok": False, "error": "too big"}


def test_fasrc_file_inspect_redirects_to_the_inspect_workspace(
        client, monkeypatch, tmp_path):
    monkeypatch.setattr(remote.STATE, "ssh", _Up())
    local = Path(os.path.realpath(tmp_path)) / "x" / "a b.fits"
    monkeypatch.setattr(
        fasrc_fetcher, "fetch_one_file",
        lambda remote_path: FetchResult(ok=True, local_path=str(local)))
    monkeypatch.setattr(
        "euclid_polish.web.routes.files._safe_relpath", lambda path: "data/x/a b.fits")
    response = client.get("/fasrc/file/inspect?remote_path=/n/x.fits")
    assert response.status_code == 302
    assert response.headers["Location"] == "/inspect?fits=data%2Fx%2Fa+b.fits"
