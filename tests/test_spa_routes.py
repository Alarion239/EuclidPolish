"""The SPA page matcher built from the committed route manifest (contract C1).

``euclid_polish/web/spa_routes.json`` is the single source of truth for page
URLs; Flask serves ``static/dist/index.html`` for exactly those paths, 308s
the legacy URLs to their new homes and lets every data endpoint through.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from euclid_polish.web import spa_routes
from euclid_polish.web.spa_routes import (
    MANIFEST_PATH,
    is_page_path,
    load_manifest,
    page_paths,
    redirect_target,
)

ROOT = Path(__file__).parents[1]
MANIFEST = json.loads(MANIFEST_PATH.read_text())


def _expanded_workspace_paths() -> list[tuple[str, list[str]]]:
    """(concrete workspace path, tabs) for every workspace × param value."""
    out = []
    for workspace in MANIFEST["workspaces"]:
        paths = [workspace["path"]]
        for name, values in (workspace.get("params") or {}).items():
            paths = [
                path.replace(f":{name}", value)
                for path in paths for value in values
            ]
        out.extend((path, workspace["tabs"]) for path in paths)
    return out


def test_manifest_path_is_the_committed_web_manifest():
    assert MANIFEST_PATH == ROOT / "euclid_polish" / "web" / "spa_routes.json"
    assert load_manifest() == MANIFEST


def test_load_manifest_reads_an_explicit_path(tmp_path):
    custom = {
        "version": 1,
        "workspaces": [
            {"id": "lab", "label": "Lab", "path": "/lab", "tabs": ["bench"]},
        ],
        "redirects": {"/old-lab": "/lab/bench"},
    }
    path = tmp_path / "routes.json"
    path.write_text(json.dumps(custom))

    manifest = load_manifest(path)

    assert manifest == custom
    assert page_paths(manifest) == frozenset({"/lab", "/lab/bench"})
    assert is_page_path("/lab/bench", manifest)
    assert not is_page_path("/sky", manifest)
    assert redirect_target("/old-lab", "", manifest) == "/lab/bench"


@pytest.mark.parametrize("path", [
    "/", "/sky", "/sky/atlas", "/sky/results", "/sky/catalog-eval",
    "/ensemble/starfull", "/ensemble/starless", "/ensemble/starless/train",
    "/ensemble/starfull/overview", "/realism/noise", "/data/records",
    "/figures/plates", "/inspect", "/ops/fasrc", "/settings",
    "/settings/about",
])
def test_workspace_and_tab_paths_are_pages(path):
    assert is_page_path(path)


def test_every_workspace_and_tab_path_is_a_page():
    expected = set()
    for path, tabs in _expanded_workspace_paths():
        expected.add(path)
        expected.update(f"{path.rstrip('/')}/{tab}" for tab in tabs)

    assert page_paths() == frozenset(expected)
    for path in expected:
        assert is_page_path(path), path


def test_trailing_slash_is_normalised():
    assert is_page_path("/sky/")
    assert is_page_path("/settings/about/")


@pytest.mark.parametrize("path", [
    "/ensemble/status.json",
    "/ensemble/foo",
    "/ensemble/starfull/foo",
    "/ensemble/overview",
    "/sky/unknown",
    "/sky/atlas/extra",
    "/inspect/preview.png",
    "/inspect/download",
    "/api/jobs",
    "/static/x.js",
    "/static/dist/index.html",
    "/viewer/meta/sky",
    "/config",
    "/ensemble",
])
def test_data_endpoints_and_unknown_segments_are_not_pages(path):
    assert not is_page_path(path)


@pytest.mark.parametrize("source,target", sorted(MANIFEST["redirects"].items()))
def test_every_redirect_entry_resolves(source, target):
    assert redirect_target(source, "") == target
    assert redirect_target(source, b"") == target
    assert redirect_target(source + "/", "") == target


@pytest.mark.parametrize("source,target", sorted(MANIFEST["redirects"].items()))
def test_redirects_preserve_the_query_string(source, target):
    assert redirect_target(source, "a=1&b=two") == f"{target}?a=1&b=two"
    assert redirect_target(source, b"x=%2F") == f"{target}?x=%2F"


def test_app_prefix_redirects_to_the_bare_route():
    assert redirect_target("/app/x", "y=1") == "/x?y=1"
    assert redirect_target("/app/x", b"y=1") == "/x?y=1"
    assert redirect_target("/app/inference", "") == "/inference"
    assert redirect_target("/app/sky/atlas", "") == "/sky/atlas"
    assert redirect_target("/app", "") == "/"
    assert redirect_target("/app/", "q=1") == "/?q=1"


@pytest.mark.parametrize("path,target", [
    ("/app//evil.example", "/evil.example"),
    ("/app//evil.example/x", "/evil.example/x"),
    ("/app///evil.example", "/evil.example"),
    ("/app/\\evil.example", "/evil.example"),
    ("/app/\\/evil.example", "/evil.example"),
    ("/app//\\evil.example", "/evil.example"),
    ("/app/\\\\evil.example", "/evil.example"),
    ("/app/\t/evil.example", "/evil.example"),
    ("/app/\n//evil.example", "/evil.example"),
    ("/app//", "/"),
])
def test_app_prefix_redirect_never_leaves_the_host(path, target):
    """``//host`` and ``/\\host`` are protocol-relative to a browser, so the
    ``/app/<rest>`` redirect collapses every leading slash/backslash: it can
    only ever point at a path on this server (no open redirect)."""
    location = redirect_target(path, "")
    assert location == target
    assert not location.startswith("//")
    assert not location.startswith("/\\")


@pytest.mark.parametrize("path", [
    "/sky", "/sky/atlas", "/api/jobs", "/ensemble/status.json", "/apple",
    "/application/x", "/", "/configure",
])
def test_non_redirect_paths_have_no_target(path):
    assert redirect_target(path, "q=1") is None


def test_redirect_targets_are_pages_and_sources_are_not():
    """Parity between the redirect table and the page set: every legacy URL
    lands on a real page and never shadows one."""
    for source, target in MANIFEST["redirects"].items():
        assert is_page_path(target), f"{source} → {target} is not a page"
        assert not is_page_path(source), f"redirect source {source} is a page"


def test_workspace_params_default_to_an_allowed_value():
    for workspace in MANIFEST["workspaces"]:
        for name, value in (workspace.get("defaultParams") or {}).items():
            assert value in workspace["params"][name]
        if workspace.get("defaultTab"):
            assert workspace["defaultTab"] in workspace["tabs"]


def test_matcher_is_cached_for_the_default_manifest():
    assert spa_routes.page_paths() is spa_routes.page_paths()


def test_no_pytest_reads_frontend_source_or_bundle_assets():
    """Frontend behaviour belongs to the frontend's own tests (vitest), not
    pytest; pytest must stay decoupled
    from the frontend source tree and the built bundle's asset files so the
    frontend can be restructured freely (the needles are split so this test
    does not match itself)."""
    needles = (
        "frontend" + "/src",
        "frontend" + '", "src',
        "dist" + "/assets",
    )
    offenders = []
    for test_file in sorted((ROOT / "tests").rglob("*.py")):
        text = test_file.read_text(encoding="utf-8")
        for needle in needles:
            if needle in text:
                offenders.append(f"{test_file.relative_to(ROOT)}: {needle}")
    assert offenders == []
