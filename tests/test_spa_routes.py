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
from euclid_polish.web.app import create_app
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
    "/", "/sky", "/sky/atlas", "/sky/targets", "/sky/compare",
    "/models/starfull", "/models/starless", "/models/starless/train",
    "/models/starfull/leaderboard", "/synthetic", "/synthetic/noise",
    "/synthetic/records", "/synthetic/psf", "/figures/plates",
    "/figures/sheet", "/files", "/runs/live", "/runs/history",
    "/notebook/log", "/system", "/system/storage",
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
    assert is_page_path("/system/code/")


@pytest.mark.parametrize("path", [
    "/ensemble/status.json",
    "/ensemble/foo",
    "/ensemble/starfull",
    "/ensemble/starfull/overview",
    "/models/foo",
    "/models/starfull/foo",
    "/models/leaderboard",
    "/sky/unknown",
    "/sky/atlas/extra",
    "/sky/results",
    "/files/x",
    "/inspect",
    "/inspect/preview.png",
    "/inspect/download",
    "/api/jobs",
    "/static/x.js",
    "/static/dist/index.html",
    "/viewer/meta/sky",
    "/config",
    "/ensemble",
    "/realism/noise",
    "/settings/about",
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
    assert redirect_target("/app/inference", "") == "/sky/targets"  # one hop
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


# ---------------------------------------------------------------------------
# Query-aware redirect rules (manifest v2 ``redirectRules``). The case file is
# shared with the SPA matcher (app/manifest.test.ts), so Flask and the router
# produce byte-identical targets.

CASES_PATH = Path(spa_routes.__file__).parent / "spa_redirect_cases.json"
CASES = json.loads(CASES_PATH.read_text(encoding="utf-8"))
RULES = MANIFEST["redirectRules"]


@pytest.mark.parametrize("case", CASES, ids=[c["from"] for c in CASES])
def test_redirect_cases(case):
    path, _, query = case["from"].partition("?")
    assert spa_routes.redirect_target(path, query) == case["to"]


def test_redirect_cases_are_unique_and_cover_every_rule_and_exact_entry():
    sources = [case["from"] for case in CASES]
    assert len(sources) == len(set(sources))
    paths = {spa_routes._normalise(s.partition("?")[0]) for s in sources}
    for source in MANIFEST["redirects"]:
        assert source in paths, f"no case for the exact redirect {source}"
    for rule in RULES:
        concrete = _concrete_sources(rule)
        assert paths & set(concrete), f"no case for the rule {rule}"


def _concrete_sources(rule) -> list[str]:
    """Every concrete ``from`` path of a rule (each allowed param value)."""
    paths = [rule["from"]]
    for name, values in (rule.get("params") or {}).items():
        paths = [p.replace(f":{name}", v) for p in paths for v in values]
    return paths


def _satisfying_query(rule) -> str:
    """A query string the rule's ``query`` condition accepts."""
    pairs = []
    for key, want in (rule.get("query") or {}).items():
        value = "x" if want == "*" else (want[0] if isinstance(want, list) else want)
        pairs.append((key, value))
    return "&".join(f"{spa_routes._form_encode(k)}={spa_routes._form_encode(v)}" for k, v in pairs)


def test_every_rule_param_is_constrained():
    """``:name`` binds only listed values, so a rule like ``/ensemble/:mode``
    can never capture a data endpoint (``/ensemble/status.json``)."""
    for rule in RULES:
        names = {seg[1:] for seg in rule["from"].split("/") if seg.startswith(":")}
        assert names == set(rule.get("params") or {}), rule["from"]
        for name, values in (rule.get("params") or {}).items():
            assert values and all(isinstance(v, str) and v for v in values), (rule["from"], name)


@pytest.mark.parametrize("rule", RULES, ids=[f"{r['from']}?{sorted(r.get('query') or {})}" for r in RULES])
def test_every_rule_lands_on_a_page_and_never_on_a_redirect(rule):
    """Each rule target is a page (no chains) and each source is not a page
    (a rule can never shadow one)."""
    query = _satisfying_query(rule)
    for source in _concrete_sources(rule):
        assert not is_page_path(source), f"rule source {source} is a page"
        target = redirect_target(source, query)
        assert target is not None, source
        path, _, _ = target.partition("?")
        assert is_page_path(path), f"{source}?{query} → {target} is not a page"
        assert redirect_target(path, target.partition("?")[2]) is None, f"{target} redirects again"
        assert "#" not in target


def test_exact_redirects_land_on_pages_without_a_query_or_chain():
    for source, target in MANIFEST["redirects"].items():
        assert "?" not in target and "#" not in target, f"{source} → {target}: use a rule for a query"
        assert redirect_target(target, "") is None, f"{source} → {target} redirects again"


def test_app_prefixed_old_paths_land_on_a_page_in_one_hop():
    """``/app/<old>`` resolves ``<old>`` too, so it never 308s onto a redirect."""
    sources = list(MANIFEST["redirects"]) + [
        rule["from"].replace(":mode", "starfull") for rule in RULES if ":" not in rule["from"].replace(":mode", "")
    ]
    for source in sources:
        target = redirect_target(f"/app{source}", "")
        assert target is not None, source
        path = target.partition("?")[0]
        assert redirect_target(path, "") is None, f"/app{source} → {target} redirects again"


def test_rules_are_well_formed():
    allowed = {"from", "params", "query", "to", "drop", "rename", "map", "prefix", "set"}
    for rule in RULES:
        assert set(rule) <= allowed, rule
        assert rule["from"].startswith("/") and rule["to"].startswith("/"), rule
        for key, want in (rule.get("query") or {}).items():
            assert want == "*" or isinstance(want, str) or (
                isinstance(want, list) and all(isinstance(v, str) for v in want)
            ), (rule["from"], key)


def test_a_rule_may_constrain_a_path_param_and_rewrite_the_query(tmp_path):
    custom = {
        "version": 2,
        "workspaces": [{"id": "lab", "label": "Lab", "path": "/lab", "tabs": ["bench", "log"]}],
        "redirectRules": [
            {"from": "/old/:kind", "params": {"kind": ["a", "b"]}, "query": {"v": ["x", "y"]},
             "to": "/lab/bench", "drop": ["v"], "rename": {"q": "find"},
             "map": {"find": {"old": "new"}}, "prefix": {"id": "run:"}, "set": {"kind": ":kind"}},
            {"from": "/old/:kind", "params": {"kind": ["a", "b"]}, "to": "/lab/log"},
        ],
        "redirects": {"/old/a": "/lab/bench"},
    }
    path = tmp_path / "routes.json"
    path.write_text(json.dumps(custom))
    manifest = load_manifest(path)

    assert redirect_target("/old/a", "v=x&q=old&id=7&z=1", manifest) == (
        "/lab/bench?find=new&id=run%3A7&z=1&kind=a"
    )
    assert redirect_target("/old/b", "v=z", manifest) == "/lab/log?v=z"
    # the rules win over the exact map; an unlisted param value matches nothing
    assert redirect_target("/old/a", "", manifest) == "/lab/log"
    assert redirect_target("/old/c", "v=x", manifest) is None


def test_no_flask_endpoint_is_a_page_or_a_redirect_source():
    """No data endpoint (every static GET rule of the app) is shadowed by the
    SPA shell or captured by a redirect."""
    app = create_app()
    for url_rule in app.url_map.iter_rules():
        if "<" in url_rule.rule or "GET" not in (url_rule.methods or ()):
            continue
        assert not is_page_path(url_rule.rule), url_rule.rule
        assert redirect_target(url_rule.rule, "") is None, url_rule.rule
