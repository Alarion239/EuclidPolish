"""``euclid_polish/web/API.md`` documents every endpoint the app serves.

The endpoint tables in API.md are the reference for the frontend and for
every backend work package: each row is ``| METHODS | `rule` | gate | notes |``
with the Flask rule syntax (``<job_id>``, ``<path:relpath>``). This test keeps
the tables in lock-step with ``app.url_map`` — an added, removed or re-methoded
route, or a changed ``@requires_fasrc`` mark, must be reflected in the doc.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from euclid_polish.web.app import create_app

API_MD = Path(__file__).parents[1] / "euclid_polish" / "web" / "API.md"

# Flask's own static route is infrastructure, not an API endpoint.
_UNDOCUMENTED = {"/static/<path:filename>"}

_ROW = re.compile(r"^\|\s*([A-Z, ]+?)\s*\|\s*`(/[^`]*)`\s*\|\s*([^|]*?)\s*\|")


def _documented() -> dict[str, tuple[frozenset[str], bool]]:
    rows: dict[str, tuple[frozenset[str], bool]] = {}
    for line in API_MD.read_text(encoding="utf-8").splitlines():
        match = _ROW.match(line)
        if not match:
            continue
        methods, rule, gate = match.groups()
        assert rule not in rows, f"{rule} is documented twice"
        rows[rule] = (
            frozenset(m.strip() for m in methods.split(",") if m.strip()),
            gate.strip().lower() == "fasrc",
        )
    return rows


@pytest.fixture(scope="module")
def served() -> dict[str, tuple[frozenset[str], bool]]:
    app = create_app()
    rules: dict[str, tuple[set[str], bool]] = {}
    for rule in app.url_map.iter_rules():
        if rule.rule in _UNDOCUMENTED:
            continue
        view = app.view_functions[rule.endpoint]
        methods, gated = rules.setdefault(rule.rule, (set(), False))
        methods.update((rule.methods or set()) - {"HEAD", "OPTIONS"})
        rules[rule.rule] = (methods, gated or bool(getattr(view, "_requires_fasrc", False)))
    return {rule: (frozenset(m), g) for rule, (m, g) in rules.items()}


def test_api_md_exists_with_the_conventions():
    text = API_MD.read_text(encoding="utf-8")
    for heading in ("## Conventions", "fasrc_offline", "TRUSTED_HOSTS",
                    "POST", "/api/jobs", "/api/version"):
        assert heading in text


def test_every_served_endpoint_is_documented(served):
    documented = _documented()
    missing = sorted(set(served) - set(documented))
    stale = sorted(set(documented) - set(served))
    assert missing == [], f"add these routes to API.md: {missing}"
    assert stale == [], f"these API.md rows no longer exist: {stale}"


def test_documented_methods_match_the_routes(served):
    documented = _documented()
    wrong = {
        rule: (sorted(documented[rule][0]), sorted(methods))
        for rule, (methods, _gated) in served.items()
        if rule in documented and documented[rule][0] != methods
    }
    assert wrong == {}


def test_documented_fasrc_gate_matches_the_marks(served):
    documented = _documented()
    wrong = {
        rule: {"doc": documented[rule][1], "route": gated}
        for rule, (_methods, gated) in served.items()
        if rule in documented and documented[rule][1] != gated
    }
    assert wrong == {}
