from __future__ import annotations

import json

import pytest

from euclid_polish.config import Config
from euclid_polish.web.helpers import q1_galaxy_radius_statistics as radius_stats


def test_radius_statistics_query_uses_only_aggregate_brackets(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(radius_stats, "Q1_GALAXY_RADIUS_BIN_COUNT", 2)
    queries: list[str] = []

    def launch(query, _relogin):
        queries.append(query)
        if "weighted_radius_sum" in query:
            return {
                "selected_radii": 20,
                "expected_radii": 16.0,
                "weighted_radius_sum": 8.0,
                "weighted_radius2_sum": 5.0,
            }
        return {"selected_radii": 30, "expected_radii": 24.0}

    monkeypatch.setattr(radius_stats, "_launch_with_relogin", launch)
    progress = []

    payload = radius_stats.query_q1_galaxy_radius_statistics(
        bright=14.0,
        faint=14.2,
        bin_width=0.1,
        progressive_stride=0.1,
        progress=lambda done, total, label: progress.append(
            (done, total, label),
        ),
    )

    assert payload["complete"] is True
    assert payload["completed_queries"] == payload["total_queries"] == 4
    assert len(payload["magnitude_bins"]) == 2
    assert len(payload["radius_bins"]) == 2
    assert "no object rows and no random sky-position" in payload["acquisition"]
    assert all("JOIN catalogue.mer_morphology AS morph" in query for query in queries)
    assert all("COUNT(*) AS selected_radii" in query for query in queries)
    assert all("SELECT TOP" not in query for query in queries)
    assert progress[-1][:2] == (4, 4)
    assert radius_stats.read_q1_galaxy_radius_statistics()["complete"] is True


def test_radius_statistics_rejects_stale_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    path = radius_stats.q1_galaxy_radius_statistics_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"version": 0}))

    with pytest.raises(ValueError, match="stale|first"):
        radius_stats.read_q1_galaxy_radius_statistics()
