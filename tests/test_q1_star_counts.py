"""Focused tests for Q1-wide PHZ stellar number counts."""

from __future__ import annotations

import json

import pytest

from euclid_polish.config import Config
from euclid_polish.web.helpers import q1_star_counts


class _Job:
    def __init__(self, row):
        self.row = row

    def get_results(self):
        return [self.row]


def test_q1_phz_counts_are_probability_weighted_and_area_normalized(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    queries: list[str] = []
    rows = iter([
        {
            "selected_point_sources": 5,
            "expected_point_sources": 4.8,
            "point_source_variance": 0.18,
        },
        {
            "classified_rows": 12,
            "expected_stars": 4.0,
            "classification_variance": 1.5,
        },
        {
            "selected_point_sources": 10,
            "expected_point_sources": 9.5,
            "point_source_variance": 0.4,
        },
        {
            "classified_rows": 20,
            "expected_stars": 9.0,
            "classification_variance": 2.5,
        },
    ])

    def launch(query):
        queries.append(query)
        return _Job(next(rows))

    monkeypatch.setattr(q1_star_counts.Euclid, "launch_job_async", launch)
    progress = []
    result = q1_star_counts.query_q1_phz_star_counts(
        bright=20.0,
        faint=20.2,
        bin_width=0.1,
        progress=lambda done, total, label: progress.append(
            (done, total, label),
        ),
    )

    assert result["edges"] == pytest.approx([20.0, 20.1, 20.2])
    assert result["expected_stars"] == pytest.approx(13.0)
    assert result["selected_point_sources"] == 15
    assert result["expected_point_sources"] == pytest.approx(14.3)
    assert result["point_source_variance"] == pytest.approx(0.58)
    assert result["bins"][0]["point_source_density_arcmin2_mag"] == pytest.approx(
        4.8 / (63.1 * 3600.0) / 0.1,
    )
    assert result["classification_variance"] == pytest.approx(4.0)
    assert result["bins"][0]["density_arcmin2_mag"] == pytest.approx(
        4.0 / (63.1 * 3600.0) / 0.1,
    )
    assert len(queries) == 4
    point_queries = queries[::2]
    phz_queries = queries[1::2]
    assert all("SUM(mer.point_like_prob)" in query for query in point_queries)
    assert all("phz_classification" not in query for query in point_queries)
    assert all("JOIN catalogue.phz_classification" in query for query in phz_queries)
    assert all("SUM(cls.phz_star_prob)" in query for query in phz_queries)
    assert all("mer.point_like_prob >= 0.9" in query for query in queries)
    assert all("mer.flux_vis_psf" in query for query in queries)
    assert all("269.733" in query and "61.241" in query and "52.932" in query
               for query in queries)
    assert all("det_quality_flag" not in query for query in queries)
    assert result["selection"].startswith("three Q1 deep-field regions")
    assert "POINT_LIKE_PROB >= 0.9" in result["selection"]
    assert progress[-1][:2] == (4, 4)
    cached = json.loads(q1_star_counts.q1_star_counts_path().read_text())
    assert cached["expected_stars"] == pytest.approx(13.0)
    assert cached["expected_point_sources"] == pytest.approx(14.3)


def test_q1_phz_count_refresh_preserves_previous_cache_on_failure(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    output = q1_star_counts.q1_star_counts_path()
    output.parent.mkdir(parents=True)
    output.write_text('{"sentinel": true}')
    monkeypatch.setattr(
        q1_star_counts.Euclid,
        "launch_job_async",
        lambda _query: None,
    )

    with pytest.raises(RuntimeError, match="rejected"):
        q1_star_counts.query_q1_phz_star_counts(
            bright=20.0, faint=20.1, bin_width=0.1,
        )

    assert json.loads(output.read_text()) == {"sentinel": True}
