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
    monkeypatch.setattr(radius_stats, "Q1_GALAXY_FWHM_BIN_COUNT", 2)
    queries: list[str] = []

    def launch(query, _relogin):
        queries.append(query)
        if "AS fwhm_bin" in query:
            return [
                {
                    "magnitude_bin": mag_index,
                    "fwhm_bin": fwhm_index,
                    "selected_fwhm": 20,
                    "expected_fwhm": 16.0,
                }
                for mag_index in range(2)
                for fwhm_index in range(2)
            ]
        return [
            {
                "magnitude_bin": mag_index,
                "radius_bin": radius_index,
                "selected_radii": 20,
                "expected_radii": 16.0,
            }
            for mag_index in range(2)
            for radius_index in range(2)
        ]

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
    assert payload["completed_queries"] == payload["total_queries"] == 2
    assert len(payload["joint_bins"]) == 4
    assert len(payload["magnitude_fwhm_bins"]) == 4
    assert len(payload["magnitude_bins"]) == 2
    assert len(payload["radius_bins"]) == 2
    assert len(payload["fwhm_bins"]) == 2
    assert "no object rows and no random sky-position" in payload["acquisition"]
    assert payload["version"] == 4
    assert payload["selection_version"] == 3
    assert payload["archive_provider"] == "ESA Euclid Science Archive"
    assert payload["archive_environment"] == "PDR"
    assert "public anonymous-compatible" in payload["archive_access"]
    assert "R_e,circ = R_e,major * sqrt(q)" in payload["radius_definition"]
    assert payload["vis_pixel_scale_arcsec"] == pytest.approx(0.1)
    assert all("JOIN catalogue.mer_morphology AS morph" in query for query in queries)
    radius_queries = [query for query in queries if "AS radius_bin" in query]
    fwhm_queries = [query for query in queries if "AS fwhm_bin" in query]
    assert all("COUNT(*) AS selected_radii" in query for query in radius_queries)
    assert all(
        "GROUP BY magnitude_bin, radius_bin" in query
        for query in radius_queries
    )
    assert all("COUNT(*) AS selected_fwhm" in query for query in fwhm_queries)
    assert all(
        "GROUP BY magnitude_bin, fwhm_bin" in query for query in fwhm_queries
    )
    assert all("mer.fwhm >= 0.5" in query for query in fwhm_queries)
    assert all("mer.fwhm < 3" in query for query in fwhm_queries)
    circularized = (
        "(morph.sersic_sersic_vis_radius * "
        "SQRT(morph.sersic_sersic_vis_axis_ratio))"
    )
    assert all(f"LOG10({circularized})" in query for query in radius_queries)
    assert all(f"{circularized} >= 0.03" in query for query in queries)
    assert all(f"{circularized} < 10" in query for query in queries)
    assert all("mer.vis_det = 1" in query for query in queries)
    assert all("mer.flux_vis_sersic > 0" in query for query in queries)
    assert all("mer.spurious_flag = 0" in query for query in queries)
    assert all("mer.det_quality_flag < 4" in query for query in queries)
    assert all("mer.flag_vis" not in query for query in queries)
    assert all("sersic_sersic_vis_axis_ratio > 0.05" in query for query in queries)
    assert all("sersic_sersic_vis_axis_ratio < 1.0" in query for query in queries)
    assert all("sersic_sersic_vis_index > 0.302" in query for query in queries)
    assert all("sersic_sersic_vis_index < 5.45" in query for query in queries)
    assert all("< 0.2 * mer.semimajor_axis" in query for query in queries)
    # Preserve compact sources: do not adopt the optional lower R_e/a cut.
    assert all("0.001 * mer.semimajor_axis" not in query for query in queries)
    assert all("SELECT TOP" not in query for query in queries)
    assert progress[-1][:2] == (2, 2)
    assert radius_stats.read_q1_galaxy_radius_statistics()["complete"] is True


def test_radius_statistics_rejects_stale_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    path = radius_stats.q1_galaxy_radius_statistics_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"version": 0}))

    with pytest.raises(ValueError, match="stale|first"):
        radius_stats.read_q1_galaxy_radius_statistics()


def test_radius_statistics_payload_can_be_rebuilt_from_grouped_bins():
    payload = radius_stats._build_radius_statistics_payload(
        [
            {
                "magnitude_bin": 0,
                "radius_bin": 1,
                "selected_radii": 5,
                "expected_radii": 4.25,
            },
        ],
        [
            {
                "magnitude_bin": 0,
                "fwhm_bin": 1,
                "selected_fwhm": 5,
                "expected_fwhm": 4.25,
            },
        ],
        magnitude_edges=[14.0, 14.1, 14.2],
        radius_edges_arcsec=[0.03, 0.3, 3.0],
        fwhm_edges_arcsec=[0.5, 1.0, 1.5],
        progressive_stride=0.1,
        completed_radius_queries=4,
        total_radius_queries=4,
        completed_fwhm_queries=2,
        total_fwhm_queries=2,
    )

    assert payload["complete"] is True
    assert payload["joint_bins"][0]["expected_radii"] == pytest.approx(4.25)
    assert payload["magnitude_bins"][0]["selected_radii"] == 5
    assert payload["magnitude_bins"][1]["selected_radii"] == 0
    assert payload["radius_bins"][1]["expected_radii"] == pytest.approx(4.25)
    assert payload["fwhm_bins"][1]["expected_fwhm"] == pytest.approx(4.25)


def test_radius_statistics_launch_retries_after_successful_relogin(monkeypatch):
    class Job:
        @staticmethod
        def get_results():
            return [{"selected_radii": 1}]

    launches = iter([None, Job()])
    queries: list[str] = []
    relogins = 0

    def launch(query):
        queries.append(query)
        return next(launches)

    def relogin():
        nonlocal relogins
        relogins += 1
        return True

    monkeypatch.setattr(radius_stats.Euclid, "launch_job_async", launch)

    assert list(radius_stats._launch_with_relogin("SELECT 1", relogin)) == [
        {"selected_radii": 1},
    ]
    assert queries == ["SELECT 1", "SELECT 1"]
    assert relogins == 1


def test_radius_statistics_launch_allows_anonymous_public_success(monkeypatch):
    class Job:
        @staticmethod
        def get_results():
            return [{"selected_radii": 1}]

    monkeypatch.setattr(
        radius_stats.Euclid,
        "launch_job_async",
        lambda query: Job(),
    )

    assert list(radius_stats._launch_with_relogin("SELECT 1", None)) == [
        {"selected_radii": 1},
    ]
