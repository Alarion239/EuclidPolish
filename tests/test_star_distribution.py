"""Focused tests for the stellar-distribution page and plot payload."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest


def test_star_distribution_builds_all_six_measured_colours():
    from euclid_polish.web.helpers.star_population import (
        _star_distribution_from_rows,
    )

    gaia_rows = [
        {
            "source_id": str(index),
            "bp_rp": str(0.2 * index),
            "g_mag": "18.0",
            "central_selected_star": "0",
        }
        for index in range(1, 10)
    ]
    euclid_rows = []
    for index in range(1, 9):
        euclid_rows.append({
            "gaia_id": str(index),
            "type": "star",
            "mag_vis": str(20.0 + index),
            "mag_y_e": str(19.0 + index),
            "mag_j_e": str(18.5 + index),
            "mag_h_e": str(18.0 + index),
            "point_like_prob": "0.95",
            **{
                f"flux_{band}_aper_uJy": "10.0"
                for band in ("vis", "y", "j", "h")
            },
            **{
                f"fluxerr_{band}_aper_uJy": "1.0"
                for band in ("vis", "y", "j", "h")
            },
        })

    payload = _star_distribution_from_rows(
        euclid_rows,
        gaia_rows,
        calibration_fingerprint="candidate-1",
        color_model={
            "bp_rp_edges": [0.0, 0.8, 2.0],
            "bp_rp_nodes": [0.4, 1.2],
            "locus_colors": [[1.0, 0.5, 0.25], [1.2, 0.4, 0.1]],
            "g_to_vis_offset": [0.3, 0.5],
            "intrinsic_color_covariance": [
                [0.04, 0.0, 0.0],
                [0.0, 0.01, 0.0],
                [0.0, 0.0, 0.0025],
            ],
        },
    )

    assert payload["matched_stars"] == 8
    assert payload["high_quality_stars"] == 8
    assert payload["pointlike_over_0_9"] == 8
    assert payload["gaia_cmd"]["cached_stars"] == 9
    assert len(payload["gaia_cmd"]["matched"]["bp_rp"]) == 8
    assert payload["gaia_cmd"]["unmatched"]["bp_rp"] == pytest.approx([1.8])
    assert set(payload["colors"]) == {
        "vis_y", "vis_j", "vis_h", "y_j", "y_h", "j_h",
    }
    assert payload["colors"]["vis_y"]["values"] == pytest.approx([1.0] * 8)
    assert payload["colors"]["vis_h"]["values"] == pytest.approx([2.0] * 8)
    assert payload["colors"]["y_j"]["values"] == pytest.approx([0.5] * 8)
    assert payload["colors"]["vis_j"]["fit"]["center"] == pytest.approx(
        [1.5, 1.6]
    )
    assert payload["colors"]["vis_h"]["fit"]["sigma"] == pytest.approx(
        0.0525 ** 0.5
    )
    assert payload["colors"]["y_h"]["fit"]["sigma"] == pytest.approx(
        0.0125 ** 0.5
    )
    assert payload["fit_note"]
    projection = payload["euclid_projection"]
    assert len(projection["matched"]["vis_mag"]) == 8
    assert len(projection["unmatched"]["vis_mag"]) == 1
    assert projection["matched"]["vis_mag"][0] == pytest.approx(18.3)
    assert projection["matched"]["colors"]["vis_j"][0] == pytest.approx(1.5)
    assert len(projection["euclid_observed"]["vis_y"]["vis_mag"]) == 8
    assert projection["euclid_observed"]["vis_h"]["color"] == pytest.approx(
        [2.0] * 8
    )
    assert payload["density_comparison"] is None


def test_stellar_density_comparison_uses_area_density_and_all_six_colours(
    monkeypatch,
):
    from euclid_polish.web.helpers import star_population

    def no_q1_cache(**_kwargs):
        raise ValueError("not cached")

    monkeypatch.setattr(
        star_population,
        "read_q1_phz_star_counts",
        no_q1_cache,
    )

    euclid_rows = [
        {
            "type": "star", "point_like_prob": "0.95",
            "mag_vis": "20", "mag_y_e": "19",
            "mag_j_e": "18.5", "mag_h_e": "18",
        },
        {
            "type": "star", "point_like_prob": "0.5",
            "mag_vis": "21", "mag_y_e": "20",
            "mag_j_e": "19.5", "mag_h_e": "19",
        },
    ]
    projected = {
        "matched": {
            "vis_mag": [18.0],
            "colors": {"vis_y": [0.5], "vis_j": [0.8], "vis_h": [1.0]},
        },
        "unmatched": {
            "vis_mag": [19.0],
            "colors": {"vis_y": [0.7], "vis_j": [1.1], "vis_h": [1.4]},
        },
    }
    stellar_model = {
        "fingerprint": "density-test",
        "population": {
            "density_arcmin2": 2.0,
            "mag_bright": 12.0,
            "mag_faint": 25.0,
            "magnitude_distribution": {
                "kind": "straight_log10_differential_counts",
                "equation": "log10(dN_dA_dm) = slope * magnitude + intercept",
                "slope": 0.0,
                "intercept": math.log10(2.0 / 13.0),
                "mag_bright": 12.0,
                "mag_faint": 25.0,
                "fit_bright": 15.0,
                "fit_faint": 22.0,
                "covariance": [[1e-4, 0.0], [0.0, 1e-3]],
                "r_squared": 1.0,
                "rms_log10_density": 0.0,
                "surface_density_arcmin2": 2.0,
                "source": "fixture",
                "fit_diagnostics": {
                    "gaia": {
                        "intercept": -2.0,
                        "fit_bright": 12.0,
                        "fit_faint": 18.0,
                    },
                    "q1": {"fit_bright": 18.0, "fit_faint": 23.0},
                },
            },
        },
        "gaia": {
            "bp_rp_quantiles": [0.0, 2.0],
            "temperature_quantiles_k": [8000.0, 3500.0],
        },
        "euclid_mapping": {
            "g_to_band_offset_coefficients": {
                key: [0.0, 0.0, 0.0]
                for key in ("mag_vis", "mag_y_e", "mag_j_e", "mag_h_e")
            },
            "residual_covariance": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        "color_model": {
            "kind": "gaia_euclid_latent_locus_v1",
            "bp_rp_edges": [0.0, 1.0, 2.0],
            "bp_rp_nodes": [0.5, 1.5],
            "temperature_nodes_k": [7000.0, 4000.0],
            "locus_colors": [[0.2, 0.1, 0.05], [1.0, 0.2, 0.1]],
            "intrinsic_color_covariance": [
                [0.01, 0.0, 0.0],
                [0.0, 0.01, 0.0],
                [0.0, 0.0, 0.01],
            ],
            "magnitude_edges": [12.0, 25.0],
            "magnitude_node_weights": [[0.5, 0.5]],
        },
    }

    result = star_population._stellar_density_comparison(
        euclid_rows,
        [
            {"g_mag": "17.9", "central_selected_star": "0"},
            {"g_mag": "18.9", "central_selected_star": "1"},
        ],
        projected,
        stellar_model,
        euclid_area_arcmin2=10.0,
        gaia_area_arcmin2=20.0,
        sample_count=500,
    )

    assert result is not None
    assert result["euclid_vis_count"] == 1
    assert result["euclid_color_count"] == 1
    assert result["gaia_count"] == 2
    assert result["gaia_native_g_count"] == 1
    magnitude = result["parameters"]["vis"]
    bin_width = magnitude["x"][1] - magnitude["x"][0]
    assert sum(magnitude["gaia"]) * bin_width == pytest.approx(0.05)
    assert magnitude["gaia_fit"] is not None
    assert magnitude["fit_ranges"] == {
        "gaia": [12.0, 18.0], "q1": [18.0, 23.0],
    }
    assert result["q1_expected_point_sources"] is None
    assert set(result["parameters"]) == {
        "vis", "vis_y", "vis_j", "vis_h", "y_j", "y_h", "j_h",
    }
    assert sum(result["parameters"]["vis"]["model"]) * 0.5 == pytest.approx(2.0)


def test_stellar_colour_cache_rejects_legacy_random_fields():
    from euclid_polish.web.helpers.q1_stellar_colors import (
        GAIA_TAP_PROVIDER,
        Q1_STELLAR_COLOR_FIELD_RADIUS_DEG,
        Q1_STELLAR_COLOR_FIELDS,
        Q1_STELLAR_COLOR_SAMPLE_VERSION,
    )
    from euclid_polish.web.helpers.star_population import (
        _require_current_gaia_field_sampling,
    )

    meta = {
        "version": Q1_STELLAR_COLOR_SAMPLE_VERSION,
        "sampling_kind": "fixed_q1_magnitude_stratified_color_fields",
        "field_count": len(Q1_STELLAR_COLOR_FIELDS),
        "radius_deg": Q1_STELLAR_COLOR_FIELD_RADIUS_DEG,
        "tap_provider": GAIA_TAP_PROVIDER,
        "query_mode": "sync",
        "random_centres": False,
        "fields": [
            {"ra": ra, "dec": dec, "name": name}
            for ra, dec, name in Q1_STELLAR_COLOR_FIELDS
        ],
    }
    _require_current_gaia_field_sampling(meta, [])
    meta["random_centres"] = True
    with pytest.raises(ValueError, match="cache is stale"):
        _require_current_gaia_field_sampling(meta, [])


def test_star_distribution_page_and_status_route(monkeypatch):
    from euclid_polish.web.app import create_app
    from euclid_polish.web.routes import star_distribution as routes

    expected_distribution = {
        "matched_stars": 2772,
        "colors": {"vis_j": {"values": [0.4]}},
    }
    monkeypatch.setattr(
        routes, "q1_stellar_color_sample_state",
        lambda: {"cached": True, "euclid": {"rows": 12}, "gaia": {"rows": 20}},
    )
    monkeypatch.setattr(routes, "star_state", lambda: {"status": "candidate"})
    monkeypatch.setattr(routes.euclid_session, "is_authenticated", lambda: True)
    monkeypatch.setattr(routes, "_q1_counts_state", lambda: {"bins": []})
    monkeypatch.setattr(
        routes,
        "star_distribution_payload",
        lambda: expected_distribution,
    )
    client = create_app().test_client()

    page = client.get("/star-distribution")
    assert page.status_code == 200
    assert b'<div id="root">' in page.data

    status = client.get("/api/star-distribution")
    assert status.status_code == 200
    assert status.get_json() == {
        "authenticated": True,
        "color_sample": {
            "cached": True,
            "euclid": {"rows": 12},
            "gaia": {"rows": 20},
        },
        "calibration": {"status": "candidate"},
        "distribution": expected_distribution,
        "q1_counts": {"bins": []},
    }


def test_checked_in_star_bundle_matches_current_api_contract():
    """Flask must not serve a stale bundle from before color_sample existed."""
    from euclid_polish.web.app import create_app

    client = create_app().test_client()
    page = client.get("/star-distribution")
    match = re.search(
        r'<script type="module" crossorigin src="([^"]+\.js)"',
        page.get_data(as_text=True),
    )

    assert page.status_code == 200
    assert match is not None
    asset = client.get(match.group(1))
    bundle = asset.get_data(as_text=True)
    assert asset.status_code == 200
    assert "color_sample" in bundle
    assert "availability.euclid_catalog" not in bundle


def test_star_page_uses_the_single_galaxy_page_mer_phz_query():
    from euclid_polish.web.app import create_app
    source = (
        Path(__file__).parents[1]
        / "euclid_polish/web/frontend/src/pages/StarDistribution.tsx"
    ).read_text()

    assert create_app().test_client().post(
        "/api/star-distribution/query-q1-counts"
    ).status_code == 404
    assert 'to="/galaxy-distributions"' in source
    assert "Open Query MER + PHZ" in source
    assert "/api/star-distribution/query-q1-counts" not in source
    assert "Random Euclid population cones" not in source


def test_cached_stellar_fit_does_not_query_gaia_or_require_euclid_login(monkeypatch):
    from euclid_polish.web.app import create_app
    from euclid_polish.web.routes import star_distribution as routes

    class Capture:
        def tick(self, *_args):
            pass

        def write(self, _message):
            pass

    calls = []
    monkeypatch.setattr(
        routes, "q1_stellar_color_sample_state",
        lambda: {"cached": True},
    )
    monkeypatch.setattr(
        routes,
        "read_q1_phz_star_counts",
        lambda: {"expected_stars": 12.0, "footprint_area_deg2": 63.1},
    )
    monkeypatch.setattr(
        routes,
        "fit_star_population",
        lambda: {"euclid_mapping": {"matched_stars": 8}},
    )

    def spawn(*, label, target):
        calls.append(label)
        target(Capture())
        return "gaia-job"

    monkeypatch.setattr(routes.REGISTRY, "spawn", spawn)
    monkeypatch.setattr(
        routes.euclid_session,
        "catalog",
        lambda: pytest.fail("Gaia query must not require Euclid login"),
    )

    response = create_app().test_client().post("/api/star-distribution/fit")

    assert response.status_code == 200
    assert response.get_json() == {"ok": True, "job_id": "gaia-job"}
    assert calls == ["star distribution: fit cached stellar colours"]
