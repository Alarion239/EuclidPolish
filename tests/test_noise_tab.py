"""The Noise tab: a read-only view of the committed Q1 noise-level table."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.sky.observation.mer_noise_levels import load_mer_noise_levels
from euclid_polish.web.app import create_app
from euclid_polish.web.helpers.noise_levels import (
    jittered_counts,
    log10_edges,
    noise_levels_payload,
    seam_splits,
    seam_step,
)

ROOT = Path(__file__).parents[1]


def test_summary_medians_are_the_band_constants():
    payload = noise_levels_payload()
    assert payload["bands"] == ["VIS", "Y_E", "J_E", "H_E"]
    for band in payload["bands"]:
        assert payload["summary"][band]["median"] == pytest.approx(
            Config.get_band(band).mer_rms_e, rel=0.01,
        )


def test_histograms_count_every_position_once():
    payload = noise_levels_payload()
    positions = payload["source"]["position_count"]
    assert positions == len(load_mer_noise_levels().levels_e) == len(payload["positions"])
    assert sum(field["positions"] for field in payload["fields"]) == positions
    for band in payload["bands"]:
        histogram = payload["histograms"][band]
        counts = np.array(list(histogram["counts_by_field"].values()))
        assert counts.shape[1] == len(histogram["log10_edges"]) - 1
        assert counts.sum() == positions
        # Each bin is rounded to 4 decimals in the payload.
        assert sum(histogram["jittered_counts"]) == pytest.approx(positions, abs=0.01)


def test_within_field_seams_match_the_committed_sub_grids():
    payload = noise_levels_payload()
    within = payload["within_field"]
    assert within["grid_side"] == 4
    assert within["cutout_arcsec"] == pytest.approx(25.6)
    assert within["sub_tile_arcsec"] == pytest.approx(6.4)
    for band in payload["bands"]:
        stats = within["bands"][band]
        # Nearly every position splits; a few have too much unobserved sky.
        positions = payload["source"]["position_count"]
        assert 0.97 * positions <= stats["fields"] <= positions
        assert sum(stats["counts"]) == stats["fields"]
        # Clean seams are the minority the generator's strip stands for.
        assert 0.0 < stats["seam_rate"] < 0.25
        assert stats["seam_count"] == pytest.approx(
            stats["seam_rate"] * stats["fields"], abs=1,
        )
        steps = stats["steps"]
        assert within["step_threshold"] <= steps["p50"] <= steps["p90"] <= steps["max"]


def test_seam_step_separates_a_clean_boundary_from_a_bright_source():
    """The discriminator the strip is calibrated on: a straight depth step
    counts as a seam, one hot sub-tile is a source and does not."""
    splits = seam_splits(4)
    assert seam_step([10.0] * 16, splits)[0] == pytest.approx(1.0)

    step, scatter = seam_step([10.0] * 8 + [12.0] * 8, splits)
    assert step == pytest.approx(1.2)
    assert scatter == pytest.approx(1.0)

    source = [10.0] * 16
    source[5] = 40.0
    step, scatter = seam_step(source, splits)
    assert step == pytest.approx(1.0)
    assert scatter > 1.5


def test_log_edges_cover_the_range_in_fixed_steps():
    edges = log10_edges(7.3, 138.4)
    assert 10 ** edges[0] <= 7.3 and 10 ** edges[-1] >= 138.4
    np.testing.assert_allclose(np.diff(edges), 0.02)


def test_jittered_histogram_matches_monte_carlo():
    rng = np.random.default_rng(1)
    values = rng.uniform(5.0, 20.0, size=50)
    edges = np.linspace(3.0, 25.0, 23)
    expected = jittered_counts(values, edges, 0.8, 1.2)
    draws = values[:, None] * rng.uniform(0.8, 1.2, size=(50, 20_000))
    empirical = np.histogram(draws, edges)[0] / 20_000
    np.testing.assert_allclose(expected, empirical, atol=0.05)


def test_pixel_scatter_ratio_reflects_bilinear_resampling():
    summary = noise_levels_payload()["summary"]
    assert summary["VIS"]["pixel_scatter_ratio"] == pytest.approx(2.0 / 3.0, abs=0.05)
    for band in ("Y_E", "J_E", "H_E"):
        assert summary[band]["pixel_scatter_ratio"] == pytest.approx(0.23, abs=0.04)


def test_log_correlation_is_symmetric_with_unit_diagonal():
    matrix = np.array(noise_levels_payload()["log_correlation"])
    np.testing.assert_allclose(np.diag(matrix), 1.0)
    np.testing.assert_allclose(matrix, matrix.T)


def test_noise_page_and_api_route():
    client = create_app().test_client()

    page = client.get("/realism/noise")
    assert page.status_code == 200
    assert b'<div id="root">' in page.data

    response = client.get("/api/noise")
    assert response.status_code == 200
    assert response.get_json()["source"]["release"] == "Q1_R1"


def test_noise_tab_is_registered_in_the_route_manifest():
    """The Noise view is the Realism workspace's ``noise`` tab; the old
    ``/noise`` URL redirects there (contract C1, read by Flask and the SPA)."""
    manifest = json.loads((ROOT / "euclid_polish/web/spa_routes.json").read_text())
    realism = next(w for w in manifest["workspaces"] if w["id"] == "realism")
    assert realism["path"] == "/realism"
    assert "noise" in realism["tabs"]
    assert manifest["redirects"]["/noise"] == "/realism/noise"
