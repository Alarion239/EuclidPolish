"""Deterministic fixed-Q1 Gaia--Euclid colour acquisition tests."""

from __future__ import annotations

import json

import pytest

from euclid_polish.config import Config
from euclid_polish.web.helpers import q1_stellar_colors as helper


class _GaiaResult(list):
    query_status = "OK"


def test_stellar_colour_query_is_magnitude_stratified_and_fixed_field(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(Config, "STAR_MAG_BRIGHT", 12.0)
    monkeypatch.setattr(Config, "STAR_MAG_FAINT", 13.0)
    euclid_queries: list[str] = []
    gaia_queries: list[str] = []

    def launch(query, _relogin):
        euclid_queries.append(query)
        index = len(euclid_queries)
        return [{
            "object_id": index,
            "gaia_id": 100 + index,
            "point_like_prob": 0.99,
            "phz_star_prob": 0.95,
            **{
                f"flux_{band}_3fwhm_aper": 10.0 + index
                for band in ("vis", "y", "j", "h")
            },
            **{
                f"fluxerr_{band}_3fwhm_aper": 0.5
                for band in ("vis", "y", "j", "h")
            },
        }]

    class Tap:
        def __init__(self, url):
            assert url == helper.GAIA_TAP_URL

        def run_sync(self, query, *, maxrec):
            assert maxrec == helper.GAIA_SYNC_MAXREC
            gaia_queries.append(query)
            index = len(gaia_queries)
            return _GaiaResult([{
                "source_id": 100 + index,
                "ra": 1.0,
                "dec": 2.0,
                "phot_g_mean_mag": 18.0,
                "phot_bp_mean_mag": 18.5,
                "phot_rp_mean_mag": 17.5,
                "phot_g_mean_flux": 100.0,
                "phot_g_mean_flux_error": 1.0,
                "phot_bp_mean_flux": 80.0,
                "phot_bp_mean_flux_error": 1.0,
                "phot_rp_mean_flux": 120.0,
                "phot_rp_mean_flux_error": 1.0,
                "bp_rp": 1.0,
                "teff_gspphot": 5500.0,
                "ag_gspphot": 0.1,
            }])

    monkeypatch.setattr(helper, "_launch_euclid", launch)
    monkeypatch.setattr("pyvo.dal.TAPService", Tap)
    progress = []

    result = helper.query_q1_stellar_color_sample(
        progress=lambda done, total, label: progress.append(
            (done, total, label),
        ),
    )

    assert len(euclid_queries) == 2
    assert all("SELECT TOP 500" in query for query in euclid_queries)
    assert all("mer.point_like_prob >= 0.9" in query for query in euclid_queries)
    assert all(
        str(field[0]) in euclid_queries[0]
        for field in helper.Q1_STELLAR_COLOR_FIELDS
    )
    assert len(gaia_queries) == len(helper.Q1_STELLAR_COLOR_FIELDS)
    assert result["euclid"]["random_centres"] is False
    assert result["gaia"]["random_centres"] is False
    assert "cone_count" not in result["gaia"]
    assert result["euclid"]["density_role"].startswith("none")
    assert progress[-1][0:2] == (5, 5)
    assert result["euclid"]["archive"] == "ESA"
    assert helper.q1_stellar_color_sample_state()["cached"] is True
    header = helper.q1_gaia_color_catalog_path().read_text().splitlines()[0]
    assert "field_index" in header
    assert "cone_index" not in header


def test_stellar_colour_state_rejects_legacy_random_metadata(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    for path in (
        helper.q1_stellar_color_catalog_path(),
        helper.q1_gaia_color_catalog_path(),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("id\n")
    payload = {
        "version": helper.Q1_STELLAR_COLOR_SAMPLE_VERSION,
        "sampling_kind": "fixed_q1_magnitude_stratified_color_fields",
        "random_centres": True,
        "field_count": len(helper.Q1_STELLAR_COLOR_FIELDS),
        "fields": [
            {"ra": ra, "dec": dec, "name": name}
            for ra, dec, name in helper.Q1_STELLAR_COLOR_FIELDS
        ],
    }
    helper.q1_stellar_color_meta_path().write_text(json.dumps(payload))
    helper.q1_gaia_color_meta_path().write_text(json.dumps(payload))

    state = helper.q1_stellar_color_sample_state()

    assert state == {"cached": False, "euclid": None, "gaia": None}


def _gaia_tap(gaia_queries):
    class Tap:
        def __init__(self, url):
            assert url == helper.GAIA_TAP_URL

        def run_sync(self, query, *, maxrec):
            gaia_queries.append(query)
            return _GaiaResult([{
                "source_id": 100 + len(gaia_queries), "ra": 1.0, "dec": 2.0,
                "phot_g_mean_mag": 18.0,
            }])
    return Tap


def _irsa_csv_row(object_id, flux):
    """One IRSA CSV row: every value arrives as text, nulls as ''."""
    return {
        "object_id": str(object_id), "point_like_prob": "0.99",
        "phz_star_prob": "", "flux_vis_psf": str(flux), "fluxerr_vis_psf": "0.1",
        **{f"flux_{b}_3fwhm_aper": str(flux) for b in ("vis", "y", "j", "h")},
        **{f"fluxerr_{b}_3fwhm_aper": "0.5" for b in ("vis", "y", "j", "h")},
    }


def test_colour_query_falls_back_to_irsa_per_cone_when_esa_fails(
    tmp_path, monkeypatch,
):
    """ESA down → IRSA, one cone per query (IRSA rejects OR-ed CONTAINS);
    the merged cones keep ESA's TOP-N-by-object_id selection exactly."""
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(Config, "STAR_MAG_BRIGHT", 12.0)
    monkeypatch.setattr(Config, "STAR_MAG_FAINT", 12.5)
    monkeypatch.setattr(helper, "Q1_STELLAR_COLOR_ROWS_PER_BIN", 3)

    def esa_down(_query, _relogin):
        raise RuntimeError("Archive down for maintenance")

    irsa_queries: list[str] = []
    # Per-cone ids (ascending within each cone, as ORDER BY returns them);
    # the union's three smallest are -7 (cone 1), 2 and 5 (cone 0).
    cone_ids = [[2, 5, 9], [-7, 11, 12], [20, 21, 22]]

    def irsa(query):
        irsa_queries.append(query)
        return [_irsa_csv_row(i, 10.0) for i in cone_ids[len(irsa_queries) - 1]]

    monkeypatch.setattr(helper, "_launch_euclid", esa_down)
    monkeypatch.setattr(helper, "_launch_irsa", irsa)
    monkeypatch.setattr("pyvo.dal.TAPService", _gaia_tap([]))

    result = helper.query_q1_stellar_color_sample()

    assert len(irsa_queries) == len(helper.Q1_STELLAR_COLOR_FIELDS)
    for query, (ra, dec, _name) in zip(
        irsa_queries, helper.Q1_STELLAR_COLOR_FIELDS, strict=True,
    ):
        assert "euclid_q1_mer_catalogue AS mer" in query
        assert "euclid_q1_phz_classification AS cls" in query
        assert f"CIRCLE('ICRS', {ra}, {dec}," in query
        assert query.count("CONTAINS(") == 1
        assert "SELECT TOP 3" in query and "ORDER BY mer.object_id" in query
    rows = helper.q1_stellar_color_catalog_path().read_text().splitlines()
    assert [line.split(",")[0] for line in rows[1:]] == ["-7", "2", "5"]
    assert result["euclid"]["archive"] == "IRSA"
    assert result["euclid"]["rows"] == 3


def test_colour_query_reports_both_archives_when_both_fail(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(Config, "STAR_MAG_BRIGHT", 12.0)
    monkeypatch.setattr(Config, "STAR_MAG_FAINT", 12.5)

    def esa_down(_query, _relogin):
        raise RuntimeError("ESA maintenance")

    def irsa_down(_query):
        raise RuntimeError("IRSA timeout")

    monkeypatch.setattr(helper, "_launch_euclid", esa_down)
    monkeypatch.setattr(helper, "_launch_irsa", irsa_down)

    with pytest.raises(RuntimeError, match="ESA maintenance.*IRSA timeout"):
        helper.query_q1_stellar_color_sample()
    assert not helper.q1_stellar_color_catalog_path().exists()
