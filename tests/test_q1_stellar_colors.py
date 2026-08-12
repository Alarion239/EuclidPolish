"""Deterministic fixed-Q1 Gaia--Euclid colour acquisition tests."""

from __future__ import annotations

import json

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
