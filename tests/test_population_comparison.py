"""Focused checks for the real-versus-synthetic field-statistics workspace."""
from __future__ import annotations

import csv
import json
import warnings

import numpy as np
import pytest

from euclid_polish.web.helpers.population_comparison import (
    CATALOG_VERSION,
    VERSION,
    _field_payload,
    _FieldAccumulator,
    _finite,
    _normalise_field,
    _parameter_payload,
    _read_synthetic_sources,
    query_euclid_population,
    refresh_population_comparison,
)


def test_population_field_payload_is_json_safe_and_keeps_four_bands():
    rng = np.random.default_rng(17)
    synthetic = _FieldAccumulator()
    real = _FieldAccumulator()
    for _ in range(3):
        base = rng.normal(100.0, 8.0, (256, 256, 4)).astype(np.float32)
        base[0, 0, :] = 0.0
        synthetic.add(base)
        real.add(base * 1.2 + rng.normal(0, 0.5, base.shape))

    payload = _field_payload(synthetic, real)

    json.dumps(payload)
    assert payload["bands"] == ["VIS", "Y_E", "J_E", "H_E"]
    assert set(payload["histograms"]) == set(payload["bands"])
    assert len(payload["power"]["VIS"]["k"]) == 24
    similarity = payload["scale_similarity"]["VIS"]
    assert len(similarity["log_shape_ratio"]["median"]) == 24
    assert 0 <= similarity["overlap"]["median"] <= 1
    assert similarity["overlap"]["p16"] <= similarity["overlap"]["p84"]
    assert similarity["variance_ratio"]["median"] > 0
    vis_range = payload["histograms"]["VIS"]["range"]
    assert vis_range[0] == min(
        float(np.min(samples)) for samples in synthetic.samples[0] + real.samples[0]
    )
    assert vis_range[1] == max(
        float(np.max(samples)) for samples in synthetic.samples[0] + real.samples[0]
    )
    assert np.isclose(sum(payload["histograms"]["VIS"]["synthetic"]), 1.0)
    assert np.isclose(sum(payload["histograms"]["VIS"]["real"]), 1.0)
    assert payload["histograms"]["VIS"]["y_label"] == (
        "fraction of sampled pixels / bin"
    )
    assert payload["quantiles"]["VIS"]["q"][0] == 0.1
    assert payload["quantiles"]["VIS"]["q"][-1] == 99.9
    assert len(payload["relations"]["mean_std"]["VIS"]["synthetic"]["x"]) == 3
    assert len(payload["relations"]["median_robust_std"]["VIS"]["real"]["y"]) == 3
    assert len(payload["band_correlation"]["pairs"]) == 6
    assert set(payload["summary"]["synthetic"]["VIS"]) >= {
        "mean", "median", "std", "robust_std", "zero_fraction",
        "negative_fraction",
    }
    zero_bin = payload["histograms"]["VIS"]["zero_bin"]
    assert zero_bin is not None
    centers = payload["histograms"]["VIS"]["x"]
    width = centers[1] - centers[0]
    assert centers[zero_bin] - width / 2 <= 0 <= centers[zero_bin] + width / 2


def test_scale_similarity_ignores_unrelated_fourier_phase():
    rng = np.random.default_rng(21)
    synthetic = _FieldAccumulator()
    real = _FieldAccumulator()
    for index in range(4):
        base = rng.normal(0.0, 1.0, (256, 256, 4)).astype(np.float32)
        synthetic.add(base)
        real.add(np.roll(base, shift=(17 + index, 29 - index), axis=(0, 1)))

    payload = _field_payload(synthetic, real)

    for band in payload["bands"]:
        similarity = payload["scale_similarity"][band]
        assert similarity["overlap"]["median"] > 0.99
        ratio = np.asarray([
            np.nan if value is None else value
            for value in similarity["log_shape_ratio"]["median"]
        ])
        assert np.nanmedian(np.abs(ratio)) < 0.03
        assert similarity["variance_ratio"]["median"] == pytest.approx(
            1.0, rel=0.02
        )


def test_population_field_normalisation_accepts_fits_plane_order():
    cube = np.arange(4 * 256 * 256, dtype=np.float32).reshape(4, 256, 256)
    normalized = _normalise_field(cube)
    assert normalized.shape == (255, 255, 4)
    np.testing.assert_array_equal(normalized[..., 2], cube[2, :255, :255])


def test_population_parameter_payload_plots_every_available_parameter():
    rows = [
        {"field_index": 0, "type": "galaxy", "mag_vis": 21.0, "z": 0.4},
        {"field_index": 0, "type": "star", "mag_vis": 18.0,
         "temperature_k": 5400.0},
        {"field_index": 1, "type": "galaxy", "mag_vis": 22.0, "z": 0.8},
    ]
    payload = _parameter_payload(rows, area_arcmin2=2.0,
                                 include_per_field=True)

    assert payload["counts"] == {"galaxy": 2, "star": 1}
    assert payload["density_arcmin2"]["galaxy"] == 1.0
    assert {"objects_per_field", "mag_vis", "z", "temperature_k"} <= set(
        payload["parameters"]
    )


def test_masked_catalog_values_are_missing_without_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _finite(np.ma.masked) is None
    assert not caught


def test_population_refresh_preserves_field_statistics(tmp_path, monkeypatch):
    from euclid_polish.web.helpers import population_comparison as comparison

    comparison_file = tmp_path / "comparison.json"
    comparison_file.write_text(json.dumps({
        "version": VERSION,
        "samples": {"synthetic": {"fields": 2}},
        "fields": {"sentinel": "unchanged"},
        "population": {},
    }))
    sources = tmp_path / "sources.csv"
    sources.write_text(
        "field_index,type,mag_vis\n"
        "0,galaxy,21.0\n"
        "1,star,18.0\n"
    )
    monkeypatch.setattr(comparison, "comparison_path", lambda: comparison_file)
    monkeypatch.setattr(comparison, "_synthetic_paths", lambda: ([], [sources]))
    monkeypatch.setattr(comparison, "_read_euclid_sources", lambda: ([
        {"type": "galaxy", "mag_vis": 22.0},
        {"type": "star", "mag_vis": 19.0},
    ], {"area_arcmin2": 4.0, "rows": 2}))

    refreshed = refresh_population_comparison()
    saved = json.loads(comparison_file.read_text())

    assert refreshed is not None
    assert refreshed["euclid"]["objects"] == 2
    assert refreshed["synthetic_field_count"] == 2
    assert saved["fields"] == {"sentinel": "unchanged"}
    assert saved["population"] == refreshed


def test_euclid_population_query_keeps_classifier_uncertainty_and_photometry(
    tmp_path, monkeypatch
):
    from euclid_polish.web.helpers import population_comparison as comparison

    rows = [
        {
            "object_id": 1, "right_ascension": 10.0, "declination": 20.0,
            "point_like_flag": 1, "extended_flag": None,
            "point_like_prob": 0.99, "extended_prob": 0.01,
            "flux_vis_psf": 10.0, "fluxerr_vis_psf": 1.0,
            "flux_vis_3fwhm_aper": 12.0, "flux_y_3fwhm_aper": 10.0,
            "flux_j_3fwhm_aper": 8.0, "flux_h_3fwhm_aper": 6.0,
        },
        {
            "object_id": 2, "right_ascension": 10.1, "declination": 20.1,
            "point_like_flag": None, "extended_flag": 1,
            "point_like_prob": 0.02, "extended_prob": 0.98,
            "flux_vis_psf": 5.0, "fluxerr_vis_psf": 1.0,
        },
        {
            "object_id": 3, "right_ascension": 10.2, "declination": 20.2,
            "point_like_flag": None, "extended_flag": None,
            "point_like_prob": 0.45, "extended_prob": 0.55,
            "flux_vis_psf": 2.0, "fluxerr_vis_psf": 1.0,
        },
    ]
    captured = {}

    class FakeJob:
        def get_results(self):
            return rows

    def launch(query):
        captured["query"] = query
        return FakeJob()

    catalog_path = tmp_path / "euclid_population.csv"
    meta_path = tmp_path / "euclid_population_meta.json"
    monkeypatch.setattr(comparison.Euclid, "launch_job_async", launch)
    monkeypatch.setattr(comparison, "euclid_catalog_path", lambda: catalog_path)
    monkeypatch.setattr(comparison, "euclid_catalog_meta_path", lambda: meta_path)

    meta = query_euclid_population(10.0, 20.0, 1.0)

    assert meta["catalog_version"] == CATALOG_VERSION
    assert meta["counts"] == {"star": 1, "galaxy": 1, "unknown": 1}
    assert "point_like_prob" in captured["query"]
    assert "flux_h_3fwhm_aper" in captured["query"]
    written = list(csv.DictReader(catalog_path.open()))
    assert [row["type"] for row in written] == ["star", "galaxy", "unknown"]
    assert float(written[0]["vis_y_color"]) != 0


def test_synthetic_lenses_are_merged_into_galaxies(tmp_path):
    sources = tmp_path / "sources.csv"
    sources.write_text(
        "field_index,type,mag_vis,theta_E_arcsec\n"
        "0,galaxy,21.0,\n"
        "0,lens,20.0,1.2\n"
        "0,star,18.0,\n"
    )
    rows = _read_synthetic_sources([sources])
    payload = _parameter_payload(rows, area_arcmin2=1.0,
                                 include_per_field=True)

    assert payload["counts"] == {"galaxy": 2, "star": 1}
    assert "theta_E_arcsec" not in payload["parameters"]


def test_ensure_ssh_connected_builds_shared_session(monkeypatch):
    from types import SimpleNamespace

    from euclid_polish.web import remote

    created = []

    class FakeSession:
        def __init__(self, cfg):
            self.cfg = cfg
            self.connected = False
            created.append(self)

        def is_connected(self):
            return self.connected

        def connect(self):
            self.connected = True

    cfg = SimpleNamespace(
        ssh_user="astro",
        ssh_host="cluster.example",
        control_socket="/tmp/test-population-ssh.sock",
        control_persist="8h",
    )
    monkeypatch.setattr(remote.STATE, "ssh", None)
    monkeypatch.setattr(remote.STATE, "connected_at", None)
    monkeypatch.setattr(remote, "SSHSession", FakeSession)
    monkeypatch.setattr("euclid_polish.web.fasrc_config.load", lambda: cfg)

    session = remote.ensure_ssh_connected()

    assert session is created[0]
    assert session.connected
    assert remote.STATE.ssh is session
    assert remote.STATE.connected_at is not None


def test_population_comparison_page_and_status_route(monkeypatch):
    from euclid_polish.web.app import create_app
    from euclid_polish.web.routes import population_comparison as routes

    expected_availability = {
        "synthetic": {"fields": 200},
        "real": {"fields": 302},
        "field_area_arcmin2": 0.18,
        "default_cone": {"ra": 1.0, "dec": 2.0, "radius_arcmin": 3.0,
                         "area_arcmin2": 28.27},
        "euclid_catalog": {"cached": False, "meta": None},
    }
    monkeypatch.setattr(routes, "availability",
                        lambda: expected_availability)
    monkeypatch.setattr(routes, "read_comparison", lambda: None)
    monkeypatch.setattr(routes.euclid_session, "is_authenticated",
                        lambda: False)
    client = create_app().test_client()

    page = client.get("/population-comparison")
    assert page.status_code == 200
    assert b'<div id="root">' in page.data

    status = client.get("/api/population-comparison")
    assert status.status_code == 200
    assert status.get_json() == {
        "comparison": None,
        "availability": expected_availability,
        "authenticated": False,
    }
