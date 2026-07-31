"""Focused checks for the real-versus-synthetic field-statistics workspace."""
from __future__ import annotations

import csv
import json
import subprocess
import warnings

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.web.helpers.population_comparison import (
    CATALOG_VERSION,
    VERSION,
    _field_payload,
    _FieldAccumulator,
    _finite,
    _normalise_field,
    _parameter_payload,
    _read_synthetic_sources,
    _shared_parameter_payload,
    _synthetic_dataset_tng_prior,
    query_euclid_population,
    refresh_population_comparison,
    select_star_cone_centers,
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


def test_star_cone_centers_are_seeded_random_and_non_overlapping(tmp_path):
    stars = tmp_path / "stars.csv"
    stars.write_text(
        "id,ra,dec,magnitude\n"
        "a,0.0,0.0,18\n"
        "b,0.1,0.0,19\n"
        "c,10.0,0.0,20\n"
        "d,20.0,0.0,21\n"
        "e,30.0,0.0,22\n"
        "f,40.0,0.0,23\n"
    )

    first = select_star_cone_centers(
        count=4, radius_arcmin=10.0, stars_csv=stars, seed=17,
    )
    replay = select_star_cone_centers(
        count=4, radius_arcmin=10.0, stars_csv=stars, seed=17,
    )

    assert first == replay
    assert len({row["star_id"] for row in first}) == 4
    assert not {"a", "b"}.issubset({row["star_id"] for row in first})


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


def test_shared_population_parameters_keep_only_comparable_observables():
    synthetic = [
        {
            "type": "galaxy", "mag_vis": 22.0, "re_arcsec": 0.3, "z": 0.8,
        },
        {
            "type": "star", "mag_vis": 18.0, "mag_y_e": 17.5,
            "vis_y_color": 0.5, "temperature_k": 5400.0,
        },
    ]
    euclid = [
        {
            "type": "unknown", "mag_vis": 23.0, "semimajor_axis": 4.0,
            "vis_snr": 12.0,
        },
        {
            "type": "star", "mag_vis": 18.5, "mag_y_e": 18.0,
            "vis_y_color": 0.5, "point_like_prob": 0.99,
        },
    ]

    payload = _shared_parameter_payload(
        synthetic, euclid, synthetic_area_arcmin2=2.0,
        euclid_area_arcmin2=4.0,
    )

    assert set(payload["parameters"]) == {
        "mag_vis", "mag_y_e", "vis_y_color",
    }
    assert set(payload["parameters"]["mag_vis"]["classes"]) == {
        "nonstellar", "star",
    }
    assert set(payload["parameters"]["mag_y_e"]["classes"]) == {"star"}
    nonstellar = payload["parameters"]["mag_vis"]["classes"]["nonstellar"]
    assert nonstellar["synthetic"]["x"] == nonstellar["euclid"]["x"]
    assert sum(nonstellar["synthetic"]["density"]) == pytest.approx(0.5)
    assert sum(nonstellar["euclid"]["density"]) == pytest.approx(0.25)
    assert "re_arcsec" not in payload["parameters"]
    assert "semimajor_axis" not in payload["parameters"]
    assert "z" not in payload["parameters"]
    assert "vis_snr" not in payload["parameters"]


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
    current = {"synthetic_field_count": 2, "euclid": {"objects": 2}}
    with_training = {
        "synthetic_field_count": 12,
        "euclid": {"objects": 2},
    }
    monkeypatch.setattr(comparison, "comparison_path", lambda: comparison_file)
    monkeypatch.setattr(
        comparison,
        "_population_variants",
        lambda field_count, detection: (current, with_training),
    )

    refreshed = refresh_population_comparison()
    saved = json.loads(comparison_file.read_text())

    assert refreshed is not None
    assert refreshed["euclid"]["objects"] == 2
    assert refreshed["synthetic_field_count"] == 2
    assert saved["fields"] == {"sentinel": "unchanged"}
    assert saved["population"] == refreshed
    assert saved["population_with_training"] == with_training


def test_synthetic_paths_exclude_training_by_default(tmp_path, monkeypatch):
    from euclid_polish.web.helpers import population_comparison as comparison

    for name in (
        "dirty_test.tfrecord",
        "dirty_validate.tfrecord",
        "sources_test.csv",
        "sources_validate.csv",
        "sources_train.csv",
    ):
        (tmp_path / name).touch()
    monkeypatch.setattr(
        comparison, "_sky_records_local_dir", lambda: str(tmp_path)
    )

    _, current = comparison._synthetic_paths()
    _, with_training = comparison._synthetic_paths(include_training=True)

    assert [path.name for path in current] == [
        "sources_test.csv",
        "sources_validate.csv",
    ]
    assert [path.name for path in with_training] == [
        "sources_test.csv",
        "sources_validate.csv",
        "sources_train.csv",
    ]


def test_population_variants_keep_calibration_on_current_splits(monkeypatch):
    from pathlib import Path

    from euclid_polish.web.helpers import population_comparison as comparison

    current_paths = [Path("sources_test.csv"), Path("sources_validate.csv")]
    all_paths = [*current_paths, Path("sources_train.csv")]
    monkeypatch.setattr(
        comparison,
        "_synthetic_paths",
        lambda *, include_training=False: (
            [],
            all_paths if include_training else current_paths,
        ),
    )

    def fake_payload(paths, field_count, detection, *, calibrate_tng_prior=True):
        return {
            "synthetic_field_count": len(paths),
            "tng_prior": "current calibration" if calibrate_tng_prior else None,
        }

    monkeypatch.setattr(comparison, "_population_payload", fake_payload)
    current, with_training = comparison._population_variants(200, {})

    assert current["synthetic_field_count"] == 2
    assert current["training_included"] is False
    assert with_training["synthetic_field_count"] == 3
    assert with_training["training_included"] is True
    assert with_training["tng_prior"] == "current calibration"
    assert with_training["calibration_splits"] == ["test", "validate"]


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


def test_synthetic_catalog_derives_shared_colours(tmp_path):
    sources = tmp_path / "sources.csv"
    sources.write_text(
        "field_index,type,mag_vis,mag_y_e,mag_j_e,mag_h_e\n"
        "0,star,20.0,19.5,19.2,19.0\n"
    )

    row = _read_synthetic_sources([sources])[0]

    assert row["vis_y_color"] == pytest.approx(0.5)
    assert row["y_j_color"] == pytest.approx(0.3)
    assert row["j_h_color"] == pytest.approx(0.2)


def test_synthetic_dataset_prior_distinguishes_legacy_and_saved_config(tmp_path):
    legacy = tmp_path / "legacy.csv"
    legacy.write_text(
        "field_index,type,render,flux_vis_e\n"
        "0,galaxy,tng,100\n"
    )
    current = tmp_path / "current.csv"
    current.write_text(
        "field_index,type,render,flux_vis_e,tng_density_arcmin2,tng_mf_alpha\n"
        "0,galaxy,tng,100,200,-1.76\n"
    )

    assert _synthetic_dataset_tng_prior(
        _read_synthetic_sources([legacy])
    ) == pytest.approx(Config.TNG_LEGACY_DATASET_DENSITY_ARCMIN2)
    assert _synthetic_dataset_tng_prior(
        _read_synthetic_sources([current])
    ) == pytest.approx(200.0)


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
    payload = status.get_json()
    assert payload["comparison"] is None
    assert payload["availability"] == expected_availability
    assert payload["authenticated"] is False
    assert set(payload["calibrations"]) == {
        "brightness_transfer", "galaxy_density", "stars",
        "galaxy_recommendation",
    }


def test_random_cone_route_accepts_one_cone_and_rejects_zero(monkeypatch):
    from euclid_polish.web.app import create_app
    from euclid_polish.web.routes import population_comparison as routes

    monkeypatch.setattr(routes.euclid_session, "catalog", lambda: object())
    monkeypatch.setattr(
        routes.REGISTRY, "spawn", lambda *args, **kwargs: "random-cones-job",
    )
    client = create_app().test_client()

    accepted = client.post(
        "/api/population-comparison/query-euclid-multi",
        data={"count": "1", "radius_arcmin": "3.5"},
    )
    rejected = client.post(
        "/api/population-comparison/query-euclid-multi",
        data={"count": "0", "radius_arcmin": "3.5"},
    )

    assert accepted.status_code == 200
    assert accepted.get_json()["job_id"] == "random-cones-job"
    assert rejected.status_code == 400
    assert "count must be 1–12" in rejected.get_json()["error"]


def test_fit_cached_cones_route_does_not_require_archive_session(monkeypatch):
    from euclid_polish.web.app import create_app
    from euclid_polish.web.routes import population_comparison as routes

    monkeypatch.setattr(
        routes,
        "availability",
        lambda: {"euclid_catalog": {"cached": True}},
    )
    monkeypatch.setattr(
        routes.REGISTRY, "spawn", lambda *args, **kwargs: "fit-cones-job",
    )
    client = create_app().test_client()

    response = client.post("/api/population-comparison/fit-euclid")

    assert response.status_code == 200
    assert response.get_json()["job_id"] == "fit-cones-job"


def test_fit_cached_cones_rebuilds_truth_without_changing_config(
    monkeypatch,
):
    from euclid_polish.web.routes import population_comparison as routes

    class Capture:
        def __init__(self):
            self.ticks = []
            self.output = []

        def tick(self, done, total, label):
            self.ticks.append((done, total, label))

        def write(self, message):
            self.output.append(message)

    fit_payload = {
        "fit": {"poisson_deviance": 10.0, "dof": 5,
                "completeness_m50": 25.0},
        "local_normalization_sensitivity_fit": {
            "poisson_deviance": 64.84,
            "dof": 10,
            "completeness_m50": 25.12,
        },
        "euclid_latent_density_estimate": {
            "use_local_normalization": True,
            "density_arcmin2": 400.3426,
        },
    }
    commands = []
    monkeypatch.setattr(
        routes.subprocess,
        "run",
        lambda argv, **kwargs: commands.append(argv),
    )
    monkeypatch.setattr(
        routes, "read_cosmos_euclid_fit", lambda: fit_payload,
    )
    monkeypatch.setattr(
        routes, "refresh_population_comparison", lambda: {"updated": True},
    )
    cap = Capture()

    result = routes._fit_and_evaluate_cached_cones(cap)

    assert [command[1] for command in commands] == [
        "scripts/fit_tng_vis_counts.py",
        "scripts/fit_cosmos_euclid_counts.py",
    ]
    assert result["euclid_latent_density_arcmin2"] == pytest.approx(400.3426)
    assert result["population_refreshed"] is True
    assert cap.ticks[-1] == (3, 3, "fit and evaluations ready")
    assert any("not a generator setting" in line for line in cap.output)
    assert any("64.84 / 10" in line for line in cap.output)


def test_analysis_script_failure_includes_stderr(monkeypatch, tmp_path):
    from euclid_polish.web.routes import population_comparison as routes

    error = subprocess.CalledProcessError(
        1,
        ["python", "broken.py"],
        stderr="ValueError: useful scientific failure",
    )
    monkeypatch.setattr(
        routes.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(error)
    )

    with pytest.raises(RuntimeError, match="useful scientific failure"):
        routes._run_analysis_script(tmp_path, "broken.py")


def test_population_comparison_status_selects_training_variant(monkeypatch):
    from euclid_polish.web.app import create_app
    from euclid_polish.web.routes import population_comparison as routes

    cached = {
        "version": VERSION,
        "population": {
            "synthetic_field_count": 200,
            "tng_prior": {"configured_prior_arcmin2": 200.0},
        },
        "population_with_training": {
            "synthetic_field_count": 6600,
            "tng_prior": {"configured_prior_arcmin2": 200.0},
        },
    }
    monkeypatch.setattr(routes, "availability", lambda: {})
    monkeypatch.setattr(routes, "read_comparison", lambda: cached)
    monkeypatch.setattr(
        routes, "read_cosmos_euclid_fit", lambda: {"version": 1}
    )
    monkeypatch.setattr(
        routes.euclid_session, "is_authenticated", lambda: False
    )
    monkeypatch.setattr(
        routes.job_config,
        "load",
        lambda: type("Config", (), {"galaxy_density_arcmin2": 320.0})(),
    )
    client = create_app().test_client()

    current = client.get("/api/population-comparison").get_json()
    with_training = client.get(
        "/api/population-comparison?include_training=1"
    ).get_json()

    assert current["comparison"]["population"]["synthetic_field_count"] == 200
    assert (
        with_training["comparison"]["population"]["synthetic_field_count"]
        == 6600
    )
    assert current["comparison"]["population"]["cosmos_euclid_fit"] == {
        "version": 1
    }
    assert (
        current["comparison"]["population"]["tng_prior"][
            "configured_prior_arcmin2"
        ]
        == 320.0
    )
    assert "population_with_training" not in with_training["comparison"]
