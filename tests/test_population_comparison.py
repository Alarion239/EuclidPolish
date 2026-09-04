"""Focused checks for the real-versus-synthetic field-statistics workspace."""
from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.web.helpers.population_comparison import (
    CATALOG_VERSION,
    VERSION,
    _archive_fields,
    _cluster_bootstrap_indices,
    _comparison_cache_state,
    _comparison_input_state,
    _field_payload,
    _FieldAccumulator,
    _finite,
    _grouped_scalar_interval,
    _normalise_field,
    _parameter_payload,
    _read_synthetic_sources,
    _scale_similarity,
    _shared_parameter_payload,
    _synthetic_dataset_tng_prior,
    availability,
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
    assert payload["sampling"]["synthetic_independent_parents"] == 3
    assert payload["sampling"]["real_independent_parents"] == 3
    assert payload["power"]["VIS"]["real"]["independent_parents"] == 3
    assert "median_ci" in payload["power"]["VIS"]["real"]
    assert "median_ci" in payload["band_correlation"]["real"]
    assert "median_ci" in payload["summary"]["real"]["VIS"]["robust_std"]
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


def test_scale_similarity_bootstraps_parent_clusters_not_cutouts():
    k = np.geomspace(0.1, 1.0, 6)
    synthetic = [
        np.asarray([1, 2, 3, 4, 5, 6], dtype=float),
        np.asarray([2, 3, 4, 5, 6, 7], dtype=float),
    ]
    parent_one = np.asarray([3, 4, 6, 8, 10, 12], dtype=float)
    parent_two = np.asarray([1, 3, 5, 7, 9, 11], dtype=float)
    payload = _scale_similarity(
        synthetic,
        [parent_one, parent_one * 1.1, parent_two, parent_two * 0.9],
        k,
        seed=37,
        synthetic_parents=["s0", "s1"],
        real_parents=["p0", "p0", "p1", "p1"],
    )
    selected = _cluster_bootstrap_indices(
        ["p0", "p0", "p1", "p1"], np.random.default_rng(37),
    )
    counts = np.bincount(selected, minlength=4)

    assert counts[0] == counts[1]
    assert counts[2] == counts[3]
    assert payload["bootstrap"] == {
        "draws": 256,
        "unit": "parent cluster",
        "synthetic_parents": 2,
        "real_parents": 2,
    }


def test_field_interval_keeps_empirical_spread_and_clusters_median_ci():
    values = [0.0, 10.0, 20.0, 30.0]
    interval = _grouped_scalar_interval(
        values, ["p0", "p0", "p1", "p1"], seed=19,
    )

    assert interval["median"] == pytest.approx(np.median(values))
    assert interval["p16"] == pytest.approx(np.percentile(values, 16))
    assert interval["p84"] == pytest.approx(np.percentile(values, 84))
    assert set(interval["median_ci"]) == {"p16", "p84"}


def test_comparison_fingerprint_changes_when_equal_count_input_bytes_change(
    monkeypatch, tmp_path,
):
    from euclid_polish.web.helpers import population_comparison as comparison

    record = tmp_path / "dirty_test.tfrecord"
    source = tmp_path / "sources_test.csv"
    record.write_bytes(b"first")
    source.write_text("field_index,type\n0,galaxy\n")

    def paths(*, include_training=False):
        del include_training
        return [record], [source]

    monkeypatch.setattr(comparison, "_synthetic_paths", paths)
    archive = {
        "ready": True,
        "collection_fingerprint": "a" * 64,
        "manifest_fingerprint": "m" * 64,
    }
    before = _comparison_input_state(archive)
    record.write_bytes(b"other")
    after = _comparison_input_state(archive)

    assert before["synthetic"]["records"][0]["size_bytes"] == (
        after["synthetic"]["records"][0]["size_bytes"]
    )
    assert before["synthetic"]["records"][0]["sha256"] != (
        after["synthetic"]["records"][0]["sha256"]
    )
    assert before["fingerprint"] != after["fingerprint"]


def test_comparison_cache_freshness_uses_input_fingerprint(
    monkeypatch, tmp_path,
):
    from euclid_polish.web.helpers import population_comparison as comparison

    cache = tmp_path / "comparison.json"
    cache.write_text(json.dumps({
        "version": VERSION,
        "provenance": {"input_fingerprint": "old"},
    }))
    monkeypatch.setattr(comparison, "comparison_path", lambda: cache)
    state = _comparison_cache_state({
        "fingerprint": "new",
        "archive": {"ready": True, "collection_fingerprint": "archive"},
    })

    assert state["fresh"] is False
    assert state["reason"] == "comparison inputs changed"


def test_availability_exposes_multipoint_readiness_and_exact_fingerprint(
    monkeypatch, tmp_path,
):
    from euclid_polish.web.helpers import population_comparison as comparison

    record = tmp_path / "dirty_test.tfrecord"
    source = tmp_path / "sources_test.csv"
    record.write_bytes(b"")
    source.write_text("field_index,type\n0,galaxy\n")
    monkeypatch.setattr(
        comparison,
        "_synthetic_paths",
        lambda **_kwargs: ([record], [source]),
    )
    monkeypatch.setattr(
        comparison,
        "_archive_collection_state",
        lambda: {
            "available": True,
            "valid": True,
            "ready": True,
            "complete": True,
            "current": True,
            "sample_count": 220,
            "planned_sample_count": 220,
            "parent_count": 44,
            "fields": {"EDF-F": 95, "EDF-N": 80, "EDF-S": 45},
            "bands": ["VIS", "Y_E", "J_E", "H_E"],
            "tile_size": 256,
            "collection_fingerprint": "a" * 64,
            "manifest_fingerprint": "b" * 64,
            "source_release": "Q1",
            "source_plan_fingerprint": "c" * 64,
            "source_manifest_sha256": "d" * 64,
        },
    )
    monkeypatch.setattr(
        comparison, "comparison_path", lambda: tmp_path / "missing.json",
    )

    state = availability()

    assert state["real"]["fields"] == 220
    assert state["real"]["independent_parents"] == 44
    assert state["real"]["ready"] is True
    assert state["real"]["collection_fingerprint"] == "a" * 64
    assert len(state["input_fingerprint"]) == 64
    assert state["comparison_cache"]["fresh"] is False


def test_population_field_normalisation_accepts_fits_plane_order():
    cube = np.arange(4 * 256 * 256, dtype=np.float32).reshape(4, 256, 256)
    normalized = _normalise_field(cube)
    assert normalized.shape == (255, 255, 4)
    np.testing.assert_array_equal(normalized[..., 2], cube[2, :255, :255])


def test_archive_field_provider_preserves_parent_and_sample_metadata(monkeypatch):
    from euclid_polish.web.helpers import population_comparison as comparison

    class Sample:
        sample_id = 12
        source_sample_id = 3
        parent_id = "archive-parent-3"
        field = "EDF-N"
        ra = 17.2
        dec = 66.1
        source_release = "Q1"
        source_plan_fingerprint = "p" * 64

    class Provider:
        @staticmethod
        def iter_fields(manifest):
            assert manifest == {"kind": "euclid_archive_fields"}
            yield Sample()

        @staticmethod
        def load_field(sample, manifest):
            assert isinstance(sample, Sample)
            assert manifest == {"kind": "euclid_archive_fields"}
            return np.ones((256, 256, 4), dtype=np.float32)

    monkeypatch.setattr(comparison, "_archive_provider", lambda: Provider)
    fields = list(_archive_fields({"kind": "euclid_archive_fields"}))

    assert len(fields) == 1
    field, metadata = fields[0]
    assert field.shape == (256, 256, 4)
    assert metadata["sample_id"] == 12
    assert metadata["parent_id"] == "archive-parent-3"
    assert metadata["field"] == "EDF-N"


def test_build_requires_ready_multipoint_archive_without_legacy_fallback(
    monkeypatch, tmp_path,
):
    from euclid_polish.web.helpers import population_comparison as comparison

    record = tmp_path / "dirty_test.tfrecord"
    record.write_bytes(b"")
    monkeypatch.setattr(
        comparison,
        "_synthetic_paths",
        lambda **_kwargs: ([record], []),
    )
    monkeypatch.setattr(
        comparison,
        "_archive_collection_state",
        lambda: {
            "ready": False,
            "reasons": ["archive_fields_manifest.json is missing"],
            "sample_count": 0,
        },
    )

    with pytest.raises(FileNotFoundError, match="multipoint Euclid archive"):
        comparison.build_comparison()


def test_generic_comparison_has_no_legacy_single_point_or_nexus_discovery():
    source = (
        Path(__file__).parents[1]
        / "euclid_polish/web/helpers/population_comparison.py"
    ).read_text()

    assert "EUCLID_INFERENCE_DIR" not in source
    assert "jwst_euclid_overlap" not in source
    assert "nexus_fields" not in source


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
    current = {"synthetic_field_count": 2}
    with_training = {"synthetic_field_count": 12}
    refreshed_inputs = {
        "fingerprint": "new-input-fingerprint",
        "archive": {"collection_fingerprint": "archive-a"},
        "synthetic": {
            "records": [{"role": "dirty_test", "sha256": "a"}],
            "current_sources": [{"role": "sources_test", "sha256": "b"}],
            "training_sources": [{"role": "sources_train", "sha256": "new"}],
        },
    }
    original_payload = json.loads(comparison_file.read_text())
    original_payload["provenance"] = {
        "input_fingerprint": "old-input-fingerprint",
        "inputs": {
            "archive": {"collection_fingerprint": "archive-a"},
            "synthetic": {
                "records": [{"role": "dirty_test", "sha256": "a"}],
                "current_sources": [{"role": "sources_test", "sha256": "b"}],
                "training_sources": [{"role": "sources_train", "sha256": "old"}],
            },
        },
    }
    comparison_file.write_text(json.dumps(original_payload))
    monkeypatch.setattr(comparison, "comparison_path", lambda: comparison_file)
    monkeypatch.setattr(
        comparison,
        "_population_variants",
        lambda field_count: (current, with_training),
    )
    monkeypatch.setattr(comparison, "_comparison_input_state", lambda: refreshed_inputs)

    refreshed = refresh_population_comparison()
    saved = json.loads(comparison_file.read_text())

    assert refreshed is not None
    assert refreshed["synthetic_field_count"] == 2
    assert "euclid" not in refreshed
    assert saved["fields"] == {"sentinel": "unchanged"}
    assert saved["population"] == refreshed
    assert saved["population_with_training"] == with_training
    assert saved["provenance"]["input_fingerprint"] == "new-input-fingerprint"
    assert saved["provenance"]["inputs"] == refreshed_inputs
    assert saved["provenance"]["population_refreshed_at"]


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


def test_population_variants_only_switch_the_synthetic_census(monkeypatch):
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

    def fake_payload(paths, field_count):
        return {
            "synthetic_field_count": len(paths),
        }

    monkeypatch.setattr(comparison, "_population_payload", fake_payload)
    current, with_training = comparison._population_variants(200)

    assert current["synthetic_field_count"] == 2
    assert current["training_included"] is False
    assert with_training["synthetic_field_count"] == 3
    assert with_training["training_included"] is True
    assert "tng_prior" not in current
    assert "tng_prior" not in with_training
    assert "calibration_splits" not in current
    assert "calibration_splits" not in with_training


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
            "flux_vis_1fwhm_aper": 7.0, "fluxerr_vis_1fwhm_aper": 0.5,
            "flux_vis_2fwhm_aper": 10.0, "fluxerr_vis_2fwhm_aper": 0.7,
            "flux_vis_3fwhm_aper": 12.0, "flux_y_3fwhm_aper": 10.0,
            "flux_j_3fwhm_aper": 8.0, "flux_h_3fwhm_aper": 6.0,
            "flux_vis_4fwhm_aper": 13.0, "fluxerr_vis_4fwhm_aper": 1.0,
            "flux_detection_total": 14.0, "fluxerr_detection_total": 1.1,
            "flux_vis_sersic": 14.5, "fluxerr_vis_sersic": 1.2,
            "vis_det": 1, "det_quality_flag": 0,
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
            "flux_vis_3fwhm_aper": 2.0,
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
    for multiple in (1, 2, 3, 4):
        assert f"flux_h_{multiple}fwhm_aper" in captured["query"]
    written = list(csv.DictReader(catalog_path.open()))
    assert [row["type"] for row in written] == ["star", "galaxy", "unknown"]
    assert float(written[0]["vis_y_color"]) != 0
    assert float(written[0]["flux_vis_1fwhm_aper_uJy"]) == 7.0
    assert float(written[0]["flux_vis_4fwhm_aper_uJy"]) == 13.0
    assert float(written[0]["flux_vis_aper_uJy"]) == 12.0
    assert float(written[0]["flux_detection_total_uJy"]) == 14.0
    assert float(written[0]["flux_vis_sersic_uJy"]) == 14.5


def test_euclid_population_query_retains_curated_mer_and_morphology_schema(
    tmp_path, monkeypatch,
):
    from euclid_polish.web.helpers import population_comparison as comparison

    row = {
        "object_id": 7,
        "right_ascension": 10.0,
        "declination": 20.0,
        "point_like_prob": 0.05,
        "extended_prob": 0.95,
        "flux_vis_psf": 5.0,
        "fluxerr_vis_psf": 0.5,
        "flux_vis_3fwhm_aper": 5.0,
        "fluxerr_vis_3fwhm_aper": 0.5,
    }
    row.update({
        key: float(index + 1)
        for index, key in enumerate(comparison.EUCLID_MER_STUDY_COLUMNS)
    })
    row.update({
        key: float(index + 101)
        for index, key in enumerate(
            comparison.EUCLID_MORPHOLOGY_STUDY_COLUMNS
        )
    })
    # Preserve meaningful classifier flags after filling the numeric contract.
    row.update({
        "point_like_flag": None,
        "extended_flag": 1,
        "spurious_flag": 0,
        "deblended_flag": 1,
        "flag_vis": 0,
    })

    class FakeJob:
        def get_results(self):
            return [row]

    queries = []
    catalog_path = tmp_path / "euclid_population.csv"
    meta_path = tmp_path / "euclid_population_meta.json"
    monkeypatch.setattr(
        comparison.Euclid, "launch_job_async",
        lambda query: queries.append(query) or FakeJob(),
    )

    meta = query_euclid_population(
        10.0, 20.0, 1.0,
        _catalog_path=catalog_path,
        _meta_path=meta_path,
    )
    query = queries[0]
    written = next(csv.DictReader(catalog_path.open()))

    assert "LEFT OUTER JOIN catalogue.mer_morphology AS morph" in query
    for cache_name, archive_name in comparison.EUCLID_MER_STUDY_COLUMNS.items():
        assert f"mer.{archive_name} AS {cache_name}" in query
        assert cache_name in written
    for cache_name, archive_name in (
        comparison.EUCLID_MORPHOLOGY_STUDY_COLUMNS.items()
    ):
        assert f"morph.{archive_name} AS {cache_name}" in query
        assert cache_name in written
    assert float(written["morph_sersic_vis_radius_arcsec"]) > 0.0
    assert float(written["morph_disk_sersic_disk_radius_arcsec"]) > 0.0
    assert meta["study_schema"]["morphology_table"] == (
        "catalogue.mer_morphology"
    )


def test_euclid_population_query_preserves_cache_on_silent_archive_failure(
    tmp_path, monkeypatch,
):
    from euclid_polish.web.helpers import population_comparison as comparison

    catalog_path = tmp_path / "euclid_population.csv"
    meta_path = tmp_path / "euclid_population_meta.json"
    catalog_path.write_text("object_id,type,mag_vis\nold,galaxy,22\n")
    meta_path.write_text('{"sentinel": "old"}')
    monkeypatch.setattr(
        comparison.Euclid, "launch_job_async", lambda _query: None,
    )
    monkeypatch.setattr(comparison, "euclid_catalog_path", lambda: catalog_path)
    monkeypatch.setattr(
        comparison, "euclid_catalog_meta_path", lambda: meta_path,
    )

    with pytest.raises(RuntimeError, match="failed before returning a job"):
        query_euclid_population(10.0, 20.0, 1.0)

    assert catalog_path.read_text() == "object_id,type,mag_vis\nold,galaxy,22\n"
    assert json.loads(meta_path.read_text()) == {"sentinel": "old"}


def test_euclid_population_query_compacts_phz_pdf_and_physical_columns(
    tmp_path, monkeypatch,
):
    from euclid_polish.web.helpers import population_comparison as comparison

    pdf = np.exp(-0.5 * ((comparison.PHZ_PDF_GRID - 1.2) / 0.15) ** 2)
    rows = [{
        "object_id": 42,
        "right_ascension": 10.0,
        "declination": 20.0,
        "point_like_flag": None,
        "point_like_prob": 0.05,
        "extended_flag": 1,
        "flux_vis_psf": 5.0,
        "fluxerr_vis_psf": 0.5,
        "flux_vis_3fwhm_aper": 5.0,
        "fluxerr_vis_3fwhm_aper": 0.5,
        "phz_star_prob": 0.01,
        "phz_gal_prob": 0.98,
        "phz_qso_prob": 0.08,
        "phz_classification": 2,
        "phz_pdf": pdf,
        "phz_median": 1.2,
        "phz_flags": 0,
        "phz_phys_flags": 0,
        "phz_phys_quality_flag": 0,
        "phz_pp_median_redshift": 1.19,
        "phz_pp_68_redshift": np.asarray([1.05, 1.34]),
        "phz_pp_median_stellarmass": 10.2,
        "phz_pp_68_stellarmass": np.asarray([10.0, 10.4]),
        "phz_pp_median_sfr": 0.2,
        "phz_pp_68_sfr": np.asarray([0.0, 0.4]),
        "phz_pp_median_sfhage": 2.0e9,
        "phz_pp_68_sfhage": np.asarray([1.5e9, 2.5e9]),
        "phz_pp_median_mu": -20.0,
        "phz_pp_68_mu": np.asarray([-20.2, -19.8]),
    }]

    class FakeJob:
        def get_results(self):
            return rows

    catalog_path = tmp_path / "euclid_population.csv"
    meta_path = tmp_path / "euclid_population_meta.json"
    queries = []
    monkeypatch.setattr(
        comparison.Euclid, "launch_job_async",
        lambda query: queries.append(query) or FakeJob(),
    )
    monkeypatch.setattr(comparison, "euclid_catalog_path", lambda: catalog_path)
    monkeypatch.setattr(comparison, "euclid_catalog_meta_path", lambda: meta_path)

    meta = query_euclid_population(10.0, 20.0, 1.0)
    compact = comparison.read_phz_pdf_cache(
        tmp_path / "euclid_population_phz_pdf.npz"
    )
    written = next(csv.DictReader(catalog_path.open()))

    assert "catalogue.phz_classification" in queries[0]
    assert "catalogue.phz_photo_z" in queries[0]
    assert "catalogue.phz_physical_parameters" in queries[0]
    assert compact["object_id"].tolist() == ["42"]
    assert compact["probability"].shape == (1, len(comparison.LF_Z_EDGES) - 1)
    assert compact["probability"].sum() == pytest.approx(1.0)
    assert float(written["phz_gal_prob"]) == pytest.approx(0.98)
    assert float(written["phz_pp_stellarmass_p16"]) == pytest.approx(10.0)
    assert float(written["phz_pp_sfhage_p84"]) == pytest.approx(2.5e9)
    assert float(written["phz_pp_mu_p16"]) == pytest.approx(-20.2)
    assert meta["phz_pdf_rows"] == 1
    assert meta["phz_quality"]["all_retained_pdfs_normalized"]


def test_archive_vector_parser_accepts_tap_string_arrays():
    from euclid_polish.web.helpers import population_comparison as comparison

    np.testing.assert_allclose(
        comparison._result_vector({"value": "[1.0, 2.5]"}, "value", 2),
        [1.0, 2.5],
    )
    np.testing.assert_allclose(
        comparison._result_vector({"value": b"{3.0 4.0}"}, "value", 2),
        [3.0, 4.0],
    )
    assert comparison._result_vector(
        {"value": "[1.0, missing]"}, "value", 2,
    ) is None
    assert comparison._result_vector(
        {"value": "[1.0, 2.0, 3.0]"}, "value", 2,
    ) is None


def test_cached_phz_summary_recovery_requires_no_archive_query(
    tmp_path, monkeypatch,
):
    from euclid_polish.web.helpers import population_comparison as comparison

    catalog_path = tmp_path / "euclid_population.csv"
    meta_path = tmp_path / "euclid_population_meta.json"
    pdf_path = tmp_path / "euclid_population_phz_pdf.npz"
    with catalog_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "object_id", "mag_vis", "phz_gal_prob", "phz_median",
            "point_like_prob",
            "phz_mode_1", "phz_mode_1_area", "phz_mode_2",
            "phz_mode_2_area",
        ))
        writer.writeheader()
        writer.writerows([
            {
                "object_id": "bright", "mag_vis": 22.0,
                "phz_gal_prob": 0.8, "point_like_prob": 0.1,
                "phz_median": 1.0,
                "phz_mode_1": 0.9, "phz_mode_1_area": 0.7,
                "phz_mode_2": 1.8, "phz_mode_2_area": 0.3,
            },
            {
                "object_id": "faint", "mag_vis": 25.0,
                "phz_gal_prob": 0.9, "point_like_prob": 0.1,
                "phz_median": 2.0,
            },
        ])
    meta_path.write_text(json.dumps({
        "rows": 2,
        "phz_coverage": {"phz_galaxy_weight": 0.8},
    }))
    monkeypatch.setattr(comparison, "euclid_catalog_path", lambda: catalog_path)
    monkeypatch.setattr(comparison, "euclid_catalog_meta_path", lambda: meta_path)
    monkeypatch.setattr(comparison, "euclid_phz_pdf_path", lambda: pdf_path)
    monkeypatch.setattr(
        comparison.Euclid, "launch_job_async",
        lambda _query: pytest.fail("recovery must not query the archive"),
    )

    recovered = comparison.recover_phz_pdf_cache_from_summaries()
    compact = comparison.read_phz_pdf_cache(pdf_path)

    assert compact["object_id"].tolist() == ["bright"]
    np.testing.assert_allclose(np.sum(compact["probability"], axis=1), 1.0)
    assert recovered["phz_pdf_source"] == "summary_reconstruction"
    assert recovered["phz_pdf_activation_eligible"] is False
    assert recovered["phz_pdf_recovery"]["archive_query_performed"] is False


def test_euclid_population_query_retries_after_session_refresh(
    tmp_path, monkeypatch,
):
    from euclid_polish.web.helpers import population_comparison as comparison

    rows = [{
        "object_id": 1,
        "right_ascension": 10.0,
        "declination": 20.0,
        "point_like_flag": None,
        "extended_flag": 1,
        "flux_vis_psf": 5.0,
        "flux_vis_3fwhm_aper": 5.0,
    }]

    class FakeJob:
        def get_results(self):
            return rows

    jobs = iter([None, FakeJob()])
    refreshes = []
    monkeypatch.setattr(
        comparison.Euclid, "launch_job_async", lambda _query: next(jobs),
    )
    monkeypatch.setattr(
        comparison, "euclid_catalog_path", lambda: tmp_path / "catalog.csv",
    )
    monkeypatch.setattr(
        comparison, "euclid_catalog_meta_path", lambda: tmp_path / "meta.json",
    )

    meta = query_euclid_population(
        10.0, 20.0, 1.0,
        relogin=lambda: refreshes.append(True) or True,
    )

    assert refreshes == [True]
    assert meta["rows"] == 1


def test_multi_cone_failure_preserves_complete_previous_cache(
    tmp_path, monkeypatch,
):
    from euclid_polish.web.helpers import population_comparison as comparison

    catalog_path = tmp_path / "euclid_population.csv"
    meta_path = tmp_path / "euclid_population_meta.json"
    original_catalog = "object_id,type,mag_vis\nold,galaxy,22\n"
    original_meta = '{"sentinel": "old"}'
    catalog_path.write_text(original_catalog)
    meta_path.write_text(original_meta)
    centers = [
        {"star_id": "a", "ra": 10.0, "dec": 20.0, "magnitude": 18.0},
        {"star_id": "b", "ra": 30.0, "dec": 40.0, "magnitude": 19.0},
    ]
    rows = [{
        "object_id": 1,
        "right_ascension": 10.0,
        "declination": 20.0,
        "point_like_flag": None,
        "extended_flag": 1,
        "flux_vis_psf": 5.0,
        "flux_vis_3fwhm_aper": 5.0,
    }]

    class FakeJob:
        def get_results(self):
            return rows

    jobs = iter([FakeJob(), None])
    monkeypatch.setattr(
        comparison.Euclid, "launch_job_async", lambda _query: next(jobs),
    )
    monkeypatch.setattr(
        comparison, "select_star_cone_centers", lambda **_kwargs: centers,
    )
    monkeypatch.setattr(comparison, "euclid_catalog_path", lambda: catalog_path)
    monkeypatch.setattr(
        comparison, "euclid_catalog_meta_path", lambda: meta_path,
    )

    with pytest.raises(RuntimeError, match="failed before returning a job"):
        comparison.query_euclid_population_multi(count=2, radius_arcmin=1.0)

    assert catalog_path.read_text() == original_catalog
    assert meta_path.read_text() == original_meta


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


def test_off_field_galaxies_do_not_enter_population_counts(tmp_path):
    sources = tmp_path / "sources.csv"
    sources.write_text(
        "field_index,type,off_field,mag_vis\n"
        "0,galaxy,0,21.0\n"
        "0,galaxy,1,22.0\n"
        "1,galaxy,true,23.0\n"
    )

    rows = _read_synthetic_sources([sources])

    assert len(rows) == 1
    assert rows[0]["mag_vis"] == 21.0


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
    }
    monkeypatch.setattr(routes, "availability",
                        lambda: expected_availability)
    monkeypatch.setattr(routes, "read_comparison", lambda: None)
    noise_state = {
        "candidate": {"fingerprint": "c" * 64, "valid": True},
        "active": None,
        "is_active": False,
        "candidate_is_active": False,
        "can_fit": True,
        "unavailable_reason": None,
        "sampling": {"independent_parent_count": 4},
    }
    monkeypatch.setattr(routes, "vis_noise_state", lambda: noise_state)
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
    assert payload["vis_noise_calibration"] == noise_state
    assert "calibrations" not in payload


def test_vis_noise_fit_and_activate_routes_are_separate_jobs(monkeypatch):
    from euclid_polish.web.app import create_app
    from euclid_polish.web.routes import population_comparison as routes

    events = []

    class Capture:
        def tick(self, current, total, label):
            events.append(("tick", current, total, label))

        def write(self, message):
            events.append(("write", message))

    def fit(*, progress=None):
        events.append(("fit",))
        assert progress is not None
        progress(3, 4, "held-out parent validation")
        return {"fingerprint": "f" * 64, "valid": True}

    def activate():
        events.append(("activate",))
        return {"fingerprint": "f" * 64, "valid": True, "active": True}

    def spawn(*, label, target):
        events.append(("spawn", label))
        result = target(Capture())
        events.append(("result", result))
        return f"job-{len([event for event in events if event[0] == 'spawn'])}"

    monkeypatch.setattr(routes, "fit_vis_noise_candidate", fit)
    monkeypatch.setattr(routes, "activate_vis_noise_candidate", activate)
    monkeypatch.setattr(routes.REGISTRY, "spawn", spawn)
    client = create_app().test_client()

    fitted = client.post("/api/population-comparison/fit-vis-noise")
    activated = client.post("/api/population-comparison/activate-vis-noise")

    assert fitted.status_code == 200
    assert fitted.get_json() == {"ok": True, "job_id": "job-1"}
    assert activated.status_code == 200
    assert activated.get_json() == {"ok": True, "job_id": "job-2"}
    assert ("fit",) in events
    assert ("activate",) in events
    assert (
        "spawn", "population comparison: fit VIS background noise"
    ) in events
    assert (
        "spawn", "population comparison: activate VIS background noise"
    ) in events
    assert any(
        event[:2] == ("tick", 3) and event[3] == "held-out parent validation"
        for event in events
    )


def test_vis_noise_sample_sync_installs_local_manifest(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from euclid_polish.web.app import create_app
    from euclid_polish.web.routes import population_comparison as routes

    remote_manifest = tmp_path / "remote-manifest.json"
    remote_manifest.write_text(json.dumps({
        "version": 1,
        "kind": "euclid_vis_noise_sampling",
        "source_release": "Q1_R1",
        "samples": [{
            "sample_id": 0,
            "parent_id": "parent-a",
            "status": "written",
            "output_path": (
                "/remote/data/euclid_sky/vis_noise_samples/"
                "cutouts/sky_0000.fits"
            ),
        }],
    }))
    cached_fits = tmp_path / "sky_0000.fits"
    cached_fits.write_bytes(b"FITS")
    local_manifest = tmp_path / "local" / "vis_noise_sampling_manifest.json"
    fetched = []

    def fetch(remote, **_kwargs):
        fetched.append(remote)
        path = remote_manifest if remote.endswith(".json") else cached_fits
        return SimpleNamespace(
            ok=True, local_path=str(path), error=None,
            size_bytes=path.stat().st_size,
        )

    class Capture:
        def tick(self, *_args):
            pass

        def write(self, *_args):
            pass

    monkeypatch.setattr(routes.fasrc_config, "load", lambda: SimpleNamespace(
        data_dir="/remote/data",
    ))
    monkeypatch.setattr(routes, "ensure_ssh_connected", lambda: None)
    monkeypatch.setattr(routes.fasrc_fetcher, "fetch_one_file", fetch)
    monkeypatch.setattr(
        routes, "default_sampling_manifest_path", lambda: local_manifest,
    )
    monkeypatch.setattr(
        routes.REGISTRY, "spawn", lambda *, label, target: (
            target(Capture()), "sync-job"
        )[1],
    )

    response = create_app().test_client().post(
        "/api/population-comparison/sync-vis-noise-samples"
    )

    assert response.status_code == 200
    assert response.get_json() == {"ok": True, "job_id": "sync-job"}
    installed = json.loads(local_manifest.read_text())
    assert installed["samples"][0]["status"] == "cached"
    installed_fits = local_manifest.parent / "cutouts" / "sky_0000.fits"
    assert installed["samples"][0]["output_path"] == str(installed_fits)
    assert installed_fits.read_bytes() == b"FITS"
    assert installed["samples"][0]["remote_output_path"].startswith("/remote/")
    assert installed["sync"]["completed_samples"] == 1
    assert fetched == [
        (
            "/remote/data/euclid_sky/vis_noise_samples/"
            "vis_noise_sampling_manifest.json"
        ),
        "/remote/data/euclid_sky/vis_noise_samples/cutouts/sky_0000.fits",
    ]


def test_field_statistics_has_no_population_query_or_fit_routes():
    from euclid_polish.web.app import create_app

    client = create_app().test_client()
    obsolete = (
        "/api/population-comparison/query-euclid",
        "/api/population-comparison/query-euclid-multi",
        "/api/population-comparison/fit-euclid",
        "/api/population-comparison/query-gaia-stars",
        "/api/population-comparison/activate-joint-galaxy",
        "/api/population-comparison/activate-star-prior",
    )
    assert all(client.post(endpoint).status_code == 404 for endpoint in obsolete)

    source = (
        Path(__file__).parents[1]
        / "euclid_polish/web/frontend/src/pages/PopulationComparison.tsx"
    ).read_text()
    assert "Random Euclid population cones" not in source
    assert "GalaxyCalibrationControls" not in source
    assert "StarCalibrationControls" not in source
    assert "comparison.population.euclid" not in source
    assert "the fit that connects them" not in source


def test_field_statistics_exposes_vis_noise_review_and_activation_controls():
    root = Path(__file__).parents[1] / "euclid_polish/web/frontend/src/pages"
    source = (root / "PopulationComparison.tsx").read_text()
    styles = (root / "population-comparison.css").read_text()

    assert '"/api/population-comparison/fit-vis-noise"' in source
    assert '"/api/population-comparison/activate-vis-noise"' in source
    assert '"/api/population-comparison/sync-vis-noise-samples"' in source
    assert 'stepId="vis_noise_sample"' in source
    assert "VIS background-noise calibration" in source
    assert "independent parents" in source
    assert "unmasked background" in source
    assert "Robust RMS" in source
    assert "Normalized lag covariance" in source
    assert "Source-masked angular power" in source
    assert "runtime?.source_release" in source
    assert "runtime?.owns_field_scale" in source
    assert "window.confirm" in source
    assert source.index("Fit cached VIS samples") < source.index(
        "Activate candidate"
    )
    assert ".vis-noise-ledger" in styles
    assert ".vis-noise-gates" in styles
    assert ".vis-noise-plot-grid" in styles


def test_population_comparison_status_selects_training_variant(monkeypatch):
    from euclid_polish.web.app import create_app
    from euclid_polish.web.routes import population_comparison as routes

    cached = {
        "version": VERSION,
        "population": {
            "synthetic_field_count": 200,
        },
        "population_with_training": {
            "synthetic_field_count": 6600,
        },
    }
    monkeypatch.setattr(routes, "availability", lambda: {})
    monkeypatch.setattr(routes, "read_comparison", lambda: cached)
    monkeypatch.setattr(
        routes,
        "vis_noise_state",
        lambda: {"candidate": None, "active": None, "is_active": False},
    )
    monkeypatch.setattr(
        routes.euclid_session, "is_authenticated", lambda: False
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
    assert "cosmos_euclid_fit" not in current["comparison"]["population"]
    assert "tng_prior" not in current["comparison"]["population"]
    assert "population_with_training" not in with_training["comparison"]
    assert current["vis_noise_calibration"]["is_active"] is False
