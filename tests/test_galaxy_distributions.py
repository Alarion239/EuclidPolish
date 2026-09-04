from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from flask import Flask

from euclid_polish.config import Config
from euclid_polish.web.app import create_app
from euclid_polish.web.helpers import galaxy_distributions as helper
from euclid_polish.web.routes import galaxy_distributions as routes


def test_curve_is_surface_density_per_parameter_width():
    curve = helper._curve(
        np.asarray([0.0, 1.0, 3.0]),
        np.asarray([20.0, 40.0]),
        10.0,
        "test counts",
    )

    assert curve["x"] == [0.5, 2.0]
    assert curve["density"] == [2.0, 2.0]
    assert curve["weighted_count"] == 60.0
    assert curve["definition"] == "test counts"


def test_observed_count_support_integrates_partial_boundary_bin():
    trust_boundary = {"kind": "empirical_5sigma", "magnitude": 25.25}
    support = helper._observed_count_support(
        [
            {"mag_lo": 24.0, "mag_hi": 25.0, "density_arcmin2_mag": 2.0},
            {"mag_lo": 25.0, "mag_hi": 26.0, "density_arcmin2_mag": 4.0},
        ],
        trust_boundary,
    )

    assert support["observed_density_cap_arcmin2_mag"] == 4.0
    assert support["observed_density_cap_magnitude"] == 25.5
    assert support[
        "observed_cumulative_density_to_boundary_arcmin2"
    ] == pytest.approx(3.0)
    assert support[
        "observed_cumulative_density_all_queried_bins_arcmin2"
    ] == pytest.approx(6.0)
    assert support["trust_boundary"] == trust_boundary


def test_radius_plot_spans_one_hundred_native_vis_pixels():
    expected_arcsec = 100.0 * float(Config.VIS_PIXEL_SCALE_ARCSEC)
    radius = helper._empty_parameters()["radius"]

    assert helper.RADIUS_MAX_VIS_PIXELS == 100.0
    assert pytest.approx(expected_arcsec) == helper.RADIUS_MAX_ARCSEC
    assert pytest.approx(np.log10(expected_arcsec)) == (
        helper.LOG_RADIUS_EDGES[-1]
    )
    assert pytest.approx(np.log10(expected_arcsec)) == radius["x_domain"][-1]
    assert "100 native VIS pixels" in radius["note"]


def test_clean_image_half_light_measurement_uses_pixels_and_rejects_blends():
    yy, xx = np.indices((81, 81), dtype=np.float64)
    image = np.exp(-0.5 * ((xx - 40) ** 2 + (yy - 40) ** 2) / 3.0**2)
    rows = [{
        "type": "galaxy", "x_pix": 40.0, "y_pix": 40.0,
        "flux_vis_e": float(np.sum(image)), "re_arcsec": 0.18,
    }]

    measured = helper._measure_field_half_light_radii(
        image, rows, pixel_scale_arcsec=0.05,
    )

    assert set(measured) == {0}
    assert measured[0] == pytest.approx(0.18, abs=0.04)


def test_actual_synthetic_catalogue_draws_are_added_to_every_parameter(
    tmp_path, monkeypatch,
):
    source = tmp_path / "sources_test.csv"
    source.write_text(
        "field_index,type,render,x_pix,y_pix,flux_vis_e,flux_y_e,"
        "flux_j_e,flux_h_e,z,re_arcsec,target_logmass,target_logssfr,"
        "achieved_vis_2fwhm_mag\n"
        "0,galaxy,tng,10,10,1000,900,800,700,0.8,0.2,9.5,-9.8,24.0\n"
        "0,galaxy,tng,20,20,500,450,400,350,1.2,0.1,8.8,-10.2,25.0\n"
    )
    monkeypatch.setattr(helper, "_synthetic_paths", lambda: ([], [source]))

    parameters = helper._empty_parameters()
    result = helper._read_synthetic(parameters)

    assert result["available"] is True
    assert result["rows"] == 2
    assert result["fields"] == 1
    assert result["measured_radius_rows"] == 0
    for key in ("redshift", "stellar_mass", "specific_sfr"):
        assert parameters[key]["series"]["synthetic"]["weighted_count"] == 2
    brightness = parameters["magnitude"]["photometry_series"]
    assert brightness["synthetic_vis_2fwhm"]["weighted_count"] == 2
    assert brightness["synthetic_vis_total"]["survey"] == "synthetic"
    radius = parameters["radius"]["radius_series"]
    assert radius["synthetic_requested_re"]["weighted_count"] == 2
    assert radius["synthetic_clean_half_light"]["weighted_count"] == 0


def test_off_field_galaxies_do_not_enter_distribution_counts(
    tmp_path, monkeypatch,
):
    source = tmp_path / "sources_test.csv"
    source.write_text(
        "field_index,type,off_field,render,x_pix,y_pix,flux_vis_e,z,re_arcsec,"
        "target_logmass,target_logssfr,achieved_vis_2fwhm_mag\n"
        "0,galaxy,0,tng,10,10,1000,0.8,0.2,9.5,-9.8,24.0\n"
        "0,galaxy,1,tng,-4,10,900,0.9,0.3,9.6,-9.7,24.2\n"
    )
    monkeypatch.setattr(helper, "_synthetic_paths", lambda: ([], [source]))

    parameters = helper._empty_parameters()
    result = helper._read_synthetic(parameters)

    assert result["available"] is True
    assert result["fields"] == 1
    assert result["rows"] == 1
    assert parameters["redshift"]["series"]["synthetic"][
        "weighted_count"
    ] == 1


def test_training_catalog_is_optional_and_never_substitutes_total_vis_for_2fwhm(
    tmp_path, monkeypatch,
):
    test_source = tmp_path / "sources_test.csv"
    train_source = tmp_path / "sources_train.csv"
    header = (
        "field_index,type,flux_vis_e,z,re_arcsec,logmass,"
        "target_vis_2fwhm_mag,mag_vis,mag_y_e,mag_j_e,mag_h_e\n"
    )
    test_source.write_text(
        header + "0,galaxy,1000,0.8,0.2,9.5,24.0,23.5,23.0,22.8,22.6\n"
    )
    train_source.write_text(
        header + "0,galaxy,500,1.1,0.4,9.0,,25.0,24.5,24.3,24.1\n"
    )
    (tmp_path / "clean_train.tfrecord").write_bytes(
        b"this training image record must never be opened"
    )

    def paths(*, include_training=False):
        sources = [test_source]
        if include_training:
            sources.append(train_source)
        return [], sources

    monkeypatch.setattr(helper, "_synthetic_paths", paths)
    parameters = helper._empty_parameters()
    result = helper._read_synthetic(
        parameters,
        include_training=True,
        measure_clean_images=True,
    )

    assert result["training_included"] is True
    assert result["training_catalog_only"] is True
    assert result["fields"] == 2
    assert result["rows"] == 2
    assert result["parameter_coverage"]["requested_re"]["splits"] == [
        "train", "test",
    ]
    assert result["parameter_coverage"]["vis_2fwhm"]["splits"] == ["test"]
    assert result["_joint_area_arcmin2"] == pytest.approx(
        helper.FIELD_AREA_ARCMIN2,
    )
    assert "not substituted with total VIS" in result["_joint_detail"]
    assert (
        parameters["radius"]["radius_series"]["synthetic_requested_re"]
        ["weighted_count"]
    ) == 2
    assert (
        parameters["magnitude"]["photometry_series"]["synthetic_vis_2fwhm"]
        ["weighted_count"]
    ) == 1


def test_q1_radius_plot_exposes_normalized_clean_shape(monkeypatch):
    payload = {
        "complete": True,
        "completed_queries": 2,
        "total_queries": 2,
        "magnitude_bins": [{}, {}],
        "footprint_area_deg2": 63.1,
        "selection": "clean circularized fixture",
        "acquisition": "aggregate fixture",
        "radius_bins": [
            {
                "radius_lo_arcsec": 0.1,
                "radius_hi_arcsec": 0.2,
                "density_arcmin2_dex": 2.0,
                "expected_radii": 20.0,
            },
            {
                "radius_lo_arcsec": 0.2,
                "radius_hi_arcsec": 0.8,
                "density_arcmin2_dex": 1.0,
                "expected_radii": 10.0,
            },
        ],
    }
    monkeypatch.setattr(
        helper, "read_q1_galaxy_radius_statistics", lambda: payload,
    )

    parameters = helper._empty_parameters()
    state = helper._read_q1_radius_statistics(parameters)
    curves = parameters["radius"]["radius_series"]
    shape = curves["euclid_sersic_re_shape"]
    widths = np.log10([0.2 / 0.1, 0.8 / 0.2])

    assert state["available"] is True
    assert curves["euclid_sersic_re"]["default_on"] is False
    assert shape["default_on"] is True
    assert shape["normalization"] == "probability_density"
    assert np.sum(np.asarray(shape["density"]) * widths) == pytest.approx(1.0)


def test_joint_maps_compare_q1_and_actual_synthetic_draws_on_one_grid(
    monkeypatch,
):
    q1 = {
        "magnitude_edges": [20.0, 21.0, 22.0, 23.0, 24.0],
        "radius_edges_arcsec": [0.03, 0.1, 0.3, 1.0],
        "footprint_area_arcmin2": 10.0,
        "joint_bins": [
            {
                "magnitude_bin": magnitude_bin,
                "radius_bin": radius_bin,
                "expected_radii": float(
                    1 + 5 * (magnitude_bin + 1) * (radius_bin + 1)
                ),
            }
            for magnitude_bin in range(4)
            for radius_bin in range(3)
        ],
    }
    monkeypatch.setattr(
        helper, "read_q1_galaxy_radius_statistics", lambda: q1,
    )
    monkeypatch.setattr(helper, "joint_galaxy_candidate", lambda: None)
    synthetic = {
        "area_arcmin2": 2.0,
        "_joint_vis_2fwhm_mag": [20.2, 20.8, 21.4, 22.1, 22.8, 23.6],
        "_joint_re_arcsec": [0.05, 0.08, 0.12, 0.2, 0.4, 0.8],
    }

    result = helper._joint_magnitude_radius_maps(synthetic)

    assert result["available"] is True
    assert [item["key"] for item in result["maps"]] == ["q1", "synthetic"]
    assert result["magnitude_edges"] == q1["magnitude_edges"]
    assert result["log_radius_edges"] == pytest.approx(
        np.log10(q1["radius_edges_arcsec"]),
    )
    assert "arcmin⁻²" in result["density_unit"]
    assert result["contour_mass_fractions"] == [
        0.999, 0.995, 0.99, 0.95, 0.80, 0.50, 0.10,
    ]
    assert result["maps"][0]["color"] == "#737373"
    assert result["maps"][1]["color"] == "#0072b2"
    assert result["maps"][1]["contour_smoothing_sigma_bins"] == 1.0
    assert "one-bin smoothing" in result["maps"][1]["detail"]
    assert not any(key.startswith("_joint") for key in synthetic)
    for item in result["maps"]:
        density = np.asarray(item["density"])
        assert density.shape == (4, 3)
        assert item["surface_density_arcmin2"] > 0.0
        assert item["contours"]
        assert all(contour["paths"] for contour in item["contours"])


def test_smoothed_joint_map_keeps_outer_tail_contours_distinct():
    magnitude_edges = np.linspace(20.0, 28.0, 41)
    log_radius_edges = np.linspace(-1.5, 0.5, 31)
    magnitude_center = 0.5 * (
        magnitude_edges[:-1] + magnitude_edges[1:]
    )
    log_radius_center = 0.5 * (
        log_radius_edges[:-1] + log_radius_edges[1:]
    )
    x, y = np.meshgrid(
        magnitude_center, log_radius_center, indexing="ij",
    )
    mass = np.exp(
        -0.5 * ((x - 25.0) / 1.2) ** 2
        -0.5 * ((y + 0.5 + 0.12 * (x - 25.0)) / 0.22) ** 2
    )

    result = helper._joint_map(
        key="synthetic",
        label="generated",
        detail="test",
        color="#0072b2",
        magnitude_edges=magnitude_edges,
        log_radius_edges=log_radius_edges,
        cell_mass_arcmin2=mass,
        contour_smoothing_sigma_bins=1.0,
    )

    fractions = {
        contour["mass_fraction"] for contour in result["contours"]
    }
    assert {0.99, 0.995, 0.999} <= fractions


def test_euclid_aperture_growth_compares_all_vis_apertures_to_total_proxies(
    tmp_path, monkeypatch,
):
    catalog = tmp_path / "euclid.csv"
    meta = tmp_path / "euclid.json"
    fieldnames = (
        "object_id", "type", "point_like_prob", "spurious_prob", "mag_vis",
        "semimajor_axis", "ellipticity", "fwhm",
        "kron_radius",
        "flux_vis_1fwhm_aper_uJy", "flux_vis_2fwhm_aper_uJy",
        "flux_vis_3fwhm_aper_uJy", "flux_vis_4fwhm_aper_uJy",
        "fluxerr_vis_4fwhm_aper_uJy", "flux_detection_total_uJy",
        "flux_vis_sersic_uJy", "vis_det", "det_quality_flag",
    )
    with catalog.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "object_id": "extended", "type": "galaxy", "point_like_prob": 0.1,
            "spurious_prob": 0.0, "mag_vis": 20.0,
            "semimajor_axis": 10.0, "ellipticity": 0.0, "fwhm": 0.5,
            "kron_radius": 14.0,
            "flux_vis_1fwhm_aper_uJy": 40.0,
            "flux_vis_2fwhm_aper_uJy": 60.0,
            "flux_vis_3fwhm_aper_uJy": 80.0,
            "flux_vis_4fwhm_aper_uJy": 100.0,
            "fluxerr_vis_4fwhm_aper_uJy": 5.0,
            "flux_detection_total_uJy": 125.0,
            "flux_vis_sersic_uJy": 150.0,
            "vis_det": 1, "det_quality_flag": 0,
        })
    meta.write_text(json.dumps({"area_arcmin2": 1.0, "rows": 1}))
    monkeypatch.setattr(helper, "euclid_catalog_path", lambda: catalog)
    monkeypatch.setattr(helper, "euclid_catalog_meta_path", lambda: meta)
    monkeypatch.setattr(
        helper, "read_phz_pdf_cache", lambda: (_ for _ in ()).throw(OSError()),
    )

    parameters = helper._empty_parameters()
    source = helper._read_euclid(parameters, lambda *_: None)
    curves = source["aperture_growth"]["curves"]
    scatter = source["aperture_scatter"]

    assert curves["1fwhm_minus_4fwhm"]["median"][6] == pytest.approx(1.0, abs=0.03)
    assert curves["3fwhm_minus_4fwhm"]["median"][6] == pytest.approx(0.24, abs=0.03)
    assert curves["4fwhm_minus_kron"]["median"][6] == pytest.approx(0.24, abs=0.03)
    assert curves["4fwhm_minus_sersic"]["median"][6] == pytest.approx(0.44, abs=0.03)
    assert scatter["count"] == 1
    assert scatter["magnitudes"]["f4"] == pytest.approx([18.9])
    assert scatter["growth"]["g1"] == pytest.approx([-2.5 * np.log10(0.4)])
    assert scatter["growth"]["g2"] == pytest.approx([-2.5 * np.log10(0.6)])
    assert scatter["growth"]["g3"] == pytest.approx([-2.5 * np.log10(0.8)])
    assert "EXTENDED_FLAG galaxies" in scatter["selection"]
    brightness = parameters["magnitude"]["photometry_series"]
    assert set(brightness) == set(helper.MER_BRIGHTNESS_SERIES)
    assert brightness["mer_vis_1fwhm"]["band"] == "Euclid VIS"
    assert brightness["mer_vis_1fwhm"]["weighted_count"] == pytest.approx(0.9)
    assert brightness["mer_vis_kron"]["estimator"] == "Kron total aperture"
    radius = parameters["radius"]["radius_series"]
    assert set(radius) == {"euclid_detection", "euclid_kron"}
    assert radius["euclid_detection"]["label"] == "Euclid · detection a"
    assert radius["euclid_detection"]["weighted_count"] == pytest.approx(0.9)
    assert radius["euclid_kron"]["weighted_count"] == pytest.approx(0.9)
    occupied = np.asarray(radius["euclid_kron"]["x"])[
        np.asarray(radius["euclid_kron"]["density"]) > 0
    ]
    assert occupied.tolist() == pytest.approx([np.log10(1.4)], abs=0.06)


def test_euclid_aperture_scatter_uses_phz_galaxies_when_extended_flag_is_unset(
    tmp_path, monkeypatch,
):
    catalog = tmp_path / "euclid.csv"
    meta = tmp_path / "euclid.json"
    fieldnames = (
        "object_id", "type", "point_like_prob", "spurious_prob", "mag_vis",
        "semimajor_axis", "ellipticity", "fwhm", "phz_gal_prob",
        "flux_vis_1fwhm_aper_uJy", "flux_vis_2fwhm_aper_uJy",
        "flux_vis_3fwhm_aper_uJy", "flux_vis_4fwhm_aper_uJy",
        "fluxerr_vis_2fwhm_aper_uJy", "fluxerr_vis_4fwhm_aper_uJy",
    )
    with catalog.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "object_id": "phz-galaxy", "type": "unknown",
            "point_like_prob": 0.05, "spurious_prob": 0.0,
            "mag_vis": 21.0, "semimajor_axis": 2.0,
            "ellipticity": 0.2, "fwhm": 0.5, "phz_gal_prob": 0.9,
            "flux_vis_1fwhm_aper_uJy": 10.0,
            "flux_vis_2fwhm_aper_uJy": 20.0,
            "flux_vis_3fwhm_aper_uJy": 30.0,
            "flux_vis_4fwhm_aper_uJy": 40.0,
            "fluxerr_vis_2fwhm_aper_uJy": 0.02,
            "fluxerr_vis_4fwhm_aper_uJy": 2.0,
        })
        writer.writerow({
            "object_id": "phz-star", "type": "unknown",
            "point_like_prob": 0.05, "spurious_prob": 0.0,
            "mag_vis": 21.0, "semimajor_axis": 2.0,
            "ellipticity": 0.2, "fwhm": 0.5, "phz_gal_prob": 0.1,
            "flux_vis_1fwhm_aper_uJy": 10.0,
            "flux_vis_2fwhm_aper_uJy": 20.0,
            "flux_vis_3fwhm_aper_uJy": 30.0,
            "flux_vis_4fwhm_aper_uJy": 40.0,
            "fluxerr_vis_2fwhm_aper_uJy": 0.03,
            "fluxerr_vis_4fwhm_aper_uJy": 2.0,
        })
    meta.write_text(json.dumps({"area_arcmin2": 1.0, "rows": 2}))
    monkeypatch.setattr(helper, "euclid_catalog_path", lambda: catalog)
    monkeypatch.setattr(helper, "euclid_catalog_meta_path", lambda: meta)
    monkeypatch.setattr(
        helper, "read_phz_pdf_cache", lambda: (_ for _ in ()).throw(OSError()),
    )

    parameters = helper._empty_parameters()
    source = helper._read_euclid(parameters, lambda *_: None)

    assert source["aperture_scatter"]["count"] == 1
    assert "PHZ_GAL_PROB >= 0.5" in source["aperture_scatter"]["selection"]
    boundary = parameters["magnitude"]["photometry_series"][
        "mer_vis_2fwhm"
    ]["trust_boundary"]
    assert boundary["kind"] == "empirical_5sigma"
    assert boundary["sample_size"] == 1
    assert boundary["magnitude"] == pytest.approx(
        float(helper.uJy_to_ab_mag(5.0 * 0.02))
    )
    assert "not an extended-galaxy completeness guarantee" in (
        boundary["caveat"]
    )


def test_euclid_radius_plot_adds_clean_phz_mer_sersic_re(
    tmp_path, monkeypatch,
):
    catalog = tmp_path / "euclid.csv"
    meta = tmp_path / "euclid.json"
    fieldnames = (
        "object_id", "type", "point_like_prob", "spurious_prob", "mag_vis",
        "semimajor_axis", "ellipticity", "kron_radius", "phz_gal_prob",
        "morph_sersic_vis_radius_arcsec", "morph_sersic_visnir_flags",
    )
    with catalog.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "object_id": "clean", "type": "galaxy", "point_like_prob": 0.1,
            "spurious_prob": 0.0, "mag_vis": 21.0,
            "semimajor_axis": 3.0, "ellipticity": 0.2,
            "kron_radius": 4.0, "phz_gal_prob": 0.8,
            "morph_sersic_vis_radius_arcsec": 0.35,
            "morph_sersic_visnir_flags": 0,
        })
        writer.writerow({
            "object_id": "flagged", "type": "galaxy",
            "point_like_prob": 0.1, "spurious_prob": 0.0, "mag_vis": 21.0,
            "semimajor_axis": 3.0, "ellipticity": 0.2,
            "kron_radius": 4.0, "phz_gal_prob": 0.9,
            "morph_sersic_vis_radius_arcsec": 1.2,
            "morph_sersic_visnir_flags": 1,
        })
    meta.write_text(json.dumps({
        "area_arcmin2": 2.0, "rows": 2, "catalog_version": 7,
    }))
    monkeypatch.setattr(helper, "euclid_catalog_path", lambda: catalog)
    monkeypatch.setattr(helper, "euclid_catalog_meta_path", lambda: meta)
    monkeypatch.setattr(
        helper, "read_phz_pdf_cache", lambda: (_ for _ in ()).throw(OSError()),
    )

    parameters = helper._empty_parameters()
    helper._read_euclid(parameters, lambda *_: None)

    curve = parameters["radius"]["radius_series"]["euclid_sersic_re"]
    assert curve["label"] == "Euclid PHZ/MER · VIS Sérsic Rₑ"
    assert curve["radius_type"] == "half_light"
    assert curve["weighted_count"] == pytest.approx(0.8)
    assert parameters["radius"]["radius_missing"] == []
    occupied = np.asarray(curve["x"])[np.asarray(curve["density"]) > 0]
    assert occupied.tolist() == pytest.approx([np.log10(0.35)], abs=0.06)


def test_euclid_radius_plot_explains_stale_cache_without_sersic_re(
    tmp_path, monkeypatch,
):
    catalog = tmp_path / "euclid.csv"
    meta = tmp_path / "euclid.json"
    catalog.write_text(
        "object_id,type,point_like_prob,spurious_prob,mag_vis\n"
        "old,galaxy,0.1,0.0,21.0\n"
    )
    meta.write_text(json.dumps({
        "area_arcmin2": 1.0, "rows": 1, "catalog_version": 6,
    }))
    monkeypatch.setattr(helper, "euclid_catalog_path", lambda: catalog)
    monkeypatch.setattr(helper, "euclid_catalog_meta_path", lambda: meta)
    monkeypatch.setattr(
        helper, "read_phz_pdf_cache", lambda: (_ for _ in ()).throw(OSError()),
    )

    parameters = helper._empty_parameters()
    helper._read_euclid(parameters, lambda *_: None)

    assert "euclid_sersic_re" not in parameters["radius"]["radius_series"]
    assert "version-7" in parameters["radius"]["radius_missing"][0]


def test_cosmos_brightness_keeps_native_estimators_separate(tmp_path, monkeypatch):
    rows = 1100
    path = tmp_path / "cosmos.npz"
    base = np.linspace(20.0, 27.0, rows)
    aperture = np.stack([base + offset for offset in (0.8, 0.5, 0.3, 0.15, 0.05)], axis=1)
    np.savez_compressed(
        path,
        mag_hst_f814w=base,
        mag_auto_hst_f814w=base + 0.1,
        mag_bd_hst_f814w=base - 0.1,
        mag_bulge_hst_f814w=base + 1.0,
        mag_disk_hst_f814w=base + 0.3,
        mag_aper_hst_f814w=aperture,
        mag_native_aper_hst_f814w=aperture + 0.2,
        z_phot=np.full(rows, 1.0),
        re_combined_arcsec=np.full(rows, 0.2),
        logssfr_lephare=np.full(rows, -9.5),
        logmass_lephare=np.full(rows, 9.5),
    )
    monkeypatch.setattr(Config, "COSMOS_POPULATION_PRIOR_PATH", str(path))

    parameters = helper._empty_parameters()
    helper._read_cosmos(parameters)
    brightness = parameters["magnitude"]["photometry_series"]

    assert len(brightness) == 15
    assert brightness["cosmos_f814w_native_aper_1"]["label"] == "F814W · 0.1″ diameter"
    assert brightness["cosmos_f814w_homogenized_aper_4"]["label"] == "F814W · 0.75″ diameter"
    assert brightness["cosmos_f814w_bd_total"]["estimator"] == "SE++ bulge+disk-model total"
    assert all(curve["band"] == "HST/ACS F814W" for curve in brightness.values())
    radius = parameters["radius"]["radius_series"]
    assert radius["cosmos_re"]["label"] == "COSMOS · combined Rₑ"
    assert radius["cosmos_re"]["weighted_count"] == pytest.approx(rows)


def test_q1_bright_counts_extend_aperture_curves_to_fourteen(monkeypatch):
    payload = {
        "edges": [14.0, 14.1, 14.2],
        "footprint_area_deg2": 63.1,
        "bright": 14.0,
        "faint": 14.2,
        "bin_width": 0.1,
        "query_count": 8,
        "completed_queries": 8,
        "total_queries": 8,
        "complete": True,
        "phases_completed": 2,
        "phase_count": 2,
        "selection": (
            "EXTENDED_FLAG = 1; POINT_LIKE_FLAG IS NULL; test selection"
        ),
        "apertures": {
            key: {
                "label": f"VIS · {index} FWHM",
                "estimator": f"{index}-FWHM diameter aperture",
                "selected_galaxies": 3,
                "expected_galaxies": 2.5,
                "bins": [
                    {
                        "mag_lo": 14.0, "mag_hi": 14.1,
                        "density_arcmin2_mag": 0.01,
                    },
                    {
                        "mag_lo": 14.1, "mag_hi": 14.2,
                        "density_arcmin2_mag": 0.02,
                    },
                ],
            }
            for index, key in enumerate(("f1", "f2", "f3", "f4"), 1)
        },
    }
    monkeypatch.setattr(
        helper,
        "read_q1_galaxy_aperture_counts",
        lambda: payload,
    )
    monkeypatch.setattr(
        helper,
        "read_q1_galaxy_aperture_fit",
        lambda: {
            "scope": "apparent-brightness aperture curves only",
            "apertures": {
                "f2": {
                    "label": "VIS · 2 FWHM",
                    "estimator": "2-FWHM diameter aperture",
                    "x": [14.05, 14.10, 14.15],
                    "density": [0.01, 0.015, 0.02],
                    "law": {
                        "fit_bright": 14.05, "fit_faint": 14.15,
                        "mag_bright": 14.0, "mag_faint": 29.0,
                    },
                    "extrapolated_faint_interval": [28.0, 29.0],
                },
            },
        },
    )

    parameters = helper._empty_parameters()
    parameters["magnitude"]["photometry_series"]["mer_vis_2fwhm"] = {
        "trust_boundary": {
            "kind": "empirical_5sigma",
            "magnitude": 14.12,
        },
    }
    state = helper._read_q1_bright_counts(parameters)
    brightness = parameters["magnitude"]["photometry_series"]

    assert state["available"] is True
    assert set(brightness) == {
        "mer_vis_2fwhm",
        "q1_vis_f1", "q1_vis_f2", "q1_vis_f3", "q1_vis_f4",
        "q1_fit_vis_f2",
    }
    assert brightness["q1_vis_f1"]["x"] == pytest.approx([14.05, 14.15])
    assert brightness["q1_vis_f4"]["density"] == pytest.approx([0.01, 0.02])
    assert brightness["q1_vis_f1"]["weighted_count"] == pytest.approx(2.5)
    assert brightness["q1_vis_f1"]["default_on"] is False
    assert brightness["q1_vis_f2"]["default_on"] is True
    assert brightness["q1_fit_vis_f2"]["default_on"] is True
    assert brightness["q1_fit_vis_f2"]["survey"] == "fit"
    assert brightness["q1_vis_f2"][
        "observed_density_cap_arcmin2_mag"
    ] == pytest.approx(0.02)
    assert brightness["q1_vis_f2"][
        "observed_density_cap_magnitude"
    ] == pytest.approx(14.15)
    assert brightness["q1_vis_f2"][
        "observed_cumulative_density_to_boundary_arcmin2"
    ] == pytest.approx(0.0014)
    assert brightness["q1_fit_vis_f2"]["trust_boundary"] == (
        brightness["q1_vis_f2"]["trust_boundary"]
    )
    assert state["fit_available"] is True


def test_fit_plot_adds_continuous_generation_and_separate_radius_shapes(
    monkeypatch,
):
    parameters = helper._empty_parameters()
    parameters["magnitude"]["photometry_series"]["q1_fit_vis_f2"] = {}
    parameters["magnitude"]["photometry_series"]["q1_vis_f2"] = {
        "trust_boundary": {
            "kind": "empirical_5sigma",
            "magnitude": 25.53,
        },
        "observed_density_cap_arcmin2_mag": 30.2,
        "observed_density_cap_magnitude": 25.55,
        "observed_cumulative_density_to_boundary_arcmin2": 46.5,
        "observed_cumulative_density_all_queried_bins_arcmin2": 74.4,
    }
    candidate = {
        "version": 12,
        "fingerprint": "f" * 64,
        "validated": True,
        "fitted_magnitude_law": {
            "fit_bright": 19.0, "fit_faint": 25.0, "slope": 0.4,
        },
        "magnitude_law": {
            "kind": (
                "continuous_three_slope_bright_bridge_main_flat_faint_counts"
            ),
            "bright_join_magnitudes": [16.4, 19.0, 20.9],
            "bright_slopes": [1.3, 0.7, 0.5],
            "straight_law": {"slope": 0.4},
        },
        "radius_law": {"log_radius_min": -1.5, "log_radius_max": 0.5},
        "magnitude_plot": {"generation_law": {
            "x": [14.0, 26.35, 29.0],
            "density": [0.01, 100.0, 100.0],
        }},
        "generation": {
            "surface_density_arcmin2": 372.83,
            "differential_density_cap_arcmin2_mag": 100.0,
            "break_magnitude": 26.35,
            "vis_magnitude_min": 14.0,
            "vis_magnitude_max": 29.0,
        },
        "plots": {"radius": {
            "x": [-1.0, 0.0],
            "density": [40.0, 60.0],
            "q1_weighted_density": [80.0, 20.0],
        }},
    }
    monkeypatch.setattr(helper, "joint_galaxy_candidate", lambda: candidate)
    monkeypatch.setattr(
        helper,
        "joint_galaxy_state",
        lambda: {"is_active": True, "active": candidate},
    )

    state = helper._read_fit(parameters)
    brightness = parameters["magnitude"]["photometry_series"][
        "generator_vis_f2"
    ]
    radius = parameters["radius"]["radius_series"]

    assert state["available"] is True
    assert brightness["generation_interval"] == [14.0, 29.0]
    assert brightness["generation_bright_join_magnitudes"] == [
        16.4, 19.0, 20.9,
    ]
    assert brightness["generation_bright_slopes"] == [1.3, 0.7, 0.5]
    assert brightness["generation_main_slope"] == 0.4
    assert brightness["generation_break_magnitude"] == 26.35
    assert brightness["generation_density_cap_arcmin2_mag"] == 100.0
    assert brightness["density"][-2:] == [100.0, 100.0]
    assert brightness["trust_boundary"]["magnitude"] == 25.53
    assert brightness["observed_density_cap_arcmin2_mag"] == 30.2
    assert brightness[
        "observed_cumulative_density_to_boundary_arcmin2"
    ] == 46.5
    assert radius["fit_re"]["weighted_count"] == 372.83
    assert radius["fit_re"]["default_on"] is False
    q1_shape = radius["fit_re_q1_weighted_shape"]
    full_shape = radius["fit_re_full_generation_shape"]
    assert q1_shape["density"] == pytest.approx([0.8, 0.2])
    assert full_shape["density"] == pytest.approx([0.4, 0.6])
    assert q1_shape["normalization"] == "probability_density"
    assert full_shape["normalization"] == "probability_density"
    assert q1_shape["radius_type"] == "half_light_shape"
    assert "Q1-magnitude-weighted" in q1_shape["label"]
    assert "faint extension" in full_shape["label"]


def test_build_writes_compact_artifact_transactionally(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(helper, "_inputs", lambda: {"fixture": 1})
    monkeypatch.setattr(
        helper,
        "_read_euclid",
        lambda parameters, progress: {"available": True, "rows": 3},
    )
    monkeypatch.setattr(
        helper,
        "_read_cosmos",
        lambda parameters: {"available": True, "rows": 4},
    )
    monkeypatch.setattr(
        helper,
        "_read_fit",
        lambda parameters: {"available": True, "fingerprint": "abc"},
    )
    monkeypatch.setattr(
        helper,
        "_read_synthetic",
        lambda parameters, progress: {"available": True, "rows": 2},
    )
    monkeypatch.setattr(
        helper,
        "_joint_magnitude_radius_maps",
        lambda synthetic: {"available": True, "maps": []},
    )
    monkeypatch.setattr(helper, "_training_variant", lambda *_args: None)

    payload = helper.build_galaxy_distributions()
    stored = json.loads(helper.artifact_path().read_text())

    assert stored == payload
    assert stored["sources"]["euclid"]["rows"] == 3
    assert stored["joint_maps"]["available"] is True
    assert not helper.artifact_path().with_suffix(".json.tmp").exists()
    assert helper.read_galaxy_distributions()["stale"] is False


def test_read_overlays_progressive_q1_checkpoints_without_full_rebuild(
    tmp_path, monkeypatch,
):
    artifact = tmp_path / "galaxy_distributions.json"
    parameters = helper._empty_parameters()
    parameters["magnitude"]["photometry_series"] = {
        "q1_vis_f1": {"x": [14.05], "density": [1.0]},
        "cosmos_fixture": {"x": [20.0], "density": [2.0]},
    }
    artifact.write_text(json.dumps({
        "version": helper.ARTIFACT_VERSION,
        "inputs": {"fixture": 1},
        "sources": {},
        "parameters": parameters,
    }))
    monkeypatch.setattr(helper, "artifact_path", lambda: artifact)
    monkeypatch.setattr(helper, "_inputs", lambda: {"fixture": 1})

    def overlay(current):
        assert "q1_vis_f1" not in current["magnitude"]["photometry_series"]
        current["magnitude"]["photometry_series"]["q1_vis_f4"] = {
            "x": [14.15], "density": [3.0],
        }
        return {"available": True, "completed_queries": 2}

    monkeypatch.setattr(helper, "_read_q1_bright_counts", overlay)
    monkeypatch.setattr(
        helper, "_read_fit", lambda _parameters: {"available": False}
    )

    result = helper.read_galaxy_distributions()

    series = result["parameters"]["magnitude"]["photometry_series"]
    assert set(series) == {"cosmos_fixture", "q1_vis_f4"}
    assert result["q1_counts"]["completed_queries"] == 2


def test_status_route_returns_plot_payload(monkeypatch):
    app = Flask(__name__)
    monkeypatch.setattr(
        routes,
        "read_galaxy_distributions",
        lambda **_kwargs: {"version": 1, "stale": False, "parameters": {}},
    )
    monkeypatch.setattr(routes, "availability", lambda: {"synthetic": {}})
    monkeypatch.setattr(routes, "_q1_counts_state", lambda: None)
    monkeypatch.setattr(
        routes.euclid_session,
        "is_authenticated",
        lambda: True,
    )
    routes.register(app)

    response = app.test_client().get("/api/galaxy-distributions")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["stale"] is False
    assert payload["authenticated"] is True
    assert payload["availability"] == {"synthetic": {}}
    assert "q1_stars" not in payload
    assert "stellar_colors" not in payload


def test_galaxy_distribution_plate_route_serves_inline_svg(monkeypatch):
    app = Flask(__name__)
    monkeypatch.setattr(
        routes, "read_galaxy_distributions", lambda **_kwargs: {"version": 17},
    )
    monkeypatch.setattr(
        routes,
        "render_galaxy_distribution_plate",
        lambda payload, **kwargs: b"<svg>fixture</svg>",
    )
    routes.register(app)

    response = app.test_client().get(
        "/view/galaxy-distribution-plate?format=svg&inline=1",
    )

    assert response.status_code == 200
    assert response.mimetype == "image/svg+xml"
    assert response.get_data().startswith(b"<svg>")
    assert "attachment" not in response.headers.get("Content-Disposition", "")


def test_recover_phz_route_is_removed():
    app = Flask(__name__)
    routes.register(app)

    response = app.test_client().post("/api/galaxy-distributions/recover-phz")

    assert response.status_code == 404


def test_q1_galaxy_query_requires_login(monkeypatch):
    app = Flask(__name__)
    monkeypatch.setattr(routes.euclid_session, "catalog", lambda: None)
    routes.register(app)

    response = app.test_client().post(
        "/api/galaxy-distributions/query-q1-counts"
    )

    assert response.status_code == 400
    assert "Log in" in response.get_json()["error"]


def test_query_mer_phz_runs_only_galaxy_brackets_and_fits(monkeypatch):
    app = Flask(__name__)
    events = []

    class Catalog:
        @staticmethod
        def relogin():
            return True

    class Capture:
        def tick(self, *_args):
            pass

        def write(self, _message):
            pass

    monkeypatch.setattr(routes.euclid_session, "catalog", lambda: Catalog())
    monkeypatch.setattr(
        routes,
        "query_q1_galaxy_aperture_counts",
        lambda **_kwargs: events.append("galaxy brightness") or {
            "bright": 14.0, "faint": 28.0,
            "completed_queries": 560, "total_queries": 560,
            "footprint_area_deg2": 63.1,
        },
    )
    monkeypatch.setattr(
        routes,
        "query_q1_galaxy_radius_statistics",
        lambda **_kwargs: events.append("galaxy radius") or {
            "completed_queries": 4, "total_queries": 4,
        },
    )
    monkeypatch.setattr(
        routes, "fit_q1_galaxy_aperture_counts",
        lambda: events.append("brightness fit") or {"apertures": {"f2": {}}},
    )
    monkeypatch.setattr(
        routes, "fit_euclid_joint_galaxy_candidate",
        lambda: events.append("joint fit") or {"fingerprint": "a" * 64},
    )
    monkeypatch.setattr(
        routes, "build_galaxy_distributions",
        lambda: events.append("plots") or {"version": 12},
    )

    def spawn(*, label, target):
        events.append(label)
        target(Capture())
        return "all-mer-phz"

    monkeypatch.setattr(routes.REGISTRY, "spawn", spawn)
    routes.register(app)

    response = app.test_client().post(
        "/api/galaxy-distributions/query-q1-counts"
    )

    assert response.status_code == 200
    assert response.get_json()["job_id"] == "all-mer-phz"
    assert events[:5] == [
        "galaxy distributions: MER + PHZ queries and fits",
        "galaxy brightness", "galaxy radius", "brightness fit", "joint fit",
    ]
    assert events[-1] == "plots"


def test_q1_aperture_fit_route_uses_count_cache_not_cones(monkeypatch):
    app = Flask(__name__)
    monkeypatch.setattr(
        routes,
        "_q1_counts_state",
        lambda: {"complete": False, "fit_ready": True},
    )
    monkeypatch.setattr(
        routes.REGISTRY,
        "spawn",
        lambda *args, **kwargs: "q1-aperture-fit-job",
    )
    routes.register(app)

    response = app.test_client().post(
        "/api/galaxy-distributions/fit-q1-counts"
    )

    assert response.status_code == 200
    assert response.get_json()["job_id"] == "q1-aperture-fit-job"


def test_q1_aperture_fit_route_requires_cached_counts(monkeypatch):
    app = Flask(__name__)
    monkeypatch.setattr(routes, "_q1_counts_state", lambda: None)
    routes.register(app)

    response = app.test_client().post(
        "/api/galaxy-distributions/fit-q1-counts"
    )

    assert response.status_code == 400
    assert "Q1 MER + PHZ aperture counts" in response.get_json()["error"]


def test_q1_aperture_fit_route_waits_for_four_queried_bins(monkeypatch):
    app = Flask(__name__)
    monkeypatch.setattr(
        routes,
        "_q1_counts_state",
        lambda: {"apertures": {"f4": {"bins": [{}, {}, {}]}}},
    )
    routes.register(app)

    response = app.test_client().post(
        "/api/galaxy-distributions/fit-q1-counts"
    )

    assert response.status_code == 400
    assert "Zero-count bins are allowed" in response.get_json()["error"]


def test_galaxy_distribution_page_is_registered_in_spa():
    response = create_app().test_client().get("/galaxy-distributions")

    assert response.status_code == 200
    assert 'id="root"' in response.get_data(as_text=True)


def test_galaxy_distribution_controls_use_one_galaxy_query_action():
    source = (
        Path(__file__).parents[1]
        / "euclid_polish/web/frontend/src/pages/GalaxyDistributions.tsx"
    ).read_text()

    assert source.count('"Query MER + PHZ"') == 1
    assert "Recover PHZ locally" not in source
    assert "/api/galaxy-distributions/recover-phz" not in source
    assert "/api/population-comparison/query-euclid-multi" not in source
    assert "/api/population-comparison/fit-euclid" not in source
    assert "/api/galaxy-distributions/fit-q1-counts" not in source
    assert "Fit cached aperture curves" not in source
    assert '"/api/galaxy-distributions/activate"' in source
    assert "Rₑ brackets" in source
    assert "stellar bins" not in source
    assert "stellar colours" not in source
    assert "POINT_LIKE_FLAG IS NULL" in source
    assert "never refreshes star caches" in source
    assert "Joint brightness–radius relation" in source
    assert "observed_mean_log10_arcsec" in source
    assert "model_core_low_log10_arcsec" in source
    assert "model_core_high_log10_arcsec" in source
    assert "?? relation.model_low_log10_arcsec" in source
    assert "?? relation.model_high_log10_arcsec" in source
    assert "VIS magnitude–MER aperture FWHM relation" in source
    assert "observed_mean_arcsec" in source
    assert "model_mean_arcsec" in source
    assert "MER catalogue FWHM (arcsec)" in source
    assert "conditionalFwhmInterval" in source
    assert "errorLow: interval.low" in source
    assert "errorHigh: interval.high" in source
    assert "weighted 16th–84th" in source
    assert "nearest populated bin where direct Q1 support is absent" in source
    assert "generation_bright_join_magnitudes" in source
    assert "generation_bright_slopes" in source
    assert "three-segment bright bridge/main/flat" in source
    assert "one fitted straight truncated-Gaussian conditional law" in source
    assert "Q1 magnitude mix" in source
    assert "full faint extension" in source
    assert "half_light_shape" in source
    assert "normalized probability / dex" in source
    assert 'const PARAMETER_ORDER = ["magnitude", "radius"]' in source
    assert "ApertureLadder" not in source
    assert "JointDensityMaps" in source
    assert "one shared Q1 plot" in source
    assert "Q1 MER + PHZ, generated, and model contours" in source
    assert "Every contour is labeled by its enclosed population mass" in source
    assert "const contourMaps = [q1, ...overlays]" in source
    assert "contourMassLabel(contour.mass_fraction)" in source
    assert "z: q1.density" not in source
    assert "neutral grayscale" not in source
    assert "Gray contours show" in source
    assert "blue dashed contours show" in source
    assert "vermillion solid contours" in source
    assert "10 / 50 / 80 / 95 / 99 / 99.5 / 99.9% contours" in source
    assert "JOINT_DENSITY_COLOR" not in source
    assert 'map.key === "synthetic" ? [7, 4]' in source
    assert "data.maps.map" not in source
    assert "include_training=${includeTraining" in source
    assert "include training catalog" in source
    assert "no training" in source.lower()
    assert "Download {format.toUpperCase()}" in source
    assert "paper figure · fixed layout" in source
    assert "broken conditional log-radius law" not in source
    assert "Added galaxies fainter than VIS 25.5" not in source
    assert "fitted mixture mean" not in source
    assert 'label="random cones"' not in source
    assert 'label="radius (arcmin)"' not in source
    assert "galaxy-q1-phases" in source

    css = (
        Path(__file__).parents[1]
        / "euclid_polish/web/frontend/src/pages/galaxy-distributions.css"
    ).read_text()
    assert (
        ".galaxy-plot-grid { display: grid; grid-template-columns: "
        "minmax(0, 1fr)"
    ) in css
    assert ".publication-plate__preview" in css


def test_logarithmic_plots_label_ticks_in_physical_units():
    source = (
        Path(__file__).parents[1]
        / "euclid_polish/web/frontend/src/pages/GalaxyDistributions.tsx"
    ).read_text()

    assert "const physicalLogTicks" in source
    assert "yTicks={physicalLogTicks(yDomain)}" in source
    assert "xTicks={physicalLogTicks(xDomain, 6)}" in source
    assert "log₁₀ ${parameter.density_unit}" not in source
    assert "(log scale)" in source
    assert "parameter.x_domain" in source
