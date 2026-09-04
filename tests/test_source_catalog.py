import os

from euclid_polish.sky.generation import source_catalog as sc
from euclid_polish.tng.renderer import TNG_RADIUS_RENDERING


def _meta():
    return {
        "galaxies": [
            {"type": "galaxy", "render": "sersic", "x_pix": 100.0, "y_pix": 120.0,
             "z_phot": 0.7, "catalog_id": 5, "flux_e_per_band": [3000.0, 1, 2, 3],
             "target_logmass": 10.3, "target_logssfr": -10.4,
             "physical_model_fingerprint": "f" * 64},
            {"type": "galaxy", "render": "tng", "x_pix": 40.0, "y_pix": 200.0,
             "z": float("nan"), "subhalo_id": 99, "orientation": 3,
             "native_tng_sfr": 0.0, "native_tng_logssfr": float("nan"),
             "native_tng_zero_sfr": True,
             "target_ssfr_quantile": 0.1, "tng_ssfr_quantile": 0.08,
             "morphology_ssfr_quantile_delta": -0.02,
             "morphology_ssfr_kernel_bandwidth_quantile": 0.2,
             "target_re_arcsec": 0.31,
             "native_halflight_px": 12.4,
             "rot_angle": 137.5, "arbitrary_rotation": True,
             "radius_scale_factor": 0.5,
             "radius_rendering": TNG_RADIUS_RENDERING,
             "radius_renderer_fingerprint": "r" * 64,
             "radius_manifest_fingerprint": "a" * 64,
             "tng_render_trace": {"subhalo_id": "99", "orientation": 3},
             "render_support_clipped": True,
             "target_vis_2fwhm_mag": 23.1,
             "achieved_vis_2fwhm_mag": 23.1,
             "mer_photometric_fwhm_arcsec": 1.3,
             "aperture_radius_arcsec": 1.3,
             "aperture_diameter_arcsec": 2.6,
             "aperture_psf_fwhm_arcsec": 1.3,
             "magnitude_fit_fingerprint": "m" * 64,
             "galaxy_density_arcmin2": 100.0,
             "galaxy_prior_density_arcmin2": 100.0,
             "galaxy_vis_magnitude_max": 29.0,
             "galaxy_magnitude_break": 26.35,
             "galaxy_faint_density_cap_arcmin2_mag": 100.0,
             "flux_e_per_band": [800.0, 1, 2, 3]},
        ],
        "lenses": [
            {"type": "lens", "x_pix": 128.0, "y_pix": 130.0, "z_lens": 0.5,
             "z_source": 2.0, "theta_E_arcsec": 1.3, "lens_subhalo_id": "g7",
             "lens_orientation": 2, "source_subhalo_id": "g8",
             "source_orientation": 5, "radius_manifest_fingerprint": "a" * 64,
             "lens_tng_trace": {"subhalo_id": "g7", "orientation": 2},
             "source_tng_trace": {"subhalo_id": "g8", "orientation": 5},
             "flux_e_per_band": [5000.0, 1, 2, 3]},
        ],
    }


def test_writer_then_reader_roundtrip(tmp_path):
    p = str(tmp_path / "sources_validate.csv")
    w = sc.SourceCatalogWriter(p)
    w.add_field(0, _meta())
    w.add_field(1, {"galaxies": [], "lenses": []})  # empty field still ok
    w.close()

    by_field = sc.read_sources(p)
    assert set(by_field) == {0}                       # field 1 contributed no rows
    rows = by_field[0]
    assert len(rows) == 3
    sersic = next(r for r in rows if r["render"] == "sersic")
    assert sersic["type"] == "galaxy" and sersic["x_pix"] == 100.0
    assert sersic["flux_vis_e"] == 3000.0 and sersic["z"] == 0.7
    assert [sersic[key] for key in ("flux_y_e", "flux_j_e", "flux_h_e")] == [
        1.0, 2.0, 3.0,
    ]
    assert sersic["target_logmass"] == 10.3
    assert sersic["target_logssfr"] == -10.4
    assert sersic["physical_model_fingerprint"] == "f" * 64
    lens = next(r for r in rows if r["type"] == "lens")
    assert lens["theta_E_arcsec"] == 1.3 and lens["subhalo_id"] == "g7"
    assert lens["lens_orientation"] == 2.0
    assert lens["source_subhalo_id"] == "g8"
    assert lens["source_orientation"] == 5.0
    assert lens["lens_tng_trace"] == {"subhalo_id": "g7", "orientation": 2}
    assert lens["source_tng_trace"] == {"subhalo_id": "g8", "orientation": 5}
    tng = next(r for r in rows if r["render"] == "tng")
    assert tng["subhalo_id"] == "99" and tng["z"] is None   # NaN -> None
    assert tng["orientation"] == 3.0
    assert tng["native_tng_sfr"] == 0.0
    assert tng["native_tng_logssfr"] is None
    assert tng["native_tng_zero_sfr"] == 1.0
    assert tng["target_ssfr_quantile"] == 0.1
    assert tng["tng_ssfr_quantile"] == 0.08
    assert tng["morphology_ssfr_quantile_delta"] == -0.02
    assert tng["target_re_arcsec"] == 0.31
    assert tng["achieved_re_arcsec"] is None
    assert tng["native_halflight_px"] == 12.4
    assert tng["rot_angle"] == 137.5
    assert tng["arbitrary_rotation"] is True
    assert tng["radius_scale_factor"] == 0.5
    assert tng["radius_rendering"] == TNG_RADIUS_RENDERING
    assert tng["radius_renderer_fingerprint"] == "r" * 64
    assert tng["radius_manifest_fingerprint"] == "a" * 64
    assert tng["tng_render_trace"] == {"subhalo_id": "99", "orientation": 3}
    assert tng["render_support_clipped"] is True
    assert tng["galaxy_density_arcmin2"] == 100.0
    assert tng["galaxy_prior_density_arcmin2"] == 100.0
    assert tng["galaxy_vis_magnitude_max"] == 29.0
    assert tng["galaxy_magnitude_break"] == 26.35
    assert tng["galaxy_faint_density_cap_arcmin2_mag"] == 100.0
    assert tng["target_vis_2fwhm_mag"] == 23.1
    assert tng["achieved_vis_2fwhm_mag"] == 23.1
    assert tng["mer_photometric_fwhm_arcsec"] == 1.3
    assert tng["aperture_radius_arcsec"] == 1.3
    assert tng["aperture_diameter_arcsec"] == 2.6
    assert tng["aperture_psf_fwhm_arcsec"] == 1.3
    assert tng["magnitude_fit_fingerprint"] == "m" * 64


def test_off_field_galaxy_roundtrips_separately_with_provenance(tmp_path):
    path = str(tmp_path / "sources_test.csv")
    off_field = {
        "type": "galaxy",
        "render": "tng",
        "x_pix": -3.5,
        "y_pix": 42.0,
        "subhalo_id": "edge",
        "orientation": 2,
        "re_arcsec": 0.2,
        "flux_e_per_band": [10.0, 8.0, 7.0, 6.0],
        "tng_render_trace": {"shape": [21, 21, 4]},
    }
    with sc.SourceCatalogWriter(path) as writer:
        writer.add_field(4, {
            "galaxies": [],
            "off_field_galaxies": [off_field],
            "lenses": [],
            "stars": [],
        })

    row = sc.read_sources(path)[4][0]
    assert row["type"] == "galaxy"
    assert row["off_field"] is True
    assert row["x_pix"] == -3.5
    assert row["subhalo_id"] == "edge"
    assert row["tng_render_trace"] == {"shape": [21, 21, 4]}
    assert sc.source_is_off_field(row)
    assert not sc.source_is_off_field({})


def test_stars_are_recorded(tmp_path):
    """Stars are now persisted (scene is starless → the forward re-injects
    them from these rows for the fixed validate/test fields)."""
    p = str(tmp_path / "sources_test.csv")
    w = sc.SourceCatalogWriter(p)
    w.add_field(0, {"galaxies": [], "lenses": [],
                    "stars": [{"type": "star", "x_pix": 12.0, "y_pix": 34.0,
                               "mag_vis": 19.5, "mag_y_e": 18.7,
                               "mag_j_e": 18.5, "mag_h_e": 18.4,
                               "temperature_k": 3420.0,
                               "extinction_av": 0.08}]})
    w.close()
    rows = sc.read_sources(p)[0]
    star = next(r for r in rows if r["type"] == "star")
    assert star["x_pix"] == 12.0 and star["y_pix"] == 34.0
    assert float(star["mag_vis"]) == 19.5
    assert [star[k] for k in ("mag_y_e", "mag_j_e", "mag_h_e")] == [
        18.7, 18.5, 18.4,
    ]
    assert star["temperature_k"] == 3420.0
    assert star["extinction_av"] == 0.08


def test_read_sources_missing_file(tmp_path):
    assert sc.read_sources(str(tmp_path / "nope.csv")) == {}


def test_current_tng_record_leaves_achieved_radius_blank(tmp_path):
    path = str(tmp_path / "sources_train.csv")
    with sc.SourceCatalogWriter(path) as writer:
        writer.add_field(0, {"galaxies": [{
            "type": "galaxy", "render": "tng", "x_pix": 1.0, "y_pix": 2.0,
            "re_arcsec": 0.03, "target_re_arcsec": 0.03,
            "native_halflight_px": 20.0,
            "radius_scale_factor": 0.03,
            "radius_rendering": TNG_RADIUS_RENDERING,
            "radius_renderer_fingerprint": "a" * 64,
            "render_support_clipped": False,
            "flux_e_per_band": [1.0, 1.0, 1.0, 1.0],
        }]})

    row = sc.read_sources(path)[0][0]
    assert row["re_arcsec"] == 0.03
    assert row["achieved_re_arcsec"] is None


def test_concat_source_csvs_preserves_order(tmp_path):
    a = str(tmp_path / "a.csv"); b = str(tmp_path / "b.csv")
    wa = sc.SourceCatalogWriter(a); wa.add_field(0, _meta()); wa.close()
    wb = sc.SourceCatalogWriter(b); wb.add_field(1, _meta()); wb.close()
    out = str(tmp_path / "sources_validate.csv")
    sc.concat_source_csvs([a, b], out)
    by_field = sc.read_sources(out)
    assert set(by_field) == {0, 1}
    with open(out) as f:
        assert sum(1 for ln in f if ln.startswith("field_index,")) == 1
