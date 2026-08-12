from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.population.magnitude_law import StraightMagnitudeLaw
from euclid_polish.sky.generation.sky_simulator import (
    SkySimulator,
    SkySimulatorConfig,
)
from euclid_polish.sky.generation.stellar_sed import EmpiricalStellarPrior
from euclid_polish.sky.generation.tng_radius_manifest import (
    write_parameter_summary,
)
from euclid_polish.web.helpers.population_calibration import (
    activate_galaxy_recommendation,
    activate_star_candidate,
    active_transfer_path,
    density_calibration_path,
    fit_density_response,
    fit_local_catalog_density,
    galaxy_recommendation_state,
    star_candidate_path,
    star_state,
)
from euclid_polish.web.helpers.q1_star_counts import (
    Q1_DEEP_FIELD_AREA_ARCMIN2,
    Q1_STAR_COUNT_VERSION,
    q1_star_counts_path,
)
from euclid_polish.web.helpers.star_population import (
    _GAIA_G_AB_MINUS_VEGA_MAG,
    _fit_straight_star_magnitude_law,
    _weighted_summary,
    fit_star_population,
)


def _straight_law_payload(
    density: float = 1.0, *, bright: float = 20.0, faint: float = 22.0,
    slope: float = float(np.log10(3.0)),
) -> dict:
    beta = slope * np.log(10.0)
    integral_without_normalisation = (
        np.exp(beta * faint) - np.exp(beta * bright)
    ) / beta
    return StraightMagnitudeLaw(
        slope=slope,
        intercept=float(np.log10(density / integral_without_normalisation)),
        mag_bright=bright,
        mag_faint=faint,
        fit_bright=bright,
        fit_faint=faint,
        covariance=((1.0e-4, 0.0), (0.0, 1.0e-3)),
        r_squared=1.0,
        rms_log10_density=0.0,
        source="fixture",
    ).to_payload()


def test_probability_weighted_summary_and_straight_magnitude_law():
    values = np.asarray([20.0, 22.0])
    weights = np.asarray([0.25, 0.75])
    summary = _weighted_summary(
        values, weights, area_arcmin2=2.0,
        classification_variance=float(np.sum(weights * (1.0 - weights))),
    )
    assert summary["expected_count"] == pytest.approx(1.0)
    assert summary["density_arcmin2"] == pytest.approx(0.5)
    assert summary["mean"] == pytest.approx(21.5)
    assert summary["effective_n"] == pytest.approx(1.6)
    assert summary["classification_sigma_count"] == pytest.approx(0.612372)


    prior = EmpiricalStellarPrior.from_payload({
        "gaia": {"bp_rp_quantiles": [0.5, 1.0],
                 "temperature_quantiles_k": [4000.0, 6000.0]},
        "euclid_mapping": {
            "g_to_band_offset_coefficients": {
                key: [0.0, 0.0, 0.0]
                for key in ("mag_vis", "mag_y_e", "mag_j_e", "mag_h_e")
            },
            "residual_covariance": np.eye(4).tolist(),
        },
        "population": {"magnitude_distribution": _straight_law_payload()},
        "color_model": {
            "kind": "gaia_euclid_latent_locus_v1",
            "bp_rp_edges": [0.0, 0.75, 1.5],
            "bp_rp_nodes": [0.4, 1.1],
            "temperature_nodes_k": [6500.0, 4500.0],
            "locus_colors": [[0.2, 0.1, 0.05], [0.8, 0.3, 0.1]],
            "intrinsic_color_covariance": (np.eye(3) * 0.01).tolist(),
            "magnitude_edges": [20.0, 21.0, 22.0],
            "magnitude_node_weights": [[0.5, 0.5], [0.5, 0.5]],
        },
    })
    draws = np.asarray([
        prior.sample_magnitude(
            np.random.default_rng(seed), slope=0.2,
            m_bright=20.0, m_faint=22.0,
        )
        for seed in range(2000)
    ])
    assert np.mean(draws < 21.0) == pytest.approx(0.25, abs=0.04)


def test_stale_gaia_count_artifact_cannot_remain_active(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    candidate = star_candidate_path()
    candidate.parent.mkdir(parents=True)
    candidate.write_text(json.dumps({
        "version": 3,
        "valid": True,
        "fingerprint": "a" * 64,
        "fingerprint_inputs": {"fit_version": "latent-locus-v1"},
    }))

    state = star_state()
    assert not state["candidate"]["valid"]
    assert "Q1 PHZ_STAR_PROB" in state["candidate"]["warnings"][-1]
    assert not state["is_active"]
    with pytest.raises(ValueError, match="No valid fitted stellar"):
        activate_star_candidate()


def test_fixed_q1_stellar_colour_artifact_can_activate(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    candidate = {
        "version": 6,
        "valid": True,
        "fingerprint": "b" * 64,
        "fingerprint_inputs": {
            "fit_version": (
                "q1-phz-gaia-shared-straight-counts-latent-locus-v5"
            ),
        },
    }
    path = star_candidate_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(candidate))

    active = activate_star_candidate()

    assert active["active"] is True
    assert star_state()["is_active"] is True


def test_gaia_shape_fit_rebins_sparse_point_one_mag_counts():
    edges = np.linspace(12.0, 25.0, 131)
    q1_bins = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        centre = 0.5 * (lower + upper)
        expected = 50.0 * 10.0 ** (0.15 * (centre - 12.0))
        q1_bins.append({
            "expected_stars": expected,
            "classified_rows": int(round(2.0 * expected)),
            "classification_variance": 0.1 * expected,
        })

    # Strong within-0.5-mag structure makes the raw 0.1-mag histogram fail a
    # straightness gate, while each five-bin sum follows one clean count law.
    gaia_rows = []
    subbin_weights = np.asarray([0.2, 1.8, 0.3, 1.7, 1.0])
    for group, group_centre in enumerate(np.arange(12.25, 25.0, 0.5)):
        group_total = int(round(20.0 * 10.0 ** (0.15 * (group_centre - 12.0))))
        counts = np.maximum(
            1, np.rint(group_total * subbin_weights / subbin_weights.sum()),
        ).astype(int)
        for subbin, count in enumerate(counts):
            g_ab = 12.05 + 0.5 * group + 0.1 * subbin
            gaia_rows.extend({
                "g_mag": str(g_ab - _GAIA_G_AB_MINUS_VEGA_MAG),
                "central_selected_star": "0",
            } for _ in range(int(count)))

    law, diagnostics = _fit_straight_star_magnitude_law(
        gaia_rows,
        {"area_arcmin2": 100.0},
        {
            "edges": edges.tolist(),
            "bins": q1_bins,
            "footprint_area_arcmin2": Q1_DEEP_FIELD_AREA_ARCMIN2,
        },
    )

    assert law.slope == pytest.approx(0.15, abs=0.01)
    assert diagnostics["gaia"]["bin_width_mag"] == 0.5
    assert diagnostics["gaia"]["r_squared"] >= 0.99
    assert diagnostics["q1"]["bin_width_mag"] == pytest.approx(0.1)


def test_density_response_is_reproducible_and_rejects_wrong_transfer():
    densities = [240.0, 280.0, 320.0, 360.0, 400.0]
    fields = [[density / 10 + offset for offset in (-1, 0, 1, 0)]
              for density in densities]
    real = [31.0, 32.0, 33.0, 32.0]
    first = fit_density_response(
        densities, fields, real, transfer_fingerprint="same",
        active_transfer_fingerprint="same", field_area_arcmin2=1.0,
        euclid_cone_detection_densities=[30.0, 32.0, 34.0],
        bootstraps=100, seed=5,
    )
    second = fit_density_response(
        densities, fields, real, transfer_fingerprint="same",
        active_transfer_fingerprint="same", field_area_arcmin2=1.0,
        euclid_cone_detection_densities=[30.0, 32.0, 34.0],
        bootstraps=100, seed=5,
    )
    assert first["valid"]
    assert first["recommended_density_arcmin2"] == pytest.approx(320.0)
    assert first["interval_arcmin2"] == second["interval_arcmin2"]
    assert first["euclid_cones"] == 3

    mismatch = fit_density_response(
        densities, fields, real, transfer_fingerprint="old",
        active_transfer_fingerprint="new", field_area_arcmin2=1.0,
        bootstraps=100,
    )
    assert not mismatch["valid"]
    assert "different" in " ".join(mismatch["warnings"]) or "not the active" in " ".join(mismatch["warnings"])


def test_local_catalog_fit_recovers_raw_density_without_rendering(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    root = tmp_path / "population_comparison"
    prior_path = root / "cosmos2025" / "prior.npz"
    fit_path = root / "cosmos2025" / "fit.json"
    prior_path.parent.mkdir(parents=True)
    monkeypatch.setattr(Config, "COSMOS_TNG_PRIOR_PATH", str(prior_path))
    monkeypatch.setattr(Config, "COSMOS_EUCLID_FIT_PATH", str(fit_path))
    atlas_summary_path = root / "tng_atlas_parameters.csv"
    monkeypatch.setattr(
        Config, "TNG_ATLAS_PARAMETERS_PATH", str(atlas_summary_path),
    )
    size = 2_000
    np.savez_compressed(
        prior_path,
        catalog_id=np.arange(size),
        mag_hst_f814w=np.full(size, 24.0),
        z_phot=np.full(size, 1.0),
        logmass_lephare=np.full(size, 10.0),
        logssfr_lephare=np.where(
            np.arange(size) % 2 == 0, -12.0, -10.0,
        ),
        re_combined_arcsec=np.full(size, 0.4),
        generator_ready=np.ones(size, dtype=bool),
    )
    properties_path = tmp_path / "tng_properties.csv"
    properties_path.write_text(
        "id,sfr,mass_stars,m_halo,reff\n"
        f"1,0.001,{10.0 ** 9.8},1e12,2\n"
        f"2,0.001,{10.0 ** 10.2},1e12,2\n"
        f"3,1,{10.0 ** 9.8},1e12,2\n"
        f"4,1,{10.0 ** 10.2},1e12,2\n"
    )
    manifest = {
        "valid": True,
        "algorithm_version": "test-cog-v1",
        "manifest_fingerprint": "radius-v1",
        "atlas_inventory_fingerprint": "atlas-v1",
        "entries": [
            {
                "subhalo_id": gid,
                "orientation": orientation,
                "native_re_px": 10.0,
                "shape": [64, 64],
                "valid": True,
            }
            for gid in ("1", "2", "3", "4")
            for orientation in range(1, 6)
        ],
    }
    write_parameter_summary(
        atlas_summary_path, manifest, properties_path=str(properties_path),
    )
    fit_path.write_text(json.dumps({
        "inputs": {
            "euclid_cone_count": 4,
            "euclid_area_arcmin2": 4.0,
            "euclid_cones": [{"star_id": str(index)} for index in range(4)],
        },
        "fit": {
            "vis_minus_f814w_mag": 0.0,
            "magnitude_slope": 1.0,
            "scatter_mag": 0.0,
            "completeness_m50": 24.0,
            "completeness_width_mag": 1.0,
            "poisson_deviance": 4.0,
            "dof": 4,
        },
    }))
    (root / "euclid_population_meta.json").write_text(json.dumps({
        "cone_count": 4,
        "area_arcmin2": 4.0,
        "cones": [{"star_id": str(index)} for index in range(4)],
    }))
    rows = []
    for cone_index in range(4):
            rows.extend({
                "type": "unknown",
                "spurious_prob": "0.0",
                "point_like_prob": "0.0",
                "mag_vis": "24.0",
            "cone_index": str(cone_index),
        } for _ in range(10))
    _write_csv(root / "euclid_population.csv", rows)

    first = fit_local_catalog_density(bootstraps=100, seed=8)
    second = fit_local_catalog_density(bootstraps=100, seed=8)

    assert first["valid"]
    assert first["retained_detection_fraction"] == pytest.approx(0.5)
    assert first["euclid_detected_density_arcmin2"] == pytest.approx(10.0)
    assert first["recommended_density_arcmin2"] == pytest.approx(20.0)
    assert first["cosmos_generator_rows"] == size
    assert first["morphology_model"]["eligible_cosmos_rows"] == size
    assert first["morphology_model"]["method"] == (
        "activity_conditioned_empirical_mass_quantile_transport"
    )
    assert first["morphology_model"]["excluded_cosmos_rows"] == 0
    assert first["magnitude_fit_quality"]["valid"]
    assert first["method"].startswith("empirical COSMOS/TNG")
    assert first["forward_integration_grid_step_mag"] == pytest.approx(0.005)
    assert "local_draws" not in first
    assert first["calibration_fingerprint"] == second["calibration_fingerprint"]
    assert first["interval_arcmin2"] == second["interval_arcmin2"]
    assert density_calibration_path().exists()


def test_nested_thinning_keeps_nuisance_population_identical(monkeypatch):
    import euclid_polish.sky.generation.sky_simulator as module

    monkeypatch.setattr(module, "list_tng_galaxies", lambda _path: [("x", "1")])
    monkeypatch.setattr(module, "load_tng_properties", lambda _path: {})

    def build(density: float) -> dict:
        simulator = SkySimulator(object(), SkySimulatorConfig(
            image_size=10, pixel_scale=6.0,
            galaxy_density_arcmin2=density,
            galaxy_thinning_max_density_arcmin2=8.0,
            star_density_arcmin2=0.0, lens_density_arcmin2=0.0,
        ))
        simulator._add_tng_galaxy = lambda _canvas, rng: {
            "proposal": int(rng.integers(0, 2**31)),
        }
        return simulator.simulate_field(np.random.default_rng(91))[1]

    lower = build(4.0)
    upper = build(8.0)
    low_ids = {item["proposal"] for item in lower["galaxies"]}
    high_ids = {item["proposal"] for item in upper["galaxies"]}
    assert low_ids <= high_ids
    assert lower["stars"] == upper["stars"]
    assert lower["lenses"] == upper["lenses"]


def test_complete_generator_recommendation_can_activate_with_fit_warnings(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    fit_path = tmp_path / "fit.json"
    monkeypatch.setattr(Config, "COSMOS_EUCLID_FIT_PATH", str(fit_path))
    fit_path.write_text(json.dumps({
        "inputs": {"euclid_cone_count": 6},
        "fit": {
            "vis_minus_f814w_mag": 0.4,
            "magnitude_slope": 0.7,
            "scatter_mag": 1.0,
            "completeness_m50": 25.1,
            "completeness_width_mag": 0.5,
            "poisson_deviance": 60.0,
            "dof": 10,
        },
    }))
    from euclid_polish.sky.generation.cosmos_tng_prior import (
        brightness_transfer_payload,
    )
    transfer = brightness_transfer_payload(fit_path)
    assert transfer is not None
    assert transfer["version"] == 3
    assert not transfer["valid"]
    density_calibration_path().parent.mkdir(parents=True, exist_ok=True)
    density_calibration_path().write_text(json.dumps({
        "valid": True,
        "warnings": [],
        "transfer_fingerprint": transfer["fingerprint"],
        "calibration_fingerprint": "sweep",
        "recommended_density_arcmin2": 315.0,
        "interval_arcmin2": {"median": 315, "p16": 295, "p84": 338},
    }))
    updates = []
    monkeypatch.setattr(
        "euclid_polish.web.job_config.update", lambda patch: updates.append(patch),
    )

    state = galaxy_recommendation_state()
    assert state["recommendation_available"]
    assert not state["validated"]
    assert state["generator_parameters"] == {
        "galaxy_density_arcmin2": 315.0,
        "cosmos_vis_offset_mag": 0.4,
        "cosmos_vis_magnitude_slope": 0.7,
        "cosmos_vis_scatter_mag": 1.0,
    }

    activated = activate_galaxy_recommendation()
    assert activated["active"]
    assert activated["brightness_transfer"][
        "activated_with_quality_warnings"
    ]
    assert json.loads(active_transfer_path().read_text())["fingerprint"] == transfer[
        "fingerprint"
    ]
    assert updates == [{"galaxy_density_arcmin2": 315.0}]


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_star_fit_uses_q1_phz_counts_and_gaia_only_for_correlated_colors(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    root = tmp_path / "population_comparison"
    gaia = []
    euclid = []
    for index in range(40):
        field_index = index % 2
        source_id = str(1000 + index)
        color = 0.5 + 0.05 * index
        g_mag = 16.0 + 0.18 * index
        gaia.append({
            "source_id": source_id, "field_index": field_index, "ra": 1,
            "dec": 2, "g_mag": g_mag, "bp_mag": g_mag + color / 2,
            "rp_mag": g_mag - color / 2, "bp_rp": color,
            "temperature_k": 8000 - 150 * index, "extinction_g_mag": 0.1,
            "central_selected_star": 1 if index < 2 else 0,
        })
        magnitudes = {
            "vis": g_mag + 0.2 * color,
            "y": g_mag - 0.4 * color,
            "j": g_mag - 0.55 * color,
            "h": g_mag - 0.60 * color,
        }
        fluxes = {
            band: 10 ** ((23.9 - magnitude) / 2.5)
            for band, magnitude in magnitudes.items()
        }
        euclid.append({
            "object_id": index, "gaia_id": source_id, "type": "star",
            "point_like_prob": "1.0",
            "mag_vis": magnitudes["vis"],
            "mag_y_e": magnitudes["y"],
            "mag_j_e": magnitudes["j"],
            "mag_h_e": magnitudes["h"],
            **{
                f"flux_{band}_aper_uJy": flux
                for band, flux in fluxes.items()
            },
            **{
                f"fluxerr_{band}_aper_uJy": flux / 20.0
                for band, flux in fluxes.items()
            },
        })
    # Deterministic native-G count law for straight-region detection; these
    # additional Gaia rows inform shape but have no Euclid match.
    for bin_index, magnitude in enumerate(np.arange(12.05, 25.0, 0.1)):
        count = max(2, int(round(
            8.0 * 10 ** (0.15 * (magnitude - 12.0))
        )))
        for repeat in range(count):
            color = 0.5 + 0.01 * ((bin_index + repeat) % 120)
            gaia.append({
                "source_id": f"extra-{bin_index}-{repeat}",
                "field_index": repeat % 3,
                "ra": 1, "dec": 2, "g_mag": magnitude,
                "bp_mag": magnitude + color / 2,
                "rp_mag": magnitude - color / 2,
                "bp_rp": color,
                "temperature_k": 6500 - 500 * color,
                "extinction_g_mag": 0.1,
                "central_selected_star": 0,
            })
    _write_csv(root / "gaia_population.csv", gaia)
    _write_csv(root / "q1_stellar_color_sample.csv", euclid)
    (root / "gaia_population.meta.json").write_text(json.dumps({
        "field_count": 3, "radius_deg": 0.35,
        "area_arcmin2": 8 * np.pi,
        "random_centres": False,
    }))
    (root / "q1_stellar_color_sample.meta.json").write_text(json.dumps({
        "field_count": 3, "radius_deg": 0.35,
        "area_arcmin2": 8 * np.pi,
        "random_centres": False,
    }))
    edges = np.linspace(12.0, 25.0, 131)
    q1_bins = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        centre = 0.5 * (lower + upper)
        expected = float(round(20.0 * 10 ** (0.15 * (centre - 12.0))))
        q1_bins.append({
            "mag_lo": float(lower),
            "mag_hi": float(upper),
            "classified_rows": int(expected * 2),
            "selected_point_sources": int(expected * 2),
            "expected_point_sources": expected * 1.1,
            "expected_stars": expected,
            "classification_variance": expected * 0.1,
            "point_source_density_arcmin2_mag": (
                expected * 1.1 / Q1_DEEP_FIELD_AREA_ARCMIN2 / 0.1
            ),
            "density_arcmin2_mag": (
                expected / Q1_DEEP_FIELD_AREA_ARCMIN2 / 0.1
            ),
        })
    expected_total = float(sum(item["expected_stars"] for item in q1_bins))
    q1_star_counts_path().write_text(json.dumps({
        "version": Q1_STAR_COUNT_VERSION,
        "survey": "Euclid Q1 deep fields",
        "fields": ["EDF-N", "EDF-S", "EDF-F"],
        "footprint_area_deg2": 63.1,
        "footprint_area_arcmin2": Q1_DEEP_FIELD_AREA_ARCMIN2,
        "magnitude_field": "MER FLUX_VIS_PSF",
        "classification_field": "PHZ_STAR_PROB",
        "selection": "POINT_LIKE_PROB >= 0.9 test Q1 selection",
        "edges": edges.tolist(),
        "bins": q1_bins,
        "expected_stars": expected_total,
        "selected_point_sources": int(sum(
            item["selected_point_sources"] for item in q1_bins
        )),
        "expected_point_sources": float(sum(
            item["expected_point_sources"] for item in q1_bins
        )),
        "classification_variance": float(sum(
            item["classification_variance"] for item in q1_bins
        )),
    }))
    monkeypatch.setattr(
        "euclid_polish.web.helpers.star_population._require_current_gaia_field_sampling",
        lambda _meta, _rows: None,
    )

    fit = fit_star_population()
    assert fit["color_sample_provenance"]["role"].endswith("locus only")
    assert fit["color_sample_provenance"]["random_centres"] is False
    assert fit["euclid_mapping"]["matched_stars"] == 38
    assert fit["population_provenance"]["classification_field"] == "PHZ_STAR_PROB"
    magnitude = fit["population"]["magnitude_distribution"]
    assert magnitude["kind"] == "straight_log10_differential_counts"
    assert magnitude["mag_bright"] == 12.0
    assert magnitude["mag_faint"] == 25.0
    assert magnitude["phz_expected_count"] == pytest.approx(expected_total)
    assert magnitude["fit_diagnostics"]["gaia"]["bin_width_mag"] == 0.5
    assert magnitude["fit_diagnostics"]["q1"]["bin_width_mag"] == pytest.approx(0.1)
    assert magnitude["expected_count_per_bin"] == pytest.approx([
        item["expected_stars"] for item in q1_bins
    ])
    assert fit["population"]["density_arcmin2"] == pytest.approx(
        magnitude["surface_density_arcmin2"],
    )
    density = fit["diagnostics"]["stellar_density_by_magnitude"]
    assert density["x_label"] == "native survey magnitude [AB]"
    assert len(density["gaia_observed"]) == len(density["gaia_x"])
    assert len(density["gaia_fitted"]) == len(density["gaia_x"])
    assert density["gaia_x"][1] - density["gaia_x"][0] == pytest.approx(0.5)
    assert density["fit_ranges"]["q1"][0] >= 12.0
    assert density["fit_ranges"]["gaia"][1] <= 25.0
    assert density["observed"] == pytest.approx([
        item["density_arcmin2_mag"] for item in q1_bins
    ])

    prior = EmpiricalStellarPrior.from_payload(fit)
    sed = prior.sample(np.random.default_rng(8), 21.5)
    assert sed.magnitudes["VIS"] == pytest.approx(21.5)
    assert set(sed.magnitudes) == {"VIS", "Y_E", "J_E", "H_E"}
