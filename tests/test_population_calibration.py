from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.population.magnitude_law import StraightMagnitudeLaw
from euclid_polish.sky.generation import sky_simulator as sky_simulator_module
from euclid_polish.sky.generation.sky_simulator import (
    SkySimulator,
    SkySimulatorConfig,
)
from euclid_polish.sky.generation.stellar_sed import EmpiricalStellarPrior
from euclid_polish.tng import (
    TNGAtlas,
    TNGGalaxy,
    TNGPropertyCatalog,
    TNGRadiusManifest,
)
from euclid_polish.tng.radius_manifest import (
    write_parameter_summary,
)
from euclid_polish.web.helpers.population_calibration import (
    activate_star_candidate,
    star_candidate_path,
    star_state,
)
from euclid_polish.web.helpers.q1_star_counts import (
    Q1_DEEP_FIELD_AREA_ARCMIN2,
    Q1_STAR_COUNT_VERSION,
    q1_star_counts_path,
)
from euclid_polish.web.helpers.q1_stellar_colors import (
    Q1_STELLAR_COLOR_SAMPLE_VERSION,
)
from euclid_polish.web.helpers.star_population import (
    _GAIA_G_AB_MINUS_VEGA_MAG,
    _fit_straight_star_magnitude_law,
    _weighted_summary,
    fit_star_population,
)
from tests._star_fixtures import fixture_star_color_model


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
        "population": {"magnitude_distribution": _straight_law_payload()},
        "color_model": fixture_star_color_model(),
    })
    draws = np.asarray([
        prior.sample_magnitude(np.random.default_rng(seed))
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


def test_previous_gaia_locus_artifact_cannot_activate(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    path = star_candidate_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "version": 6, "valid": True, "fingerprint": "c" * 64,
        "fingerprint_inputs": {
            "fit_version": "q1-phz-gaia-shared-straight-counts-latent-locus-v5",
        },
    }))
    assert not star_state()["candidate"]["valid"]
    with pytest.raises(ValueError, match="No valid fitted stellar"):
        activate_star_candidate()


def test_fixed_q1_stellar_colour_artifact_can_activate(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    candidate = {
        "version": 7,
        "valid": True,
        "fingerprint": "b" * 64,
        "fingerprint_inputs": {
            "fit_version": "q1-phz-straight-counts-euclid-colors-v6",
        },
    }
    path = star_candidate_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(candidate))

    active = activate_star_candidate()

    assert active["active"] is True
    assert star_state()["is_active"] is True


def test_count_slope_is_q1_alone_and_gaia_is_a_rebinned_comparison():
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
        # A deliberately different Gaia slope: it must not move the law.
        group_total = int(round(20.0 * 10.0 ** (0.2 * (group_centre - 12.0))))
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

    assert law.slope == pytest.approx(0.15, abs=1e-6)
    assert diagnostics["q1"]["slope"] == pytest.approx(0.15, abs=1e-6)
    assert diagnostics["gaia"]["role"] == "comparison only"
    assert diagnostics["gaia"]["slope"] == pytest.approx(0.2, abs=0.01)
    assert diagnostics["gaia"]["slope_difference_sigma"] > 3.0
    assert diagnostics["gaia"]["bin_width_mag"] == 0.5
    assert diagnostics["gaia"]["r_squared"] >= 0.99
    assert diagnostics["q1"]["bin_width_mag"] == pytest.approx(0.1)


def test_nested_thinning_keeps_nuisance_population_identical(monkeypatch):
    galaxy = TNGGalaxy(Path("x"), "1")
    atlas = TNGAtlas(
        root=Path("x"),
        galaxies=(galaxy,),
        properties=TNGPropertyCatalog({}, (None, 0, 0)),
        radii=TNGRadiusManifest(
            {("1", orientation): 20.0 for orientation in range(1, 6)},
            "test",
        ),
    )
    monkeypatch.setattr(
        sky_simulator_module.TNGAtlas,
        "open",
        classmethod(lambda cls, *args, **kwargs: atlas),
    )

    def build(density: float) -> dict:
        simulator = SkySimulator(object(), SkySimulatorConfig(
            image_size=10, pixel_scale=6.0,
            galaxy_density_arcmin2=density,
            galaxy_thinning_max_density_arcmin2=8.0,
            star_density_arcmin2=0.0, lens_density_arcmin2=0.0,
        ))
        simulator._add_tng_galaxy = lambda _canvas, rng, **_kwargs: {
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


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_star_fit_uses_q1_counts_and_euclid_colours_without_gaia(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    root = tmp_path / "population_comparison"
    rng = np.random.default_rng(4)
    euclid = []
    for index in range(400):
        vis = float(rng.uniform(18.0, 23.0))
        vis_y = float(rng.uniform(0.3, 1.5))
        colours = (vis_y, 0.05 + 0.1 * vis_y, 0.02 * vis_y)
        magnitudes = {"vis": vis}
        magnitudes["y"] = vis - colours[0]
        magnitudes["j"] = magnitudes["y"] - colours[1]
        magnitudes["h"] = magnitudes["j"] - colours[2]
        fluxes = {
            band: 10 ** ((23.9 - magnitude) / 2.5)
            for band, magnitude in magnitudes.items()
        }
        error = {band: flux / 40.0 for band, flux in fluxes.items()}
        measured = {
            band: flux + error[band] * rng.standard_normal()
            for band, flux in fluxes.items()
        }
        euclid.append({
            "object_id": index, "type": "star",
            "point_like_prob": "1.0", "phz_star_prob": "0.9",
            "mag_vis_psf": vis,
            **{f"flux_{band}_aper_uJy": flux for band, flux in measured.items()},
            **{f"fluxerr_{band}_aper_uJy": value for band, value in error.items()},
        })
    # The Gaia comparison counts: a native-G law with a different slope.
    gaia = []
    for bin_index, magnitude in enumerate(np.arange(12.05, 25.0, 0.1)):
        count = max(2, int(round(8.0 * 10 ** (0.2 * (magnitude - 12.0)))))
        gaia.extend({
            "source_id": f"g-{bin_index}-{repeat}", "g_mag": magnitude,
            "central_selected_star": 0,
        } for repeat in range(count))
    _write_csv(root / "gaia_population.csv", gaia)
    _write_csv(root / "q1_stellar_color_sample.csv", euclid)
    (root / "gaia_population.meta.json").write_text(json.dumps({
        "field_count": 3, "radius_deg": 0.35,
        "area_arcmin2": 8 * np.pi,
        "random_centres": False,
    }))
    (root / "q1_stellar_color_sample.meta.json").write_text(json.dumps({
        "version": Q1_STELLAR_COLOR_SAMPLE_VERSION,
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
    assert fit["color_sample_provenance"]["role"] == "Euclid point-source colours only"
    assert fit["color_sample_provenance"]["random_centres"] is False
    assert fit["coverage"]["color_rows"] == 400
    assert fit["color_model"]["kind"] == "q1_point_source_color_resampling_v1"
    assert "gaia" not in fit and "euclid_mapping" not in fit
    assert fit["quality_gates"]["passed"] and fit["valid"]
    # The law's slope is Q1 alone; Gaia's own (different) slope is reported.
    assert fit["population"]["magnitude_slope"] == pytest.approx(0.15, abs=1e-3)
    assert magnitude_gaia_slope(fit) == pytest.approx(0.2, abs=0.01)
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


def magnitude_gaia_slope(fit: dict) -> float:
    return fit["population"]["magnitude_distribution"]["fit_diagnostics"]["gaia"]["slope"]
