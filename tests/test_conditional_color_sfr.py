from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from euclid_polish.photometry import ab_mag_to_uJy
from euclid_polish.population.conditional_color_sfr import (
    COLOR_SFR_MODEL_VERSION,
    COLOR_SFR_SPLIT_CENSOR_LOGSFR,
    ConditionalColorSFRSampler,
    ratios_to_colors,
    weighted_mid_quantiles,
)
from euclid_polish.population.conditional_color_sfr_fit import (
    color_sfr_split_problem,
    fit_conditional_color_sfr_payload,
    read_color_sfr_rows,
)
from euclid_polish.population.euclid_galaxy_prior import (
    RADIUS_MODEL_VERSION,
    ConditionalRadiusLaw,
)

_CSV_COLUMNS = [
    "flux_vis_2fwhm_aper_uJy", "fluxerr_vis_2fwhm_aper_uJy",
    "flux_y_2fwhm_aper_uJy", "fluxerr_y_2fwhm_aper_uJy",
    "flux_j_2fwhm_aper_uJy", "fluxerr_j_2fwhm_aper_uJy",
    "flux_h_2fwhm_aper_uJy", "fluxerr_h_2fwhm_aper_uJy",
    "vis_det", "det_quality_flag", "spurious_prob", "point_like_prob",
    "morph_sersic_vis_radius_arcsec", "morph_sersic_vis_axis_ratio",
    "morph_sersic_visnir_flags",
    "phz_pp_median_sfr", "phz_pp_median_stellarmass",
    "phz_phys_flags", "phz_phys_quality_flag",
]


def fixture_radius_law() -> ConditionalRadiusLaw:
    return ConditionalRadiusLaw(
        version=RADIUS_MODEL_VERSION,
        pivot_mag=23.0,
        intercept_log10_arcsec=-0.4,
        slope_log10_arcsec_per_mag=-0.08,
        scatter_dex=0.18,
        log_radius_min=np.log10(0.03),
        log_radius_max=np.log10(10.0),
        fitted_rows=1000,
        clipped_rows=0,
        weighted_rows=800.0,
        residual_rms_dex=0.18,
        r_squared=0.3,
        covariance=((1e-4, 0.0), (0.0, 1e-5)),
        selection="fixture",
        fit_min_selected_per_magnitude_bin=20,
        fit_effective_weight_cap=1000.0,
        fit_faint_magnitude=25.5,
    )


def write_fixture_catalog(
    tmp_path: Path,
    rows: list[dict[str, object]],
    *,
    catalog_version: int = 7,
) -> tuple[Path, Path]:
    catalog_path = tmp_path / "euclid_population.csv"
    with catalog_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in _CSV_COLUMNS})
    meta_path = tmp_path / "euclid_population_meta.json"
    meta_path.write_text(json.dumps({
        "catalog_version": catalog_version,
        "rows": len(rows),
        "area_arcmin2": 100.0,
    }))
    return catalog_path, meta_path


def synthetic_rows(
    n_rows: int = 320,
    *,
    seed: int = 7,
    noise_sigma_ratio: float = 1e-7,
    intrinsic_sigma: float = 0.0,
    magnitude_range: tuple[float, float] = (19.0, 25.0),
    sfr_missing_above_mag: float = float("inf"),
) -> list[dict[str, object]]:
    """Two colour regimes vs magnitude, resolved radii, quenched+SF mix."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for index in range(n_rows):
        magnitude = float(np.interp(
            index, [0, n_rows - 1], list(magnitude_range),
        ))
        flux_vis = float(ab_mag_to_uJy(magnitude))
        base = 1.0 + 0.1 * (magnitude - 22.0)
        quenched = index % 3 == 0
        offset = 0.6 if quenched else 0.0
        intrinsic = np.asarray([
            base + offset, 1.1 * base + offset, 1.2 * base + offset,
        ]) + intrinsic_sigma * rng.standard_normal(3)
        noise_sigma = noise_sigma_ratio * flux_vis
        observed = intrinsic * flux_vis + noise_sigma * rng.standard_normal(3)
        radius = 10.0 ** (-0.4 - 0.08 * (magnitude - 23.0))
        sfr_known = magnitude < sfr_missing_above_mag
        log_mass = 9.5 + 0.3 * rng.standard_normal()
        log_ssfr = -11.8 if quenched else -9.6
        rows.append({
            "flux_vis_2fwhm_aper_uJy": flux_vis,
            "fluxerr_vis_2fwhm_aper_uJy": 1e-9 * flux_vis,
            "flux_y_2fwhm_aper_uJy": float(observed[0]),
            "fluxerr_y_2fwhm_aper_uJy": max(noise_sigma, 1e-12),
            "flux_j_2fwhm_aper_uJy": float(observed[1]),
            "fluxerr_j_2fwhm_aper_uJy": max(noise_sigma, 1e-12),
            "flux_h_2fwhm_aper_uJy": float(observed[2]),
            "fluxerr_h_2fwhm_aper_uJy": max(noise_sigma, 1e-12),
            "vis_det": 1,
            "det_quality_flag": 0,
            "spurious_prob": 0.01,
            "point_like_prob": 0.1,
            "morph_sersic_vis_radius_arcsec": radius,
            "morph_sersic_vis_axis_ratio": 0.8,
            "morph_sersic_visnir_flags": 0,
            "phz_pp_median_sfr": (
                log_mass + log_ssfr if sfr_known else ""
            ),
            "phz_pp_median_stellarmass": log_mass if sfr_known else "",
            "phz_phys_flags": 0 if sfr_known else "",
            "phz_phys_quality_flag": 0 if sfr_known else "",
        })
    return rows


def fit_fixture(tmp_path, rows, **kwargs):
    catalog_path, meta_path = write_fixture_catalog(tmp_path, rows)
    defaults = {
        "radius_law": fixture_radius_law(),
        "tree_count": 8,
        "min_leaf_weight": 5.0,
        "minimum_rows": 50,
    }
    defaults.update(kwargs)
    return fit_conditional_color_sfr_payload(
        catalog_path, meta_path, **defaults,
    )


def test_weighted_mid_quantiles_share_censored_floor_rank():
    values = np.asarray([-5.0, -5.0, -5.0, -1.0, 0.0])
    weights = np.asarray([1.0, 1.0, 2.0, 3.0, 1.0])

    ranks = weighted_mid_quantiles(values, weights)

    assert ranks[0] == ranks[1] == ranks[2] == pytest.approx(0.25)
    assert ranks[3] == pytest.approx((4.0 + 1.5) / 8.0)
    assert 0.0 < ranks[0] < ranks[3] < ranks[4] < 1.0


def test_fit_round_trips_and_rejects_tampering(tmp_path):
    payload, diagnostics = fit_fixture(tmp_path, synthetic_rows())

    sampler = ConditionalColorSFRSampler(payload)
    assert sampler.to_payload() is payload
    assert payload["version"] == COLOR_SFR_MODEL_VERSION
    assert len(payload["calibration_fingerprint"]) == 64
    assert diagnostics["selected_rows"] == sampler.row_count
    assert diagnostics["sfr_valid_weight_fraction"] == pytest.approx(1.0)

    corrupted = json.loads(json.dumps(payload))
    encoded = corrupted["rows"]["ratio_zlib_base64"]
    corrupted["rows"]["ratio_zlib_base64"] = encoded[:-8] + "AAAAAAA="
    with pytest.raises(ValueError):
        ConditionalColorSFRSampler(corrupted)

    wrong_version = json.loads(json.dumps(payload))
    wrong_version["version"] = COLOR_SFR_MODEL_VERSION + 1
    with pytest.raises(ValueError, match="version"):
        ConditionalColorSFRSampler(wrong_version)


def test_every_row_is_routed_to_its_recorded_leaf(tmp_path):
    rows = synthetic_rows()
    payload, _ = fit_fixture(tmp_path, rows)
    sampler = ConditionalColorSFRSampler(payload)

    for row_index in range(0, sampler.row_count, 17):
        source = rows[row_index]
        magnitude = float(
            -2.5 * math.log10(float(source["flux_vis_2fwhm_aper_uJy"]))
            + 23.90
        )
        radius = float(
            float(source["morph_sersic_vis_radius_arcsec"]) * math.sqrt(0.8)
        )
        x = np.asarray([magnitude, math.log10(radius), 1.0])
        for tree in sampler._trees:
            node = sampler._leaf_node(tree, x)
            start = int(tree["node_row_start"][node])
            stop = int(tree["node_row_start"][node + 1])
            assert row_index in tree["row_order"][start:stop]


def test_seeded_sampling_is_deterministic(tmp_path):
    payload, _ = fit_fixture(tmp_path, synthetic_rows())
    first = ConditionalColorSFRSampler(payload)
    second = ConditionalColorSFRSampler(payload)

    draws_a = [
        first.sample(22.0, 0.3, np.random.default_rng(5)) for _ in range(3)
    ]
    draws_b = [
        second.sample(22.0, 0.3, np.random.default_rng(5)) for _ in range(3)
    ]

    assert draws_a == draws_b


def test_out_of_support_queries_land_in_terminal_leaves(tmp_path):
    payload, _ = fit_fixture(tmp_path, synthetic_rows())
    sampler = ConditionalColorSFRSampler(payload)

    faint = sampler.sample(28.9, 0.05, np.random.default_rng(3))
    bright = sampler.sample(12.0, 5.0, np.random.default_rng(4))

    assert faint.neighborhood_rows > 0
    assert bright.neighborhood_rows > 0
    assert np.isfinite([faint.vis_minus_y, bright.vis_minus_y]).all()


def test_zero_noise_draws_reproduce_real_rows_exactly(tmp_path):
    rows = synthetic_rows(noise_sigma_ratio=1e-9)
    payload, _ = fit_fixture(tmp_path, rows)
    sampler = ConditionalColorSFRSampler(payload)
    catalog_ratios = np.asarray([
        [
            float(row[f"flux_{band}_2fwhm_aper_uJy"])
            / float(row["flux_vis_2fwhm_aper_uJy"])
            for band in ("y", "j", "h")
        ]
        for row in rows
    ])

    rng = np.random.default_rng(11)
    for _ in range(20):
        draw = sampler.sample(22.0, 0.3, rng)
        distances = np.max(
            np.abs(catalog_ratios - np.asarray(draw.ratios)[None, :]), axis=1,
        )
        assert float(np.min(distances)) < 1e-5


def test_noisy_draw_variance_matches_deconvolved_intrinsic(tmp_path):
    """Deconvolving noisy rows gives back the spread of the same galaxies
    measured without noise (not the raw spread, which adds the noise).

    The reference is a noise-free fit of the same fixture rather than an
    analytic variance: the neighbourhood's robust 16–84 spread under-reads
    the variance of the fixture's two-population (quenched offset) mixture,
    so only a like-for-like comparison isolates the noise removal. Leaves of
    100 keep the neighbourhood statistics stable (a 30-row neighbourhood's
    spread estimate swings by ±2× between seeds).
    """
    noise_ratio = 0.35

    def draw_variance(noise_sigma_ratio: float, directory: Path) -> float:
        directory.mkdir()
        rows = synthetic_rows(
            n_rows=1200, seed=8,
            noise_sigma_ratio=noise_sigma_ratio,
            intrinsic_sigma=0.2,
            magnitude_range=(21.9, 22.1),
        )
        payload, _ = fit_fixture(
            directory, rows, tree_count=8, min_leaf_weight=100.0,
        )
        sampler = ConditionalColorSFRSampler(payload)
        rng = np.random.default_rng(2)
        draws = np.asarray([
            sampler.sample(22.0, 0.3, rng).ratios for _ in range(2500)
        ])
        return float(np.var(draws[:, 0]))

    clean = draw_variance(1e-7, tmp_path / "clean")
    deconvolved = draw_variance(noise_ratio, tmp_path / "noisy")
    assert deconvolved == pytest.approx(clean, rel=0.2)
    # The raw rows would carry clean + 0.1225; the noise must be gone.
    assert deconvolved < clean + 0.5 * noise_ratio**2


def _legacy_payload(payload: dict) -> dict:
    """The same fit as a version-1 (per-band deconvolution) artifact."""
    legacy = json.loads(json.dumps(payload))
    legacy["version"] = 1
    legacy["rows"].pop("vis_snr_zlib_base64")
    legacy["rows"].pop("vis_snr_sha256")
    return legacy


def test_joint_deconvolution_keeps_the_correlation_between_bands(tmp_path):
    """Noisy rows must come back as correlated across bands as the same
    galaxies measured without noise. The per-band (version-1) posterior
    adds independent scatter to each band and loses most of it."""

    def band_correlation(noise_sigma_ratio: float, directory: Path, *, legacy: bool) -> float:
        directory.mkdir()
        rows = synthetic_rows(
            n_rows=1200, seed=9, noise_sigma_ratio=noise_sigma_ratio,
            intrinsic_sigma=0.2, magnitude_range=(21.9, 22.1),
        )
        payload, _ = fit_fixture(directory, rows, tree_count=8, min_leaf_weight=100.0)
        sampler = ConditionalColorSFRSampler(_legacy_payload(payload) if legacy else payload)
        rng = np.random.default_rng(2)
        draws = np.asarray([sampler.sample(22.0, 0.3, rng).ratios for _ in range(2500)])
        return float(np.corrcoef(draws[:, 0], draws[:, 1])[0, 1])

    clean = band_correlation(1e-7, tmp_path / "clean", legacy=False)
    joint = band_correlation(0.35, tmp_path / "joint", legacy=False)
    per_band = band_correlation(0.35, tmp_path / "per_band", legacy=True)
    assert clean > 0.5
    assert joint == pytest.approx(clean, abs=0.15)
    assert per_band < clean - 0.3


def test_version_one_artifacts_still_load_and_sample(tmp_path):
    payload, _ = fit_fixture(tmp_path, synthetic_rows())
    sampler = ConditionalColorSFRSampler(_legacy_payload(payload))
    assert sampler.version == 1
    draw = sampler.sample(22.0, 0.3, np.random.default_rng(0))
    assert all(value > 0.0 for value in draw.ratios)


def test_negative_fluxes_are_kept_and_draw_positive_ratios(tmp_path):
    rows = synthetic_rows(
        n_rows=240,
        noise_sigma_ratio=0.8,
        magnitude_range=(24.4, 24.6),
    )
    negative_inputs = sum(
        1 for row in rows if float(row["flux_y_2fwhm_aper_uJy"]) <= 0.0
    )
    assert negative_inputs > 0
    payload, diagnostics = fit_fixture(tmp_path, rows, tree_count=4)
    sampler = ConditionalColorSFRSampler(payload)
    assert diagnostics["negative_ratio_row_fraction"][0] > 0.0

    rng = np.random.default_rng(9)
    for _ in range(50):
        draw = sampler.sample(24.5, 0.2, rng)
        assert min(draw.ratios) >= sampler.ratio_floor
        assert np.isfinite(
            [draw.vis_minus_y, draw.y_minus_j, draw.j_minus_h]
        ).all()


def test_missing_sfr_borrows_from_neighbourhood(tmp_path):
    rows = synthetic_rows(sfr_missing_above_mag=22.0)
    payload, diagnostics = fit_fixture(tmp_path, rows)
    sampler = ConditionalColorSFRSampler(payload)
    assert diagnostics["sfr_valid_weight_fraction"] < 1.0

    rng = np.random.default_rng(21)
    borrowed = 0
    for _ in range(80):
        draw = sampler.sample(24.5, 0.2, rng)
        assert np.isfinite(draw.log_sfr)
        assert 0.0 < draw.sfr_rank < 1.0
        assert draw.sfr_class in ("quenched", "star_forming")
        if draw.sfr_borrowed:
            borrowed += 1
            assert not draw.sfr_valid
    assert borrowed > 0


def test_class_conditioned_draws_respect_the_class(tmp_path):
    payload, _ = fit_fixture(tmp_path, synthetic_rows())
    sampler = ConditionalColorSFRSampler(payload)

    rng = np.random.default_rng(31)
    for _ in range(20):
        quenched = sampler.sample_class_conditioned(22.0, "quenched", rng)
        forming = sampler.sample_class_conditioned(22.0, "star_forming", rng)
        assert quenched.sfr_class == "quenched"
        assert forming.sfr_class == "star_forming"
        # The fixture's quenched population is redder at fixed magnitude.
        assert quenched.vis_minus_y > forming.vis_minus_y - 0.5

    with pytest.raises(ValueError, match="unknown SFR class"):
        sampler.sample_class_conditioned(22.0, "green_valley", rng)


def test_low_vis_snr_rows_are_excluded_from_the_color_table(tmp_path):
    rows = synthetic_rows(n_rows=120)
    for row in rows[:30]:
        # S/N 1: the VIS flux sits in the ratio denominator at its own
        # noise level, so these colours are meaningless.
        row["fluxerr_vis_2fwhm_aper_uJy"] = float(
            row["flux_vis_2fwhm_aper_uJy"]
        )

    payload, diagnostics = fit_fixture(tmp_path, rows)

    assert diagnostics["selected_rows"] == 90
    assert payload["vis_snr_floor"] == 5.0


def test_junk_error_rows_cannot_zero_the_intrinsic_variance(tmp_path):
    intrinsic_sigma = 0.2
    rows = synthetic_rows(
        n_rows=400,
        noise_sigma_ratio=1e-6,
        intrinsic_sigma=intrinsic_sigma,
        magnitude_range=(21.9, 22.1),
    )
    for row in rows[::50]:
        # The MER junk-σ tail: a few percent of rows report errors hundreds
        # of times the typical value. Robust neighbourhood statistics must
        # keep them from collapsing every drawn colour to the median.
        row["fluxerr_y_2fwhm_aper_uJy"] = 1e4
        row["fluxerr_j_2fwhm_aper_uJy"] = 1e4
        row["fluxerr_h_2fwhm_aper_uJy"] = 1e4
    payload, _ = fit_fixture(tmp_path, rows, tree_count=4)
    sampler = ConditionalColorSFRSampler(payload)

    rng = np.random.default_rng(6)
    draws = np.asarray([
        sampler.sample(22.0, 0.3, rng).ratios for _ in range(1500)
    ])

    assert float(np.std(draws[:, 0])) > 0.5 * intrinsic_sigma


def test_deconvolution_metadata_names_the_trimmed_mean_noise_estimator(tmp_path):
    # The sampler's noise term is a weighted MEAN of the reported variances
    # with rows above 25× the weighted-median variance dropped, not a median.
    payload, _ = fit_fixture(tmp_path, synthetic_rows(n_rows=80), tree_count=2)

    text = payload["deconvolution"]

    assert "weighted-median reported" not in text
    assert "trimmed weighted-mean reported noise variance" in text
    assert "25x the weighted-median variance" in text


def test_stale_catalog_version_is_refused(tmp_path):
    rows = synthetic_rows(n_rows=80)
    catalog_path, meta_path = write_fixture_catalog(
        tmp_path, rows, catalog_version=6,
    )

    with pytest.raises(ValueError, match="catalog_version"):
        fit_conditional_color_sfr_payload(
            catalog_path, meta_path,
            radius_law=fixture_radius_law(),
            tree_count=4, min_leaf_weight=5.0, minimum_rows=50,
        )

    payload, _ = fit_conditional_color_sfr_payload(
        catalog_path, meta_path,
        radius_law=fixture_radius_law(),
        require_catalog_version=False,
        tree_count=4, min_leaf_weight=5.0, minimum_rows=50,
    )
    assert payload["catalog_version"] == 6


def test_mean_colors_track_the_fixture_trend(tmp_path):
    payload, _ = fit_fixture(tmp_path, synthetic_rows())
    sampler = ConditionalColorSFRSampler(payload)

    colors = sampler.mean_colors(np.asarray([20.0, 24.0]))

    assert colors.shape == (2, 3)
    # Fixture ratios rise with magnitude, so VIS-Y is redder at 24 than 20.
    assert colors[1, 0] > colors[0, 0]


def test_sfr_tail_is_censored_in_the_split_target_only(tmp_path):
    """A junk log-SFR tail must not set the SFR target's scale, but the
    sampled SFRs stay the raw PHZ values."""
    rows = synthetic_rows(240)
    for index in range(0, 240, 20):
        # sSFR ≈ −250: valid (below the pathological limit), absurd SFR.
        rows[index]["phz_pp_median_sfr"] = (
            float(rows[index]["phz_pp_median_stellarmass"]) - 250.0
        )
    catalog_path, _ = write_fixture_catalog(tmp_path, rows)
    selected = read_color_sfr_rows(catalog_path, radius_law=fixture_radius_law())
    _, targets = color_sfr_split_problem(selected)
    raw_sfr = selected.log_sfr[selected.sfr_valid]
    assert raw_sfr.min() < -200.0
    # The ordinary (quenched vs star-forming) SFR spread must survive
    # standardization; an uncensored tail would squash it to ~0.02 sigma.
    sfr_target = targets[:, 3]
    ordinary = selected.sfr_valid & (selected.log_sfr > -50.0)
    assert np.std(sfr_target[ordinary]) > 0.3

    payload, _ = fit_fixture(tmp_path, rows)
    assert payload["sfr_split_censor_logsfr"] == COLOR_SFR_SPLIT_CENSOR_LOGSFR
    sampler = ConditionalColorSFRSampler(payload)
    rng = np.random.default_rng(3)
    draws = [sampler.sample(19.2, 0.6, rng).log_sfr for _ in range(400)]
    assert min(draws) < -200.0


def test_ratios_to_colors_matches_ab_definition():
    vis_minus_y, y_minus_j, j_minus_h = ratios_to_colors(
        np.asarray([2.0, 2.0, 4.0])
    )

    assert vis_minus_y == pytest.approx(2.5 * math.log10(2.0))
    assert y_minus_j == pytest.approx(0.0)
    assert j_minus_h == pytest.approx(2.5 * math.log10(2.0))
