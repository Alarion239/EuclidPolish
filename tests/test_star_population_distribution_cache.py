"""The stellar plot cache (``star_distribution.json``) after a prior fit: the
fit job's persisted default view carries the generated-star overlay."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.web.helpers import star_population
from euclid_polish.web.helpers.q1_star_counts import (
    Q1_DEEP_FIELD_AREA_ARCMIN2,
    Q1_STAR_COUNT_VERSION,
    q1_star_counts_path,
)


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cache_fit_inputs(root):
    """A fixed-field Gaia–Euclid colour sample + Q1 PHZ counts the strict
    stellar fit accepts (40 matched stars, a straight native-G count law)."""
    gaia = []
    euclid = []
    for index in range(40):
        source_id = str(1000 + index)
        color = 0.5 + 0.05 * index
        g_mag = 16.0 + 0.18 * index
        gaia.append({
            "source_id": source_id, "field_index": index % 2, "ra": 1,
            "dec": 2, "g_mag": g_mag, "bp_mag": g_mag + color / 2,
            "rp_mag": g_mag - color / 2, "bp_rp": color,
            "temperature_k": 8000 - 150 * index, "extinction_g_mag": 0.1,
            "central_selected_star": 1 if index < 2 else 0,
        })
        magnitudes = {
            "vis": g_mag + 0.2 * color, "y": g_mag - 0.4 * color,
            "j": g_mag - 0.55 * color, "h": g_mag - 0.60 * color,
        }
        fluxes = {band: 10 ** ((23.9 - mag) / 2.5) for band, mag in magnitudes.items()}
        euclid.append({
            "object_id": index, "gaia_id": source_id, "type": "star",
            "point_like_prob": "1.0",
            "mag_vis": magnitudes["vis"], "mag_y_e": magnitudes["y"],
            "mag_j_e": magnitudes["j"], "mag_h_e": magnitudes["h"],
            **{f"flux_{band}_aper_uJy": flux for band, flux in fluxes.items()},
            **{f"fluxerr_{band}_aper_uJy": flux / 20.0 for band, flux in fluxes.items()},
        })
    for bin_index, magnitude in enumerate(np.arange(12.05, 25.0, 0.1)):
        count = max(2, int(round(8.0 * 10 ** (0.15 * (magnitude - 12.0)))))
        for repeat in range(count):
            color = 0.5 + 0.01 * ((bin_index + repeat) % 120)
            gaia.append({
                "source_id": f"extra-{bin_index}-{repeat}",
                "field_index": repeat % 3, "ra": 1, "dec": 2, "g_mag": magnitude,
                "bp_mag": magnitude + color / 2, "rp_mag": magnitude - color / 2,
                "bp_rp": color, "temperature_k": 6500 - 500 * color,
                "extinction_g_mag": 0.1, "central_selected_star": 0,
            })
    _write_csv(root / "gaia_population.csv", gaia)
    _write_csv(root / "q1_stellar_color_sample.csv", euclid)
    meta = {"field_count": 3, "radius_deg": 0.35, "area_arcmin2": 8 * np.pi,
            "random_centres": False}
    (root / "gaia_population.meta.json").write_text(json.dumps(meta))
    (root / "q1_stellar_color_sample.meta.json").write_text(json.dumps(meta))
    edges = np.linspace(12.0, 25.0, 131)
    bins = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        expected = float(round(20.0 * 10 ** (0.15 * (0.5 * (lower + upper) - 12.0))))
        bins.append({
            "mag_lo": float(lower), "mag_hi": float(upper),
            "classified_rows": int(expected * 2),
            "selected_point_sources": int(expected * 2),
            "expected_point_sources": expected * 1.1,
            "expected_stars": expected,
            "classification_variance": expected * 0.1,
            "point_source_density_arcmin2_mag": expected * 1.1 / Q1_DEEP_FIELD_AREA_ARCMIN2 / 0.1,
            "density_arcmin2_mag": expected / Q1_DEEP_FIELD_AREA_ARCMIN2 / 0.1,
        })
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
        "bins": bins,
        "expected_stars": float(sum(item["expected_stars"] for item in bins)),
        "selected_point_sources": int(sum(item["selected_point_sources"] for item in bins)),
        "expected_point_sources": float(sum(item["expected_point_sources"] for item in bins)),
        "classification_variance": float(sum(item["classification_variance"] for item in bins)),
    }))


def _generated_star_catalogue(path):
    """A ``sources_test.csv`` with 30 generated stars over two fields."""
    _write_csv(path, [
        {"field_index": index % 2, "type": "star",
         "mag_vis": 20.0 + 0.1 * index, "mag_y_e": 19.6 + 0.1 * index,
         "mag_j_e": 19.4 + 0.1 * index, "mag_h_e": 19.3 + 0.1 * index}
        for index in range(30)
    ])


@pytest.fixture
def fitted_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    _cache_fit_inputs(tmp_path / "population_comparison")
    sources = tmp_path / "records" / "sources_test.csv"
    _generated_star_catalogue(sources)
    monkeypatch.setattr(
        star_population, "_synthetic_paths",
        lambda include_training=False: ([], [sources]),
    )
    monkeypatch.setattr(
        star_population, "_require_current_gaia_field_sampling",
        lambda _meta, _rows: None,
    )
    star_population._DISTRIBUTION_MEMO.clear()
    yield tmp_path
    star_population._DISTRIBUTION_MEMO.clear()


def test_fit_job_persists_the_default_view_with_the_generated_stars(fitted_inputs):
    """The fit job runs the fit, then persists the default plot view; that
    view (in memory and on disk) carries the generated-star overlay and the
    split keys, not a fit-time payload without synthetic rows."""
    star_population.fit_star_population()
    served = star_population.star_distribution_payload(
        include_training=False, persist=True,
    )

    assert served["synthetic_splits"] == ["test"]
    assert served["training_included"] is False
    assert served["training_catalog_only"] is False
    comparison = served["density_comparison"]
    assert comparison["synthetic_star_count"] == 30
    assert sum(comparison["parameters"]["vis"]["synthetic"]) > 0

    on_disk = json.loads(star_population.star_distribution_path().read_text())
    assert on_disk["synthetic_splits"] == ["test"]
    assert on_disk["density_comparison"]["synthetic_star_count"] == 30
    # A restarted server (no memo) reads the complete view back from disk.
    star_population._DISTRIBUTION_MEMO.clear()
    again = star_population.star_distribution_payload()
    assert again["synthetic_splits"] == ["test"]
    assert again["density_comparison"]["synthetic_star_count"] == 30


def test_fit_alone_leaves_no_plot_cache_without_the_generated_stars(fitted_inputs):
    """The fit writes the candidate only; a GET right after it computes the
    full default view (generated stars included) rather than reading a
    fit-time cache stamped current without them."""
    fit = star_population.fit_star_population()
    target = star_population.star_distribution_path()
    if target.exists():
        cached = json.loads(target.read_text())
        assert "synthetic_splits" in cached, "a cache without the generated stars is stamped current"
    served = star_population.star_distribution_payload()
    assert served["calibration_fingerprint"] == fit["fingerprint"]
    assert served["synthetic_splits"] == ["test"]
    assert served["density_comparison"]["synthetic_star_count"] == 30
