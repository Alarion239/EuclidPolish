"""The Q1/DR1 switch of the population calibration (calibration_release)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.table import MaskedColumn, Table

from euclid_polish.config import Config
from euclid_polish.sky.observation import calibration_release
from euclid_polish.sky.observation.calibration_release import (
    RELEASE_ENV,
    active_release,
    population_dir,
)
from euclid_polish.web.helpers import (
    population_calibration,
    population_comparison,
    q1_star_counts,
    q1_stellar_colors,
)


@pytest.fixture
def dr1(monkeypatch):
    monkeypatch.setenv(RELEASE_ENV, "dr1")
    return active_release()


def test_q1_is_the_default(monkeypatch, tmp_path):
    monkeypatch.delenv(RELEASE_ENV, raising=False)
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    release = active_release()
    assert release.name == "q1"
    assert release.table("mer") == "catalogue.mer_catalogue"
    assert release.area_deg2 == pytest.approx(63.1)
    predicate = release.footprint_predicate("mer")
    assert predicate.count("CONTAINS") == 3 and "61.241" in predicate
    assert population_dir() == tmp_path / "population_comparison"
    assert population_calibration.calibration_dir() == (
        tmp_path / "population_comparison" / "calibrations")
    assert not release.split_outer_joins


def test_dr1_uses_the_frozen_wide_tile_sample(dr1, monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    payload = json.loads(calibration_release.DR1_TILES_PATH.read_text())
    tiles = payload["tiles"]
    assert dr1.name == "dr1"
    assert dr1.table("mer") == "dr1.mer_catalogue_wide_survey"
    assert dr1.table("phz_class") == "dr1.phz_classification_wide"
    assert len(dr1.tiles) == len(tiles) == len({t["tile"] for t in tiles})
    assert dr1.area_deg2 == pytest.approx(0.25 * len(tiles))
    predicate = dr1.footprint_predicate("mer")
    assert predicate.startswith("mer.tile_index IN (") and "CONTAINS" not in predicate
    assert all(tile["rows"] > 0 for tile in tiles)
    # Every output of a DR1 run lands beside, never on top of, the Q1 files.
    assert population_dir() == tmp_path / "population_comparison" / "dr1"
    assert q1_star_counts.q1_star_counts_path().parent.name == "dr1"
    assert population_comparison.euclid_catalog_path().parent.name == "dr1"
    assert population_calibration.calibration_dir().parent.name == "dr1"
    assert len(q1_stellar_colors.stellar_color_fields()) == (
        calibration_release.DR1_COLOR_FIELD_COUNT)
    assert dr1.irsa_tables["mer"] == "euclid_dr1_mer_catalogue_wide_survey"
    assert dr1.split_outer_joins


def test_unknown_release_is_refused():
    with pytest.raises(ValueError, match="unknown calibration release"):
        calibration_release.release("dr2")


class _Job:
    def __init__(self, row):
        self.row = row

    def get_results(self):
        return [self.row]


def test_dr1_star_counts_query_dr1_tables_and_normalise_by_its_area(
    dr1, monkeypatch, tmp_path,
):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    queries: list[str] = []
    rows = iter([
        {"selected_point_sources": 10, "expected_point_sources": 9.0,
         "point_source_variance": 0.5},
        {"classified_rows": 10, "expected_stars": 6.0, "classification_variance": 1.0},
    ])

    def launch(query):
        queries.append(query)
        return _Job(next(rows))

    monkeypatch.setattr(q1_star_counts.Euclid, "launch_job_async", launch)
    result = q1_star_counts.query_q1_phz_star_counts(bright=20.0, faint=20.1, bin_width=0.1)
    assert all("dr1.mer_catalogue_wide_survey" in q and "mer.tile_index IN (" in q
               for q in queries)
    assert "JOIN dr1.phz_classification_wide AS cls" in queries[1]
    assert result["release"] == "dr1"
    assert result["footprint_area_deg2"] == pytest.approx(dr1.area_deg2)
    assert result["bins"][0]["density_arcmin2_mag"] == pytest.approx(
        6.0 / dr1.area_arcmin2 / 0.1)
    assert (tmp_path / "population_comparison" / "dr1" / "q1_phz_star_counts.json").is_file()
    assert not (tmp_path / "population_comparison" / "q1_phz_star_counts.json").exists()
    # The DR1 artifact validates under DR1 and is refused as a Q1 one.
    assert q1_star_counts.read_q1_phz_star_counts(bright=20.0, faint=20.1)
    monkeypatch.delenv(RELEASE_ENV)
    monkeypatch.setattr(q1_star_counts, "q1_star_counts_path", lambda: Path(
        tmp_path / "population_comparison" / "dr1" / "q1_phz_star_counts.json"))
    with pytest.raises(ValueError, match="incompatible"):
        q1_star_counts.read_q1_phz_star_counts(bright=20.0, faint=20.1)


def test_outer_join_in_parts_matches_a_left_outer_join(dr1, monkeypatch):
    queries: list[str] = []
    mer = Table({"object_id": np.array([1, 2, 3], dtype=np.int64),
                 "flux_vis_psf": np.array([1.0, 2.0, 3.0])})
    cls = Table({"object_id": np.array([1, 3], dtype=np.int64),
                 "phz_gal_prob": np.array([0.9, 0.2])})
    phys = Table({"object_id": np.array([2], dtype=np.int64),
                  "phz_phys_flags": np.array([7])})

    def launch(query, _relogin):
        queries.append(query)
        if " JOIN " not in query:
            return mer
        return cls if " AS cls " in query else phys

    monkeypatch.setattr(population_comparison, "_launch_population_query", launch)
    result = population_comparison._outer_join_in_parts(
        dr1,
        "mer.object_id, mer.flux_vis_psf,\n cls.phz_gal_prob,\n"
        " phys.phys_param_flags AS phz_phys_flags",
        "mer.flux_vis_psf > 0",
        50,
        None,
    )
    assert len(queries) == 3
    assert all("ORDER BY mer.object_id" in q and q.startswith("SELECT TOP 50") for q in queries)
    assert "LEFT OUTER" not in " ".join(queries)
    assert "JOIN dr1.phz_classification_wide AS cls" in queries[1]
    result.sort("object_id")
    assert list(result["object_id"]) == [1, 2, 3]
    gal = result["phz_gal_prob"]
    assert isinstance(gal, MaskedColumn) and list(gal.mask) == [False, True, False]
    assert gal[0] == pytest.approx(0.9)
    flags = result["phz_phys_flags"]
    assert list(flags.mask) == [True, False, True] and flags[1] == 7
