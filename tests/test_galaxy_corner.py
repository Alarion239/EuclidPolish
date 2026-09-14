from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from euclid_polish.web.helpers.galaxy_corner import (
    CORNER_VARIABLES,
    build_galaxy_corner,
    mass_fraction_contours,
)
from tests.test_conditional_color_sfr import synthetic_rows, write_fixture_catalog
from tests.test_euclid_galaxy_prior import active_payload


def test_mass_fraction_contours_enclose_the_requested_gaussian_mass():
    centers = np.linspace(-4.0, 4.0, 161)
    xx, yy = np.meshgrid(centers, centers, indexing="ij")
    density = np.exp(-0.5 * (xx**2 + yy**2))

    contours = mass_fraction_contours(
        density, density, centers, centers, (0.5,),
    )

    path = contours[0]["paths"][0]
    radius = np.hypot(path["x"], path["y"])
    assert np.median(radius) == pytest.approx(
        np.sqrt(2.0 * np.log(2.0)), rel=0.03,
    )


def test_corner_places_q1_below_and_model_draws_above_the_diagonal(tmp_path):
    rows = synthetic_rows(n_rows=400, sfr_missing_above_mag=23.0)
    catalog_path, _meta = write_fixture_catalog(tmp_path, rows)

    corner = build_galaxy_corner(catalog_path, active_payload(), model_draws=400)

    assert corner["available"] is True
    assert [variable["key"] for variable in corner["variables"]] == [
        key for key, _label, _unit in CORNER_VARIABLES
    ]
    assert corner["q1_rows"] == 400
    assert corner["model_draws"] == 400
    assert 18.9 < corner["vis_range"][0] < corner["vis_range"][1] < 25.1

    count = len(CORNER_VARIABLES)
    cells = {(cell["row"], cell["col"]): cell for cell in corner["cells"]}
    assert len(cells) == count * (count - 1)
    for (row, col), cell in cells.items():
        assert cell["source"] == ("q1" if row > col else "model")

    sfr_valid = sum(1 for row in rows if row["phz_pp_median_sfr"] != "")
    # VIS × SFR in the Q1 triangle only sees rows with a usable PHZ SFR.
    assert 0.9 * sfr_valid <= cells[(1, 0)]["rows"] <= sfr_valid

    vis_window = corner["variables"][0]["domain"]
    radius_window = corner["variables"][2]["domain"]
    q1_vis_radius = cells[(2, 0)]
    assert q1_vis_radius["contours"]
    for contour in q1_vis_radius["contours"]:
        for path in contour["paths"]:
            assert vis_window[0] <= min(path["x"]) <= max(path["x"]) <= vis_window[1]
            assert radius_window[0] <= min(path["y"]) <= max(path["y"]) <= radius_window[1]

    for diagonal in corner["diagonal"]:
        widths = np.diff(diagonal["edges"])
        for source in ("q1", "model"):
            assert np.sum(np.asarray(diagonal[source]) * widths) == (
                pytest.approx(1.0, abs=1e-3)
            )


def test_corner_requires_a_fitted_model(tmp_path):
    catalog_path, _meta = write_fixture_catalog(
        tmp_path, synthetic_rows(n_rows=80),
    )

    with pytest.raises(ValueError, match="Fit the galaxy population model"):
        build_galaxy_corner(catalog_path, None)


def test_corner_card_is_rendered_on_the_galaxy_page():
    pages = Path(__file__).parents[1] / "euclid_polish/web/frontend/src/pages"
    page = (pages / "GalaxyDistributions.tsx").read_text()
    corner = (pages / "GalaxyCorner.tsx").read_text()

    assert "<GalaxyCorner data={api.corner} />" in page
    assert "Lower triangle" in corner
    assert "Upper triangle" in corner


def test_corner_sfr_window_ignores_the_unphysical_phz_tail(tmp_path):
    rows = synthetic_rows(n_rows=400)
    for row in rows[::20]:
        # The PHZ physical fit reports log SFR down to about -250.
        row["phz_pp_median_sfr"] = float(row["phz_pp_median_stellarmass"]) - 200.0
    catalog_path, _meta = write_fixture_catalog(tmp_path, rows)

    corner = build_galaxy_corner(catalog_path, active_payload(), model_draws=300)

    sfr = corner["variables"][1]
    assert sfr["key"] == "log_sfr"
    assert sfr["domain"][0] > -6.0
    assert sfr["outside_fraction"]["q1"] == pytest.approx(0.05, abs=0.02)
    vis = corner["variables"][0]
    assert vis["domain"][0] < corner["vis_range"][0]
    assert vis["domain"][1] > corner["vis_range"][1]
