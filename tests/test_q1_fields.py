"""The three Euclid Q1 deep fields: one committed table of centres, and field
labels derived from sky position (the stored EDF-F/EDF-S labels of the old
cutout script were swapped)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from euclid_polish.sky.observation.q1_fields import Q1_FIELDS, q1_field_for

REPO = Path(__file__).parents[1]


def test_field_centres_are_the_q1_deep_fields():
    by_name = {field.name: field for field in Q1_FIELDS}
    assert [field.name for field in Q1_FIELDS] == ["EDF-N", "EDF-S", "EDF-F"]
    assert (by_name["EDF-N"].ra, by_name["EDF-N"].dec) == (269.733, 66.018)
    # Fornax is near the Chandra Deep Field South; EDF-S is further south.
    assert (by_name["EDF-F"].ra, by_name["EDF-F"].dec) == (52.932, -28.088)
    assert (by_name["EDF-S"].ra, by_name["EDF-S"].dec) == (61.241, -48.423)
    assert all(field.radius_deg == 6.0 for field in Q1_FIELDS)


@pytest.mark.parametrize("ra,dec,name", [
    (269.733, 66.018, "EDF-N"),
    (267.42, 64.89, "EDF-N"),
    (53.16, -27.78, "EDF-F"),        # JADES/GOODS-S lies in Fornax
    (61.0, -48.0, "EDF-S"),
    (359.9, 66.0, None),
    (150.1, 2.2, None),              # COSMOS is not a Q1 field
])
def test_field_for_a_position(ra, dec, name):
    assert q1_field_for(ra, dec) == name


def test_ra_wraps():
    assert q1_field_for(269.733 - 360.0, 66.018) == "EDF-N"


def _tuples(path: Path) -> set[tuple]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Tuple) and len(node.elts) >= 3:
            try:
                value = ast.literal_eval(node)
            except ValueError:
                continue
            if any(isinstance(v, str) and v.startswith("EDF-") for v in value):
                found.add(value)
    return found


def test_download_script_uses_the_correct_centres():
    rows = _tuples(REPO / "scripts" / "fasrc_download_euclid_sky_cutouts.py")
    named = {row[0]: row[1:3] for row in rows if isinstance(row[0], str)}
    assert named["EDF-F"] == (52.932, -28.088)
    assert named["EDF-S"] == (61.241, -48.423)


@pytest.mark.parametrize("module", [
    "q1_star_counts.py", "q1_galaxy_counts.py", "q1_stellar_colors.py",
    "q1_galaxy_radius_statistics.py",
])
def test_population_modules_use_the_shared_table(module):
    source = (REPO / "euclid_polish" / "web" / "helpers" / module).read_text()
    assert "q1_fields" in source
    assert "61.241" not in source and "52.932" not in source
