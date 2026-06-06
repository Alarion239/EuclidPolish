"""Unit tests for euclid_polish.tng.selection (mode + temperature picks)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from euclid_polish.tng import properties as Pr
from euclid_polish.tng import selection as S


def _props(n: int = 20):
    """id i → mass_stars=i, sfr=(n-i), reff=2i. So the most massive / biggest
    is the highest id; the most star-forming is the lowest id."""
    return {str(i): {"mass_stars": float(i), "sfr": float(n - i),
                     "m_halo": float(i), "reff": float(2 * i)}
            for i in range(n)}


# ---------------------------------------------------------------------------
# deterministic extreme (temperature 0)
# ---------------------------------------------------------------------------

def test_zero_temperature_is_deterministic_extreme():
    p = _props(20)
    assert S.select_ids(p, "most_massive", 5, temperature=0) == \
        ["19", "18", "17", "16", "15"]
    assert S.select_ids(p, "least_massive", 5, temperature=0) == \
        ["0", "1", "2", "3", "4"]
    assert S.select_ids(p, "biggest_radius", 5, temperature=0) == \
        ["19", "18", "17", "16", "15"]
    assert S.select_ids(p, "smallest_radius", 5, temperature=0) == \
        ["0", "1", "2", "3", "4"]
    # sfr = n - i, so the lowest ids are the most star-forming.
    assert S.select_ids(p, "most_star_forming", 5, temperature=0) == \
        ["0", "1", "2", "3", "4"]
    assert S.select_ids(p, "least_star_forming", 5, temperature=0) == \
        ["19", "18", "17", "16", "15"]


# ---------------------------------------------------------------------------
# temperature weighting
# ---------------------------------------------------------------------------

def test_temperature_returns_valid_distinct_set():
    p = _props(40)
    rng = np.random.default_rng(0)
    sel = S.select_ids(p, "most_massive", 5, temperature=0.5, rng=rng)
    assert len(sel) == 5 and len(set(sel)) == 5
    assert all(s in p for s in sel)


def test_temperature_favors_the_extreme():
    """Over many draws the #1-ranked galaxy is picked far more than a mid one."""
    p = _props(80)
    rng = np.random.default_rng(0)
    counts = Counter()
    for _ in range(400):
        counts.update(S.select_ids(p, "most_massive", 5,
                                   temperature=0.3, rng=rng))
    assert counts["79"] > counts["40"]            # most massive >> mid-rank


def test_higher_temperature_spreads_more():
    """A larger temperature samples deeper into the ranking."""
    p = _props(80)
    rng = np.random.default_rng(1)

    def spread(temp):
        seen = set()
        for _ in range(200):
            seen.update(S.select_ids(p, "most_massive", 5,
                                     temperature=temp, rng=rng))
        return len(seen)
    assert spread(0.1) < spread(0.9)


# ---------------------------------------------------------------------------
# random + edge cases
# ---------------------------------------------------------------------------

def test_random_mode_returns_n_distinct():
    p = _props(20)
    sel = S.select_ids(p, "random", 5, rng=np.random.default_rng(1))
    assert len(sel) == 5 and len(set(sel)) == 5 and all(s in p for s in sel)


def test_unknown_mode_is_random():
    p = _props(10)
    sel = S.select_ids(p, "nonsense", 4, rng=np.random.default_rng(2))
    assert len(sel) == 4 and all(s in p for s in sel)


def test_fewer_candidates_than_n():
    p = _props(3)
    sel = S.select_ids(p, "most_massive", 5, temperature=0)
    assert sorted(sel, key=int) == ["0", "1", "2"]


def test_nan_values_excluded():
    p = {"a": {"mass_stars": float("nan"), "sfr": 1, "m_halo": 1, "reff": 1},
         "b": {"mass_stars": 5.0, "sfr": 1, "m_halo": 1, "reff": 1},
         "c": {"mass_stars": 9.0, "sfr": 1, "m_halo": 1, "reff": 1}}
    sel = S.select_ids(p, "most_massive", 5, temperature=0)
    assert sel == ["c", "b"]                       # 'a' (NaN) dropped


def test_all_nan_falls_back_to_random():
    p = {str(i): {"mass_stars": float("nan"), "sfr": float("nan"),
                  "m_halo": 1.0, "reff": float("nan")} for i in range(5)}
    sel = S.select_ids(p, "most_massive", 3, rng=np.random.default_rng(0))
    assert len(sel) == 3 and all(s in p for s in sel)


def test_empty_props_returns_empty():
    assert S.select_ids({}, "random", 5) == []
    assert S.select_ids({}, "most_massive", 5) == []


# ---------------------------------------------------------------------------
# pick_by_mode (reads the local cache)
# ---------------------------------------------------------------------------

def test_pick_by_mode_no_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "local_properties_path",
                        lambda: str(tmp_path / "nope.csv"))
    assert S.pick_by_mode("most_massive", 5) == []


def test_pick_by_mode_reads_cache(monkeypatch, tmp_path):
    path = str(tmp_path / "props.csv")
    Pr._write_cache(path, {
        "1": {"sfr": 1, "mass_stars": 10, "m_halo": 1, "reff": 1},
        "2": {"sfr": 1, "mass_stars": 20, "m_halo": 1, "reff": 1}})
    monkeypatch.setattr(S, "local_properties_path", lambda: path)
    assert S.pick_by_mode("most_massive", 1, temperature=0) == ["2"]
    assert S.pick_by_mode("least_massive", 1, temperature=0) == ["1"]
