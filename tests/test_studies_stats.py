"""Paired bootstrap intervals and group bands (``euclid_polish/studies/stats.py``)."""
from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.studies import stats


def test_paired_bootstrap_recovers_a_known_shift():
    rng = np.random.default_rng(1)
    base = rng.normal(40.0, 3.0, 200)            # large between-field spread …
    a = base + 0.5 + rng.normal(0, 0.1, 200)     # … but a tight paired difference
    out = stats.paired_bootstrap(a, base, n=2000, seed=0)
    assert out["n_fields"] == 200 and out["n_resamples"] == 2000 and out["seed"] == 0
    assert out["mean"] == pytest.approx(0.5, abs=0.03)
    assert out["lo"] < out["mean"] < out["hi"]
    assert out["hi"] - out["lo"] < 0.06          # pairing removes the field spread


def test_paired_bootstrap_is_deterministic_by_seed():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=30), rng.normal(size=30)
    assert stats.paired_bootstrap(a, b, seed=7) == stats.paired_bootstrap(a, b, seed=7)
    assert stats.paired_bootstrap(a, b, seed=7) != stats.paired_bootstrap(a, b, seed=8)


def test_paired_bootstrap_interval_covers_zero_for_no_difference():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, 100)
    b = a + rng.normal(0, 1, 100)
    out = stats.paired_bootstrap(a, b, seed=0)
    assert out["lo"] < 0 < out["hi"]


def test_paired_bootstrap_drops_unpaired_nans_and_needs_two_fields():
    a = np.array([1.0, 2.0, np.nan, 4.0])
    b = np.array([0.0, 1.0, 1.0, np.nan])
    assert stats.paired_bootstrap(a, b, seed=0)["n_fields"] == 2
    with pytest.raises(ValueError):
        stats.paired_bootstrap([1.0], [0.0])
    with pytest.raises(ValueError):
        stats.paired_bootstrap([1.0, 2.0], [0.0])


def test_group_band_median_and_p16_p84_across_members():
    values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])
    out = stats.group_band(values)
    assert out["median"].tolist() == [3.0, 30.0]
    assert out["lo"].tolist() == pytest.approx(np.percentile(values, 16, axis=0).tolist())
    assert out["hi"].tolist() == pytest.approx(np.percentile(values, 84, axis=0).tolist())
    assert out["n"] == 5


def test_group_labels_by_recipe_field():
    rows = [{"label": "1", "loss": "l1"}, {"label": "2", "loss": "l2"},
            {"label": "3", "loss": "l1"}, {"label": "4", "loss": None}]
    assert stats.groups(rows, "loss") == {"l1": ["1", "3"], "l2": ["2"], "—": ["4"]}
    assert stats.groups(rows, None) == {"1": ["1"], "2": ["2"], "3": ["3"], "4": ["4"]}
