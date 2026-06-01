"""Magnitude-window args to ``StarCatalog.query_brightest_stars``.

We can't easily hit the live Euclid TAP service in unit tests, so we
monkey-patch ``Euclid.launch_job_async`` and inspect the ADQL query
that was sent — confirming that the new ``magnitude_min`` parameter
appears as a flux UPPER bound (i.e. excludes BRIGHT stars).
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest
from astropy.table import Table

from euclid_polish.config import Config
from euclid_polish.euclid import catalog as catalog_module
from euclid_polish.euclid.catalog import StarCatalog


class _FakeJob:
    def __init__(self, table: Table):
        self._table = table

    def get_results(self) -> Table:
        return self._table


@pytest.fixture
def captured_query(monkeypatch):
    """Patch ``Euclid.launch_job_async`` to record the ADQL string and
    return an empty result. Tests inspect the captured query."""
    captured = {}

    def fake_async(query, *args, **kwargs):
        captured["query"] = query
        # Return an empty (but valid) astropy Table so the wrapper
        # exits cleanly.
        empty = Table(
            names=("right_ascension", "declination",
                   "flux_vis_psf", "fluxerr_vis_psf"),
            dtype=("f8", "f8", "f8", "f8"),
        )
        return _FakeJob(empty)

    monkeypatch.setattr(catalog_module.Euclid, "launch_job_async", fake_async)
    return captured


def _flux_for_mag(mag: float) -> float:
    return 10 ** ((Config.AB_ZP_UJY - mag) / 2.5)


def test_magnitude_min_emits_flux_upper_bound(tmp_path, captured_query):
    cat = StarCatalog(str(tmp_path))
    cat.query_brightest_stars(num_stars=10, magnitude_min=15.0)
    q = captured_query["query"]
    expected_flux_max = _flux_for_mag(15.0)
    # Pull out the numeric upper bound following 'flux_vis_psf <'
    m = re.search(r"flux_vis_psf\s*<\s*([0-9.eE+\-]+)", q)
    assert m is not None, f"no upper-bound clause in: {q}"
    assert float(m.group(1)) == pytest.approx(expected_flux_max, rel=1e-9)


def test_magnitude_min_and_limit_together(tmp_path, captured_query):
    cat = StarCatalog(str(tmp_path))
    cat.query_brightest_stars(num_stars=10, magnitude_min=15.0, magnitude_limit=21.0)
    q = captured_query["query"]
    # Lower bound must come from magnitude_limit=21 (faint-end cutoff).
    m_lo = re.search(r"flux_vis_psf\s*>\s*([0-9.eE+\-]+)", q)
    m_hi = re.search(r"flux_vis_psf\s*<\s*([0-9.eE+\-]+)", q)
    assert m_lo and m_hi
    # The "> 0" guard appears first in the WHERE; pick the one that matches
    # magnitude 21 (not 0).
    candidates = re.findall(r"flux_vis_psf\s*>\s*([0-9.eE+\-]+)", q)
    flux_lo_candidates = [float(c) for c in candidates]
    expected_lo = _flux_for_mag(21.0)
    assert any(abs(c - expected_lo) / expected_lo < 1e-9 for c in flux_lo_candidates)
    # Upper bound matches magnitude 15.
    assert float(m_hi.group(1)) == pytest.approx(_flux_for_mag(15.0), rel=1e-9)


def test_snr_min_emits_error_ratio_clause(tmp_path, captured_query):
    cat = StarCatalog(str(tmp_path))
    cat.query_brightest_stars(num_stars=10, snr_min=50)
    q = captured_query["query"]
    assert "fluxerr_vis_psf" in q
    assert re.search(r"flux_vis_psf\s*>\s*50(\.0)?\s*\*\s*fluxerr_vis_psf", q), q


def test_inverted_window_raises(tmp_path):
    cat = StarCatalog(str(tmp_path))
    # Bright cutoff >= faint cutoff → empty window.
    with pytest.raises(ValueError, match="window would be empty"):
        cat.query_brightest_stars(num_stars=10, magnitude_min=22, magnitude_limit=20)


def test_no_magnitude_min_omits_upper_bound(tmp_path, captured_query):
    cat = StarCatalog(str(tmp_path))
    cat.query_brightest_stars(num_stars=10)
    q = captured_query["query"]
    # No '< <number>' clause for flux_vis_psf when magnitude_min unset.
    assert re.search(r"flux_vis_psf\s*<\s*[0-9]", q) is None, q
