"""Magnitude-window args to ``EuclidCatalog.query_bright_stars``.

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

import euclid_polish.catalog.client as client_module
from euclid_polish.catalog.client import EuclidCatalog
from euclid_polish.config import Config


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

    monkeypatch.setattr(client_module.Euclid, "launch_job_async", fake_async)
    return captured


def _flux_for_mag(mag: float) -> float:
    return 10 ** ((Config.AB_ZP_UJY - mag) / 2.5)


def test_magnitude_min_emits_flux_upper_bound(captured_query):
    cat = EuclidCatalog._unauthenticated()
    cat.query_bright_stars(10, magnitude_min=15.0)
    q = captured_query["query"]
    expected_flux_max = _flux_for_mag(15.0)
    # Pull out the numeric upper bound following 'flux_vis_psf <'
    m = re.search(r"flux_vis_psf\s*<\s*([0-9.eE+\-]+)", q)
    assert m is not None, f"no upper-bound clause in: {q}"
    assert float(m.group(1)) == pytest.approx(expected_flux_max, rel=1e-9)


def test_magnitude_min_and_limit_together(captured_query):
    cat = EuclidCatalog._unauthenticated()
    cat.query_bright_stars(10, magnitude_min=15.0, magnitude_limit=21.0)
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


def test_snr_min_emits_error_ratio_clause(captured_query):
    cat = EuclidCatalog._unauthenticated()
    cat.query_bright_stars(10, snr_min=50)
    q = captured_query["query"]
    assert "fluxerr_vis_psf" in q
    assert re.search(r"flux_vis_psf\s*>\s*50(\.0)?\s*\*\s*fluxerr_vis_psf", q), q


def test_inverted_window_raises():
    cat = EuclidCatalog._unauthenticated()
    # Bright cutoff >= faint cutoff → empty window.
    with pytest.raises(ValueError, match="magnitude_min must be < magnitude_limit"):
        cat.query_bright_stars(10, magnitude_min=22, magnitude_limit=20)


def test_no_magnitude_min_omits_upper_bound(captured_query):
    cat = EuclidCatalog._unauthenticated()
    cat.query_bright_stars(10)
    q = captured_query["query"]
    # No '< <number>' clause for flux_vis_psf when magnitude_min unset.
    assert re.search(r"flux_vis_psf\s*<\s*[0-9]", q) is None, q


def test_unmasked_cut_on_by_default(captured_query):
    """Mask-free (``det_quality_flag = 0``) is the default — the clean point
    sources wanted for ePSF construction (no saturation/blending/bright-star
    masks)."""
    cat = EuclidCatalog._unauthenticated()
    cat.query_bright_stars(10)
    assert re.search(r"det_quality_flag\s*=\s*0", captured_query["query"])


def test_allow_masked_drops_the_cut(captured_query):
    cat = EuclidCatalog._unauthenticated()
    cat.query_bright_stars(10, require_unmasked=False)
    assert "det_quality_flag" not in captured_query["query"]
