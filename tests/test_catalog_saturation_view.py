"""Per-band saturation view for the Catalog page.

Validates the cutoff heuristic and that the 2×2 valid/invalid figure renders
for the configured bands without needing a real catalog on disk.
"""

from __future__ import annotations

import io

import numpy as np

from euclid_polish.catalog.catalog_object import CatalogObject
from euclid_polish.config import Config
from euclid_polish.web.helpers.sky_render import (
    _render_saturation_view,
    _saturation_cutoff,
)


def _star(mag, per_band):
    """per_band: {band: 'valid'|'corrupted'} → CatalogObject with those flags."""
    o = CatalogObject(ra=150.0, dec=2.0, magnitude=mag)
    for band, kind in per_band.items():
        if kind == "valid":
            o.set_valid(511, band=band)
        else:
            o.set_corrupted(511, band=band)
    return o


def test_cutoff_finds_bright_saturation_boundary():
    # faint stars valid, bright (< 15) saturated
    valid = list(np.linspace(15.0, 22.0, 200))
    invalid = list(np.linspace(10.0, 14.9, 200))
    bins = np.linspace(10, 22, 30)
    cut = _saturation_cutoff(valid, invalid, bins)
    assert cut is not None
    assert 14.0 <= cut <= 16.0


def test_cutoff_none_when_no_saturation():
    bins = np.linspace(10, 22, 30)
    assert _saturation_cutoff([12, 15, 18, 20], [], bins) is None


def test_render_saturation_view_all_bands():
    rng = np.random.default_rng(0)
    cutoffs = {"VIS": 15.0, "Y_E": 16.5, "J_E": 17.0, "H_E": 17.5}
    stars = []
    for _ in range(800):
        m = float(rng.uniform(10, 22))
        per_band = {b: ("corrupted" if m < cutoffs[b] else "valid")
                    for b in cutoffs}
        stars.append(_star(m, per_band))

    fig = _render_saturation_view(stars)
    # one visible panel per configured band
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) >= len([b.name for b in Config.BANDS])
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    assert buf.getvalue()[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_saturation_view_handles_empty_band():
    # a star with no processed flag in any band → "no processed stars" panel
    fig = _render_saturation_view([CatalogObject(ra=150.0, dec=2.0, magnitude=18.0)])
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    assert buf.getvalue()[:8] == b"\x89PNG\r\n\x1a\n"
