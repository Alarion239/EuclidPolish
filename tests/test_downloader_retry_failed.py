"""``download_failed`` flags: honest reporting + ``--retry-failed`` recovery.

A transient TAP/session error during mosaic resolution flags *every* pending
star ``download_failed``. Those stars are then skipped on later runs (correct,
unless the failure was transient) — but the early-return used to omit the
``failed`` count, so a band where ~everything was flagged reported
``downloaded=0, failed=0`` and looked "done". ``retry_failed`` clears the flags
so the stars are re-attempted.
"""

from __future__ import annotations

import os

from euclid_polish.config import Config
from euclid_polish.catalog.catalog_object import CatalogObject
from euclid_polish.catalog.client import EuclidCatalog
from euclid_polish.catalog.downloader import DownloadConfig

_SIZE = 16


def _seed_all_failed(tmp_path, n):
    """Catalog of ``n`` objects, every one flagged download_failed for VIS@_SIZE,
    with no cutouts on disk. Returns ``(output_dir, objects, EuclidCatalog)``."""
    output_dir = str(tmp_path)
    objects = []
    for i in range(n):
        o = CatalogObject(ra=150.0 + i * 1e-3, dec=2.0 + i * 1e-3, id=i,
                          magnitude=18.0)
        o.set_download_failed(_SIZE, band="VIS")
        objects.append(o)
    CatalogObject.write(objects, os.path.join(output_dir, Config.CATALOG_FILE))
    return output_dir, objects, EuclidCatalog._unauthenticated()


def _cfg():
    return DownloadConfig.for_band("VIS", cutout_size=_SIZE)


def test_clear_download_failed_primitive():
    o = CatalogObject(ra=1.0, dec=2.0)
    o.set_download_failed(_SIZE, band="VIS")
    assert o.is_download_failed(_SIZE, band="VIS") is True
    o.clear_download_failed(_SIZE, band="VIS")
    assert o.is_download_failed(_SIZE, band="VIS") is False
    # Other bands untouched.
    o.set_download_failed(_SIZE, band="Y_E")
    o.clear_download_failed(_SIZE, band="VIS")
    assert o.is_download_failed(_SIZE, band="Y_E") is True


def test_early_return_reports_failed_count(tmp_path):
    # Every object failed → nothing pending → early return, but it must surface
    # the failed count (not the old misleading failed=0) and download nothing.
    output_dir, objects, cat = _seed_all_failed(tmp_path, 12)
    result = cat.download_cutouts(objects, output_dir, _cfg(), show_progress=False)
    assert result["downloaded"] == 0
    assert result["failed"] == 12
    assert result["valid"] == 0


def test_retry_failed_clears_flags_and_reattempts(tmp_path, monkeypatch):
    output_dir, objects, cat = _seed_all_failed(tmp_path, 8)

    # Stub the network: pretend no tile covers the stars, so they re-flag failed
    # (the point under test is that retry_failed re-entered them as *pending*,
    # i.e. resolution was attempted at all rather than an instant early return).
    calls = {"resolve": 0}

    def fake_resolve(objs, config):
        calls["resolve"] += 1
        return {}                       # nothing covered → all re-flag failed
    monkeypatch.setattr(cat, "_resolve_mosaics", fake_resolve)

    result = cat.download_cutouts(objects, output_dir, _cfg(),
                                  show_progress=False, retry_failed=True)
    assert calls["resolve"] == 1        # we got past the early return
    assert result["downloaded"] == 0
    # Persisted catalog: the objects are re-flagged failed (genuinely uncovered).
    reloaded = CatalogObject.read(os.path.join(output_dir, Config.CATALOG_FILE))
    assert all(o.is_download_failed(_SIZE, band="VIS") for o in reloaded)


def test_no_retry_takes_early_return_without_resolving(tmp_path, monkeypatch):
    output_dir, objects, cat = _seed_all_failed(tmp_path, 8)
    called = {"resolve": False}
    monkeypatch.setattr(
        cat, "_resolve_mosaics",
        lambda objs, config: called.__setitem__("resolve", True) or {})
    cat.download_cutouts(objects, output_dir, _cfg(),
                         show_progress=False, retry_failed=False)
    assert called["resolve"] is False   # never resolved — all stayed failed
