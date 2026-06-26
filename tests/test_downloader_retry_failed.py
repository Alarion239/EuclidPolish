"""``download_failed`` flags: honest reporting + ``--retry-failed`` recovery.

A transient TAP/session error during mosaic resolution flags *every* pending
star ``download_failed``. Those stars are then skipped on later runs (correct,
unless the failure was transient) — but the early-return used to omit the
``failed`` count, so a band where ~everything was flagged reported
``downloaded=0, failed=0`` and looked "done". ``retry_failed`` clears the flags
so the stars are re-attempted.
"""

from __future__ import annotations

from euclid_polish.catalog.star_catalog import StarCatalog
from euclid_polish.catalog.downloader import DownloadConfig, EuclidCutoutDownloader

_SIZE = 16


def _seed_all_failed(tmp_path, n):
    """Catalog of ``n`` stars, every one flagged download_failed for VIS@_SIZE,
    with no cutouts on disk."""
    cat = StarCatalog(str(tmp_path))
    stars = []
    for i in range(n):
        s = {"id": i, "ra": 150.0 + i * 1e-3, "dec": 2.0 + i * 1e-3,
             "magnitude": 18.0}
        StarCatalog.set_download_failed(s, _SIZE, band="VIS")
        stars.append(s)
    cat.save({"stars": stars, "next_id": n})
    cfg = DownloadConfig.for_band("VIS", cutout_size=_SIZE)
    return cat, EuclidCutoutDownloader(cat, cfg)


def test_clear_download_failed_primitive():
    s: dict = {}
    StarCatalog.set_download_failed(s, _SIZE, band="VIS")
    assert StarCatalog.is_download_failed(s, _SIZE, band="VIS") is True
    StarCatalog.clear_download_failed(s, _SIZE, band="VIS")
    assert StarCatalog.is_download_failed(s, _SIZE, band="VIS") is False
    # Other bands untouched.
    StarCatalog.set_download_failed(s, _SIZE, band="Y_E")
    StarCatalog.clear_download_failed(s, _SIZE, band="VIS")
    assert StarCatalog.is_download_failed(s, _SIZE, band="Y_E") is True


def test_early_return_reports_failed_count(tmp_path):
    # Every star failed → nothing pending → early return, but it must surface
    # the failed count (not the old misleading failed=0) and download nothing.
    cat, dl = _seed_all_failed(tmp_path, 12)
    result = dl.download(show_progress=False)
    assert result["downloaded"] == 0
    assert result["failed"] == 12
    assert result["valid"] == 0


def test_retry_failed_clears_flags_and_reattempts(tmp_path, monkeypatch):
    cat, dl = _seed_all_failed(tmp_path, 8)

    # Stub the network: pretend no tile covers the stars, so they re-flag failed
    # (the point under test is that retry_failed re-entered them as *pending*,
    # i.e. resolution was attempted at all rather than an instant early return).
    calls = {"resolve": 0}

    def fake_resolve(stars):
        calls["resolve"] += 1
        return {}                       # nothing covered → all re-flag failed
    monkeypatch.setattr(dl, "_resolve_mosaics", fake_resolve)

    result = dl.download(show_progress=False, retry_failed=True)
    assert calls["resolve"] == 1        # we got past the early return
    assert result["downloaded"] == 0
    # Persisted catalog: the stars are re-flagged failed (genuinely uncovered).
    reloaded = cat.load()["stars"]
    assert all(StarCatalog.is_download_failed(s, _SIZE, band="VIS")
               for s in reloaded)


def test_no_retry_takes_early_return_without_resolving(tmp_path, monkeypatch):
    cat, dl = _seed_all_failed(tmp_path, 8)
    called = {"resolve": False}
    monkeypatch.setattr(
        dl, "_resolve_mosaics",
        lambda stars: called.__setitem__("resolve", True) or {})
    dl.download(show_progress=False, retry_failed=False)
    assert called["resolve"] is False   # never resolved — all stayed failed
