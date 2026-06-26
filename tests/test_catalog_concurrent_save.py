"""Concurrent-save safety for ``StarCatalog`` (parallel band downloads).

Regression for the bug where ``scripts/download_all_bands.py`` ran all four
bands' ``download()`` at once. Each band loads, mutates its band's flags, and
``save()``s the shared ``stars.csv``. The old ``save`` wrote a *fixed*
``stars.csv.tmp``; two bands writing it at once truncated the catalog, and a
later band then ``load()``ed a short catalog and decided it had nothing left to
download — the silent ``downloaded=0`` for every NISP band.

These tests hammer ``save``/``load`` from many threads and assert the catalog
is never observed truncated.
"""

from __future__ import annotations

import threading

import pytest

from euclid_polish.catalog.star_catalog import StarCatalog


def _catalog(n_stars: int) -> dict:
    return {
        "stars": [
            {"id": i, "ra": 10.0 + i * 1e-4, "dec": -5.0 + i * 1e-4,
             "magnitude": 18.0 + (i % 7) * 0.1}
            for i in range(n_stars)
        ],
        "next_id": n_stars,
    }


def test_concurrent_saves_never_truncate(tmp_path):
    N = 500
    cat = StarCatalog(str(tmp_path))
    cat.save(_catalog(N))                      # seed the file

    errors: list[str] = []
    stop = threading.Event()

    def writer():
        # Each writer saves a full N-star catalog repeatedly.
        full = _catalog(N)
        for _ in range(40):
            if stop.is_set():
                return
            cat.save(full)

    def reader():
        # A reader must NEVER see a short/garbled catalog while writers run.
        for _ in range(120):
            if stop.is_set():
                return
            loaded = cat.load()
            k = len(loaded["stars"])
            if k != N:
                errors.append(f"observed {k} stars (expected {N})")
                stop.set()
                return

    threads = [threading.Thread(target=writer) for _ in range(4)]
    threads += [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors[:5]
    assert len(cat.load()["stars"]) == N


def test_no_stray_tmp_files_left_behind(tmp_path):
    cat = StarCatalog(str(tmp_path))
    full = _catalog(50)

    threads = [threading.Thread(target=lambda: [cat.save(full) for _ in range(20)])
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], leftovers
    assert len(cat.load()["stars"]) == 50


def test_save_lock_is_process_wide(tmp_path):
    # Separate StarCatalog instances (one per band) point at the same file, so
    # the lock that serialises them must be shared, not per-instance.
    a = StarCatalog(str(tmp_path))
    b = StarCatalog(str(tmp_path))
    assert a._save_lock is b._save_lock
