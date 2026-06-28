"""Concurrent-write safety for ``CatalogObject`` (parallel band downloads).

Regression for the bug where ``scripts/download_all_bands.py`` ran all four
bands' download at once. Each band reads, mutates its band's flags, and writes
the shared ``stars.csv``. A non-atomic write (fixed ``stars.csv.tmp`` shared by
two bands at once) truncated the catalog, and a later band then read a short
catalog and decided it had nothing left to download — the silent
``downloaded=0`` for every NISP band.

These tests hammer ``write``/``read`` from many threads and assert the catalog
is never observed truncated.
"""

from __future__ import annotations

import os
import threading

from euclid_polish.config import Config
from euclid_polish.catalog import catalog_object as co
from euclid_polish.catalog.catalog_object import CatalogObject


def _objects(n_stars: int):
    return [CatalogObject(ra=10.0 + i * 1e-4, dec=-5.0 + i * 1e-4, id=i,
                          magnitude=18.0 + (i % 7) * 0.1)
            for i in range(n_stars)]


def _path(tmp_path):
    return os.path.join(str(tmp_path), Config.CATALOG_FILE)


def test_concurrent_writes_never_truncate(tmp_path):
    N = 500
    path = _path(tmp_path)
    CatalogObject.write(_objects(N), path)         # seed the file

    errors: list[str] = []
    stop = threading.Event()

    def writer():
        full = _objects(N)
        for _ in range(40):
            if stop.is_set():
                return
            CatalogObject.write(full, path)

    def reader():
        for _ in range(120):
            if stop.is_set():
                return
            k = len(CatalogObject.read(path))
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
    assert len(CatalogObject.read(path)) == N


def test_no_stray_tmp_files_left_behind(tmp_path):
    path = _path(tmp_path)
    full = _objects(50)

    threads = [threading.Thread(target=lambda: [CatalogObject.write(full, path)
                                                for _ in range(20)])
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], leftovers
    assert len(CatalogObject.read(path)) == 50


def test_write_lock_is_module_wide():
    # All writes go through one process-wide lock, so parallel band writers
    # (separate call sites pointing at the same file) are serialised.
    assert isinstance(co._write_lock, type(threading.Lock()))
