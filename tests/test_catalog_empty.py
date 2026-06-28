"""CatalogObject.read tolerates an empty / headerless stars.csv.

A truncated or half-written catalog used to crash pd.read_csv with
EmptyDataError, 500-ing every page that reads the catalog summary
(/catalog, etc.). read() now degrades to an empty list.
"""

from __future__ import annotations

import os

from euclid_polish.config import Config
from euclid_polish.catalog.catalog_object import CatalogObject, summarize


def _path(tmp_path):
    return os.path.join(str(tmp_path), Config.CATALOG_FILE)


def test_read_empty_file_returns_empty_list(tmp_path):
    path = _path(tmp_path)
    open(path, "w").close()                       # 0-byte file
    assert os.path.exists(path) is True
    assert CatalogObject.read(path) == []


def test_summarize_on_empty_file(tmp_path):
    path = _path(tmp_path)
    open(path, "w").close()
    summary = summarize(CatalogObject.read(path))
    assert summary["total"] == 0
    assert summary["valid"] == 0


def test_header_only_file_is_empty(tmp_path):
    path = _path(tmp_path)
    with open(path, "w") as f:
        f.write("id,ra,dec,magnitude\n")          # header, no rows
    assert CatalogObject.read(path) == []
