"""StarCatalog tolerates an empty / headerless stars.csv.

A truncated or half-written catalog used to crash pd.read_csv with
EmptyDataError, 500-ing every page that reads the catalog summary
(/catalog, etc.). load() now degrades to an empty catalog.
"""

from __future__ import annotations

from euclid_polish.catalog.star_catalog import StarCatalog


def test_load_empty_file_returns_empty_catalog(tmp_path):
    cat = StarCatalog(str(tmp_path))
    open(cat.catalog_path, "w").close()          # 0-byte file
    assert cat.exists() is True
    assert cat.load() == {"stars": [], "next_id": 0}


def test_get_summary_on_empty_file(tmp_path):
    cat = StarCatalog(str(tmp_path))
    open(cat.catalog_path, "w").close()
    summary = cat.get_summary()
    assert summary["total"] == 0
    assert summary["valid"] == 0


def test_header_only_file_is_empty(tmp_path):
    cat = StarCatalog(str(tmp_path))
    with open(cat.catalog_path, "w") as f:
        f.write("id,ra,dec,magnitude\n")          # header, no rows
    assert cat.load()["stars"] == []
