"""The star catalog gets a STABLE identity — minted once, reused across the
incremental writes of a download run."""

from __future__ import annotations

import os

import pytest

import euclid_polish.catalog.catalog_object as catalog_mod
from euclid_polish.catalog.catalog_object import CatalogObject
from euclid_polish.config import Config
from euclid_polish.provenance.store import ProvStore


@pytest.fixture(autouse=True)
def _tmp_store(tmp_path, monkeypatch):
    """Mint the catalog id through a store under tmp, not the live data dir."""
    store = ProvStore(str(tmp_path / "prov"))
    monkeypatch.setattr(catalog_mod, "default_store", lambda: store)


def _path(tmp_path):
    return os.path.join(str(tmp_path), Config.CATALOG_FILE)


def test_catalog_identity_is_stable(tmp_path):
    path = _path(tmp_path)
    CatalogObject.write([], path)
    first = CatalogObject.prov_id(path)
    assert first is not None
    assert not first.is_sentinel

    CatalogObject.write([], path)             # write again (incremental)
    assert CatalogObject.prov_id(path) == first   # same id, not a new one


def test_catalog_without_write_has_no_identity(tmp_path):
    assert CatalogObject.prov_id(_path(tmp_path)) is None
