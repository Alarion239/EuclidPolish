"""Tests for euclid_polish.catalog.catalog_object.CatalogObject."""
import os

import numpy as np

from euclid_polish.catalog.catalog_object import CatalogObject, next_id


def _obj(i=0, ra=10.0, dec=-5.0, mag=18.0):
    return CatalogObject(ra=ra, dec=dec, id=i, magnitude=mag,
                         flux_psf_uJy=123.0, fluxerr_psf_uJy=4.0)


def test_defaults_and_kind():
    o = CatalogObject(ra=1.0, dec=2.0)
    assert o.kind == "star"
    assert o.id is None
    assert not o.is_valid() and not o.is_corrupted()


def test_flag_set_read_and_supersede():
    o = _obj()
    o.set_valid(64, band="VIS")
    assert o.is_valid(64, "VIS") and o.is_valid(band="VIS")  # size=None → any
    assert o.valid_sizes("VIS") == [64]
    assert o.valid_bands() == ["VIS"]
    # corrupted supersedes valid for that (band,size)
    o.set_corrupted(64, "VIS")
    assert o.is_corrupted(64, "VIS") and not o.is_valid(64, "VIS")
    # a fresh valid clears corrupted again
    o.set_valid(64, "VIS")
    assert o.is_valid(64, "VIS") and not o.is_corrupted(64, "VIS")


def test_download_failed_flag():
    o = _obj()
    o.set_download_failed(64)
    assert o.is_download_failed(64) and o.has_any("download_failed")
    o.clear_download_failed(64)
    assert not o.is_download_failed(64)


def test_to_row_from_row_roundtrip():
    o = _obj(i=3)
    o.set_valid(64, "VIS")
    o.set_corrupted(128, "Y_E")
    back = CatalogObject.from_row(o.to_row())
    assert (back.id, back.ra, back.dec) == (3, 10.0, -5.0)
    assert back.flux_psf_uJy == 123.0
    assert back.is_valid(64, "VIS") and back.is_corrupted(128, "Y_E")


def test_write_read_roundtrip_and_stable_prov(tmp_path):
    path = str(tmp_path / "stars.csv")
    objs = [_obj(0), _obj(1)]
    objs[0].set_valid(64, "VIS")
    CatalogObject.write(objs, path)
    assert os.path.exists(path)
    pid = CatalogObject.prov_id(path)
    assert pid is not None

    back = CatalogObject.read(path)
    assert sorted(o.id for o in back) == [0, 1]
    assert next(o for o in back if o.id == 0).is_valid(64, "VIS")

    # rewriting reuses the same provenance id
    CatalogObject.write(back, path)
    assert CatalogObject.prov_id(path) == pid


def test_read_missing_file_is_empty(tmp_path):
    assert CatalogObject.read(str(tmp_path / "nope.csv")) == []


def test_next_id():
    assert next_id([]) == 0
    assert next_id([_obj(0), _obj(4), _obj(2)]) == 5
