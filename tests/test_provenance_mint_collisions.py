"""Every producer mints through the store, and the store sees every id on disk.

The trainer (checkpoint ``provenance.json``) and the catalog writer
(``<path>.prov.json``) keep their id inside a stamp file rather than in a
sidecar name, so the store must read those files for a fresh mint to avoid
them, and those producers must mint through the store so their ids avoid the
sidecars.
"""

from __future__ import annotations

import json
import os

import pytest
import tensorflow as tf

import euclid_polish.catalog.catalog_object as catalog_mod
import euclid_polish.provenance.defaults as defaults_mod
import euclid_polish.provenance.ids as ids_mod
import euclid_polish.training.trainer as trainer_mod
from euclid_polish.catalog.catalog_object import CatalogObject
from euclid_polish.config import Config
from euclid_polish.provenance.checkpoint import (
    read_checkpoint_provenance,
    write_checkpoint_provenance,
)
from euclid_polish.provenance.ids import ProvId
from euclid_polish.provenance.records import Process, Stamp
from euclid_polish.provenance.store import ProvStore
from euclid_polish.training.trainer import Trainer

_TAKEN = "aaaaaaaa"
_FREE = "bbbbbbbb"


@pytest.fixture
def scripted_draws(monkeypatch):
    """Make ``ProvId.mint`` draw ``_TAKEN`` first, then ``_FREE``."""
    draws = iter([_TAKEN, _FREE, "cccccccc", "dddddddd"])
    monkeypatch.setattr(ids_mod.secrets, "token_hex", lambda _n: next(draws))


def _store_with_taken_sidecar(tmp_path) -> ProvStore:
    store = ProvStore(str(tmp_path / "prov"))
    store.put(Process.generation(id=ProvId(_TAKEN), git=None))
    return store


def _tiny_model():
    inp = tf.keras.Input(shape=(None, None, 1))
    out = tf.keras.layers.Conv2D(1, 3, padding="same")(inp)
    return tf.keras.Model(inp, out)


def test_trainer_checkpoint_id_avoids_existing_sidecar(tmp_path, monkeypatch, scripted_draws):
    store = _store_with_taken_sidecar(tmp_path)
    monkeypatch.setattr(trainer_mod, "default_store", lambda: store)
    ckpt = str(tmp_path / "ckpt")
    tr = Trainer(_tiny_model(), checkpoint_dir=ckpt)
    tr._emit_checkpoint_provenance()
    assert str(read_checkpoint_provenance(ckpt).id) == _FREE


def test_trainer_writes_no_stamp_when_the_store_fails(tmp_path, monkeypatch):
    def _broken():
        raise OSError("store unavailable")

    monkeypatch.setattr(trainer_mod, "default_store", _broken)
    ckpt = str(tmp_path / "ckpt")
    tr = Trainer(_tiny_model(), checkpoint_dir=ckpt)
    tr._emit_checkpoint_provenance()           # best-effort: must not raise
    assert read_checkpoint_provenance(ckpt) is None
    assert tr._model_prov_id is None           # the next save retries


def test_catalog_id_avoids_existing_sidecar(tmp_path, monkeypatch, scripted_draws):
    store = _store_with_taken_sidecar(tmp_path)
    monkeypatch.setattr(catalog_mod, "default_store", lambda: store)
    path = str(tmp_path / "stars.csv")
    CatalogObject.write([], path)
    assert str(CatalogObject.prov_id(path)) == _FREE


def test_store_sees_checkpoint_and_catalog_stamp_ids(tmp_path):
    root = tmp_path / "data"
    write_checkpoint_provenance(str(root / "ckpt" / "m1"), Stamp(id=ProvId("1234abcd")))
    (root / "stars.csv.prov.json").write_text(Stamp(id=ProvId("5678ef01")).to_json())
    # A ``provenance.json`` that is not a stamp (no id) is ignored, not fatal.
    (root / "plates").mkdir()
    (root / "plates" / "provenance.json").write_text(json.dumps({"tiles": []}))

    store = ProvStore(str(tmp_path / "prov"), data_roots=[str(root)])
    assert store.exists(ProvId("1234abcd"))
    assert store.exists(ProvId("5678ef01"))
    assert not store.exists(ProvId("deadbeef"))


def test_store_mint_skips_a_checkpoint_stamp_id(tmp_path, scripted_draws):
    root = tmp_path / "data"
    write_checkpoint_provenance(str(root / "ckpt" / "m1"), Stamp(id=ProvId(_TAKEN)))
    store = ProvStore(str(tmp_path / "prov"), data_roots=[str(root)])
    assert str(store.mint()) == _FREE


def test_store_never_mints_the_same_id_twice(tmp_path, monkeypatch):
    # The same draw twice in a row: the second mint (nothing written yet for
    # the first) must still re-draw.
    draws = iter([_TAKEN, _TAKEN, _FREE])
    monkeypatch.setattr(ids_mod.secrets, "token_hex", lambda _n: next(draws))
    store = ProvStore(str(tmp_path / "prov"))
    first = store.mint()
    second = store.mint()
    assert str(first) == _TAKEN and str(second) == _FREE
    assert os.listdir(str(tmp_path / "prov")) == []


def test_default_store_sees_ensemble_member_stamps(tmp_path, monkeypatch):
    """Members live in ``<ckpt parent>/ensemble``, beside the anchor dir."""
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(Config, "RECORDS_DIR_V2", str(tmp_path / "data" / "records"))
    monkeypatch.setattr(Config, "PROV_DIR", str(tmp_path / "data" / "_prov"))
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR", str(tmp_path / "ckpt" / "wdsr"))
    monkeypatch.setattr(defaults_mod, "_default_store_cache", None)
    member = tmp_path / "ckpt" / "ensemble" / "member_001"
    write_checkpoint_provenance(str(member), Stamp(id=ProvId("1234abcd")))
    assert defaults_mod.default_store().exists(ProvId("1234abcd"))
