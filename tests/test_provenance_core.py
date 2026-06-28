"""Phase 0 — provenance core unit tests.

Pure-Python: no TensorFlow, no numpy, no disk-heavy fixtures beyond tmp_path.
Covers ProvId (mint / collision / format), the record model + Stamp JSON
round-trip, ProvStore (put / get / find / rebuild), and Lineage graph queries.
"""

from __future__ import annotations

import pytest

# --------------------------------------------------------------------------- #
# ProvId
# --------------------------------------------------------------------------- #

def test_provid_accepts_8hex():
    from euclid_polish.provenance.ids import ProvId
    pid = ProvId("4b1e7a90")
    assert str(pid) == "4b1e7a90"


def test_provid_normalises_to_lowercase():
    from euclid_polish.provenance.ids import ProvId
    assert str(ProvId("4B1E7A90")) == "4b1e7a90"


def test_provid_rejects_wrong_length():
    from euclid_polish.provenance.ids import ProvId
    with pytest.raises(ValueError):
        ProvId("4b1e7a9")      # 7 chars
    with pytest.raises(ValueError):
        ProvId("4b1e7a901")    # 9 chars


def test_provid_rejects_non_hex():
    from euclid_polish.provenance.ids import ProvId
    with pytest.raises(ValueError):
        ProvId("zzzzzzzz")


def test_provid_filename_token_is_the_hex():
    from euclid_polish.provenance.ids import ProvId
    assert ProvId("4b1e7a90").as_filename_token() == "4b1e7a90"


def test_provid_equality_and_hash():
    from euclid_polish.provenance.ids import ProvId
    a = ProvId("4b1e7a90")
    b = ProvId("4B1E7A90")
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_provid_sentinel_is_unknown_provenance():
    from euclid_polish.provenance.ids import ProvId
    s = ProvId.sentinel()
    assert str(s) == "00000000"
    assert s.is_sentinel
    assert not ProvId("4b1e7a90").is_sentinel


def test_provid_mint_returns_valid_id():
    from euclid_polish.provenance.ids import ProvId
    pid = ProvId.mint(exists=lambda _id: False)
    assert isinstance(pid, ProvId)
    assert not pid.is_sentinel


def test_provid_mint_redraws_on_collision():
    from euclid_polish.provenance.ids import ProvId
    seen = []

    def exists(pid):
        # Collide on the first candidate only, then accept.
        seen.append(pid)
        return len(seen) == 1

    pid = ProvId.mint(exists=exists)
    assert isinstance(pid, ProvId)
    assert len(seen) >= 2            # proves the redraw loop ran
    assert pid == seen[-1]           # the accepted (non-colliding) candidate


# --------------------------------------------------------------------------- #
# Stamp
# --------------------------------------------------------------------------- #

def test_stamp_json_round_trip():
    from euclid_polish.provenance.ids import ProvId
    from euclid_polish.provenance.records import Stamp
    s = Stamp(
        id=ProvId("4b1e7a90"),
        produced_by=ProvId("7f3a9c21"),
        parents=(ProvId("2b8e44d1"), ProvId("9c1f0a7d")),
        schema_version=3,
        subset="train",
    )
    back = Stamp.from_json(s.to_json())
    assert back == s
    assert back.parents == (ProvId("2b8e44d1"), ProvId("9c1f0a7d"))


def test_stamp_legacy_is_sentinel():
    from euclid_polish.provenance.records import Stamp
    s = Stamp.legacy()
    assert s.id.is_sentinel
    assert s.produced_by is None


# --------------------------------------------------------------------------- #
# ConfigSnapshot
# --------------------------------------------------------------------------- #

def test_config_snapshot_from_dataclass_captures_fields():
    from dataclasses import dataclass

    from euclid_polish.provenance.records import ConfigSnapshot

    @dataclass
    class FakeGenCfg:
        n_galaxies: int = 12
        tng_fraction: float = 1.0

    snap = ConfigSnapshot.from_dataclass(FakeGenCfg())
    assert snap.config_type == "FakeGenCfg"
    assert snap.fields == {"n_galaxies": 12, "tng_fraction": 1.0}
    assert ConfigSnapshot.from_dict(snap.to_dict()) == snap


# --------------------------------------------------------------------------- #
# Records — polymorphic JSON round-trip via record_from_dict
# --------------------------------------------------------------------------- #

def test_generation_run_round_trip():
    from euclid_polish.provenance.ids import ProvId
    from euclid_polish.provenance.records import (
        ConfigSnapshot,
        Process,
        record_from_dict,
    )
    run = Process.generation(
        id=ProvId("7f3a9c21"),
        config=ConfigSnapshot("SkySimulatorConfig", {"n_galaxies": 12}),
        outputs=(ProvId("4b1e7a90"),),
        status="ok",
        created_at="2026-06-25T00:00:00+00:00",
    )
    assert run.kind == "generationrun"
    back = record_from_dict(run.to_dict())
    assert isinstance(back, Process)
    assert back.kind == "generationrun"
    assert back == run
    assert back.config.fields == {"n_galaxies": 12}


def test_checkpoint_artifact_round_trip_preserves_kind():
    from euclid_polish.provenance.ids import ProvId
    from euclid_polish.provenance.records import (
        Artifact,
        Format,
        record_from_dict,
    )
    art = Artifact.checkpoint(
        id=ProvId("2f9c81aa"),
        produced_by=ProvId("2b8e44d1"),
        format=Format.CKPT,
        path="ckpt/wdsr",
        descriptors={"nchan_in": 4, "nchan_out": 4},
        created_at="2026-06-25T00:00:00+00:00",
    )
    d = art.to_dict()
    assert d["kind"] == "checkpointartifact"
    back = record_from_dict(d)
    assert isinstance(back, Artifact)
    assert back.kind == "checkpointartifact"
    assert back == art
    assert back.format is Format.CKPT


# --------------------------------------------------------------------------- #
# ProvStore
# --------------------------------------------------------------------------- #

def _gen_run(store, **kw):
    from euclid_polish.provenance.records import Process
    return Process.generation(id=store.mint(), git=None, **kw)


def test_store_put_get_round_trip(tmp_path):
    from euclid_polish.provenance.store import ProvStore
    store = ProvStore(str(tmp_path))
    run = _gen_run(store, status="ok")
    store.put(run)
    assert store.get(run.id) == run


def test_store_mint_is_unique_and_registers(tmp_path):
    from euclid_polish.provenance.store import ProvStore
    store = ProvStore(str(tmp_path))
    a = _gen_run(store)
    b = _gen_run(store)
    store.put(a)
    store.put(b)
    assert a.id != b.id
    assert store.exists(a.id) and store.exists(b.id)


def test_store_exists_false_before_put(tmp_path):
    from euclid_polish.provenance.ids import ProvId
    from euclid_polish.provenance.store import ProvStore
    store = ProvStore(str(tmp_path))
    assert not store.exists(ProvId("deadbeef"))


def test_store_find_by_kind(tmp_path):
    from euclid_polish.provenance.records import Artifact, Format
    from euclid_polish.provenance.store import ProvStore
    store = ProvStore(str(tmp_path))
    run = _gen_run(store)
    ckpt = Artifact.checkpoint(id=store.mint(), git=None,
                               produced_by=run.id, format=Format.CKPT)
    store.put(run)
    store.put(ckpt)
    found = store.find(kind="checkpointartifact")
    assert [r.id for r in found] == [ckpt.id]


def test_store_rebuild_index_from_disk(tmp_path):
    from euclid_polish.provenance.store import ProvStore
    store = ProvStore(str(tmp_path))
    run = _gen_run(store, status="ok")
    store.put(run)
    # A fresh store over the same dir must rediscover the record by scanning.
    fresh = ProvStore(str(tmp_path))
    assert fresh.get(run.id) == run


def test_store_sidecar_next_to_data(tmp_path):
    from euclid_polish.provenance.store import ProvStore
    index_dir = tmp_path / "_prov"
    data_dir = tmp_path / "records"
    data_dir.mkdir()
    store = ProvStore(str(index_dir), data_roots=[str(data_dir)])
    run = _gen_run(store, status="ok")
    path = store.put(run, sidecar_dir=str(data_dir))
    assert str(data_dir) in path           # written next to the data
    assert store.get(run.id) == run


# --------------------------------------------------------------------------- #
# Lineage
# --------------------------------------------------------------------------- #

def _build_sr_graph(store):
    """A model → inference → SR-cutout chain. Returns (model, infer, sr)."""
    from euclid_polish.provenance.records import Artifact, Format, Process
    train = Process.training(id=store.mint(), git=None, status="ok")
    model = Artifact.checkpoint(id=store.mint(), git=None,
                                produced_by=train.id, format=Format.CKPT)
    infer = Process.inference(id=store.mint(), git=None,
                              inputs=(model.id,), status="ok")
    sr = Artifact.sr_cutout(id=store.mint(), git=None, produced_by=infer.id,
                            format=Format.FITS, parents=(model.id,))
    for r in (train, model, infer, sr):
        store.put(r)
    return model, infer, sr


def test_lineage_ancestors_walks_to_root(tmp_path):
    from euclid_polish.provenance.lineage import Lineage
    from euclid_polish.provenance.store import ProvStore
    store = ProvStore(str(tmp_path))
    model, infer, sr = _build_sr_graph(store)
    anc = Lineage(store).ancestors(sr.id)
    assert model.id in anc
    assert infer.id in anc


def test_lineage_descendants_finds_outputs(tmp_path):
    from euclid_polish.provenance.lineage import Lineage
    from euclid_polish.provenance.store import ProvStore
    store = ProvStore(str(tmp_path))
    model, infer, sr = _build_sr_graph(store)
    desc = Lineage(store).descendants(model.id)
    assert sr.id in desc


def test_lineage_is_stale_current_vs_stale_vs_unknown(tmp_path):
    from euclid_polish.provenance.ids import ProvId
    from euclid_polish.provenance.lineage import Lineage
    from euclid_polish.provenance.records import Artifact, Format
    from euclid_polish.provenance.store import ProvStore
    store = ProvStore(str(tmp_path))
    model, infer, sr = _build_sr_graph(store)
    lin = Lineage(store)
    assert lin.is_stale(sr.id, current_model_id=model.id) is False
    assert lin.is_stale(sr.id, current_model_id=ProvId("aaaaaaaa")) is True
    # An SR with sentinel (unknown) model → unknown, not False.
    legacy_sr = Artifact.sr_cutout(id=store.mint(), git=None, format=Format.FITS,
                                   parents=(ProvId.sentinel(),))
    store.put(legacy_sr)
    assert lin.is_stale(legacy_sr.id, current_model_id=model.id) is None


# --------------------------------------------------------------------------- #
# Persistable contract / StampCarrier mixin
# --------------------------------------------------------------------------- #

def test_stamp_carrier_with_stamp_returns_unmutated_copy():
    from dataclasses import dataclass

    from euclid_polish.provenance.ids import ProvId
    from euclid_polish.provenance.persistable import StampCarrier
    from euclid_polish.provenance.records import Stamp

    @dataclass
    class Thing(StampCarrier):
        x: int = 0      # a required-looking field after the kw-only stamp

    t = Thing(x=5)
    assert t.prov_stamp() is None
    s = Stamp(id=ProvId("4b1e7a90"))
    t2 = t.with_stamp(s)
    assert t2.prov_stamp() == s
    assert t2.x == 5
    assert t.prov_stamp() is None        # original untouched


def test_persistable_protocol_recognises_a_full_implementer():
    from dataclasses import dataclass

    from euclid_polish.provenance.persistable import Persistable, StampCarrier
    from euclid_polish.provenance.records import Format

    @dataclass
    class Doc(StampCarrier):
        PROV_FORMAT = Format.FITS

    assert isinstance(Doc(), Persistable)
