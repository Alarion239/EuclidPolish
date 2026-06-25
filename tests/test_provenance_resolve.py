"""Net-new resolution layer: checkpoint identity + resolve-by-identity.

These touch no existing code — a checkpoint's identity lives in a
`<ckpt_dir>/provenance.json` sidecar, and records resolve by their id-tokenized
filename with a non-breaking fallback to the legacy `<role>_<subset>.tfrecord`.
"""

from __future__ import annotations

from euclid_polish.provenance.ids import ProvId
from euclid_polish.provenance.records import Stamp


# --------------------------------------------------------------------------- #
# Checkpoint identity sidecar
# --------------------------------------------------------------------------- #

def test_checkpoint_provenance_round_trip(tmp_path):
    from euclid_polish.provenance.checkpoint import (
        write_checkpoint_provenance, read_checkpoint_provenance,
        model_id_of_checkpoint,
    )
    ckpt = tmp_path / "wdsr"
    stamp = Stamp(id=ProvId("2f9c81aa"), produced_by=ProvId("2b8e44d1"),
                  parents=(ProvId("4b1e7a90"),), schema_version=3)
    write_checkpoint_provenance(str(ckpt), stamp)
    assert read_checkpoint_provenance(str(ckpt)) == stamp
    assert model_id_of_checkpoint(str(ckpt)) == ProvId("2f9c81aa")


def test_checkpoint_provenance_absent_is_none(tmp_path):
    from euclid_polish.provenance.checkpoint import (
        read_checkpoint_provenance, model_id_of_checkpoint,
    )
    assert read_checkpoint_provenance(str(tmp_path)) is None
    assert model_id_of_checkpoint(str(tmp_path)) is None


# --------------------------------------------------------------------------- #
# resolve_record — id-tokenized preferred, legacy fallback
# --------------------------------------------------------------------------- #

def test_resolve_prefers_id_tokenized(tmp_path):
    from euclid_polish.provenance.resolve import resolve_record
    (tmp_path / "clean_train.4b1e7a90.tfrecord").write_bytes(b"x")
    (tmp_path / "clean_train.tfrecord").write_bytes(b"x")     # legacy also present
    r = resolve_record(str(tmp_path), role="clean", subset="train")
    assert r.path.endswith("clean_train.4b1e7a90.tfrecord")
    assert r.prov_id == ProvId("4b1e7a90")
    assert r.legacy is False


def test_resolve_falls_back_to_legacy(tmp_path):
    from euclid_polish.provenance.resolve import resolve_record
    (tmp_path / "dirty_train.tfrecord").write_bytes(b"x")
    r = resolve_record(str(tmp_path), role="dirty", subset="train")
    assert r.path.endswith("dirty_train.tfrecord")
    assert r.prov_id.is_sentinel
    assert r.legacy is True


def test_resolve_handles_compound_role(tmp_path):
    from euclid_polish.provenance.resolve import resolve_record
    (tmp_path / "dirty_anchor_validate.0a1b2c3d.tfrecord").write_bytes(b"x")
    r = resolve_record(str(tmp_path), role="dirty_anchor", subset="validate")
    assert r.prov_id == ProvId("0a1b2c3d")
    assert r.legacy is False
