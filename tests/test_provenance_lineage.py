"""End-to-end lineage + staleness over a realistic generate→train→SR chain,
and the `prov` CLI that surfaces it.
"""

from __future__ import annotations

from euclid_polish.provenance.ids import ProvId
from euclid_polish.provenance.lineage import Lineage
from euclid_polish.provenance.store import ProvStore
from euclid_polish.provenance.records import (
    GenerationRun, TrainingRun, InferenceRun,
    SkyTFRecordArtifact, CheckpointArtifact, SRCutoutArtifact, Format,
)


def _build_chain(store, *, model_id=None):
    """Persist a generate→convolve→train→super-resolve chain. Returns ids."""
    gen = GenerationRun(id=store.mint(), git=None, status="ok")
    clean = SkyTFRecordArtifact(id=store.mint(), git=None, produced_by=gen.id,
                                format=Format.TFRECORD)
    fwd = GenerationRun(id=store.mint(), git=None, parents=(clean.id,), status="ok")
    dirty = SkyTFRecordArtifact(id=store.mint(), git=None, produced_by=fwd.id,
                                format=Format.TFRECORD, parents=(clean.id,))
    train = TrainingRun(id=store.mint(), git=None, inputs=(dirty.id,), status="ok")
    model = CheckpointArtifact(id=model_id or store.mint(), git=None,
                               produced_by=train.id, format=Format.CKPT)
    infer = InferenceRun(id=store.mint(), git=None, inputs=(model.id, dirty.id),
                         status="ok")
    sr = SRCutoutArtifact(id=store.mint(), git=None, produced_by=infer.id,
                          format=Format.FITS, parents=(model.id, dirty.id))
    for r in (gen, clean, fwd, dirty, train, model, infer, sr):
        store.put(r)
    return dict(gen=gen.id, clean=clean.id, dirty=dirty.id, train=train.id,
                model=model.id, infer=infer.id, sr=sr.id)


def test_sr_is_stale_after_retraining(tmp_path):
    store = ProvStore(str(tmp_path))
    ids = _build_chain(store)
    lin = Lineage(store)

    # Fresh: SR was made by the current model.
    assert lin.is_stale(ids["sr"], current_model_id=ids["model"]) is False

    # Retrain: a new model exists → the old SR is now stale.
    new_train = TrainingRun(id=store.mint(), git=None, status="ok")
    new_model = CheckpointArtifact(id=store.mint(), git=None,
                                   produced_by=new_train.id, format=Format.CKPT)
    store.put(new_train)
    store.put(new_model)
    assert lin.is_stale(ids["sr"], current_model_id=new_model.id) is True


def test_ancestors_reach_generation_root(tmp_path):
    store = ProvStore(str(tmp_path))
    ids = _build_chain(store)
    anc = Lineage(store).ancestors(ids["sr"])
    for key in ("gen", "clean", "dirty", "train", "model", "infer"):
        assert ids[key] in anc, f"{key} missing from SR ancestry"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def test_cli_ancestors_lists_the_model(tmp_path, capsys):
    from euclid_polish.provenance.cli import main
    store = ProvStore(str(tmp_path))
    ids = _build_chain(store)
    rc = main(["ancestors", str(ids["sr"])], store=store)
    out = capsys.readouterr().out
    assert rc == 0
    assert str(ids["model"]) in out


def test_cli_stale_flags_old_sr(tmp_path, capsys):
    from euclid_polish.provenance.cli import main
    store = ProvStore(str(tmp_path))
    ids = _build_chain(store)
    new_model = ProvId("aaaaaaaa")
    rc = main(["stale", "--model", str(new_model)], store=store)
    out = capsys.readouterr().out
    assert rc == 0
    assert str(ids["sr"]) in out      # the SR is listed as stale
